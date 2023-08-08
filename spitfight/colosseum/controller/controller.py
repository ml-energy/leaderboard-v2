from __future__ import annotations

import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Literal, Optional, TYPE_CHECKING

from pytz import timezone
from pydantic import BaseModel, Field
from text_generation.errors import OverloadedError, ValidationError
from fastapi.exceptions import HTTPException

from spitfight.log import get_logger
from spitfight.utils import BoundedExpiringDict
from spitfight.colosseum.controller.worker import WorkerService
from spitfight.prompt import get_system_prompt, add_system_prompt

if TYPE_CHECKING:
    from spitfight.colosseum.controller.router import ControllerConfig

controller_logger = get_logger(__name__)
request_logger = get_logger("colosseum_requests")


def now() -> datetime:
    return datetime.now(tz=timezone("US/Eastern"))


# Internal states
# The two "chose_*" stages are both the result of voting on a response.
# A normal user will sequentially go through either
#   "prompted" -> "chose_less_energy_response", or
#   "prompted" -> "chose_more_energy_response" -> "voted_energy"
UserStage = Literal[
    "prompted",
    "chose_less_energy_response",
    "chose_more_energy_response",
    "voted_energy",
]


class RequestState(BaseModel):
    """Models the state of a Colosseum play.

    This model is also serialized as is and logged.
    """
    request_id: str
    prompt: str
    model_names: list[str]
    responses: list[str] = ["EMPTY", "EMPTY"]
    energy_consumptions: list[float] = [-1.0, -1.0]
    response_victory_index: Optional[Literal[0, 1]] = None
    energy_victory_index: Optional[Literal[0, 1]] = None

    # The time when the user's stage changed.
    _timestamp: datetime = Field(default_factory=now)
    # The user's current stage.
    _user_stage: UserStage = "prompted"
    # When the the user is not going through the aforementioned stages,
    # the user's stage transition is recorded here.
    _abnormal_stage_change: list[tuple[UserStage, UserStage]] = []

    def set_response_and_energy(self, model_index: Literal[0, 1], response: str, energy_consumption: float) -> None:
        self._timestamp = now()
        self.energy_consumptions[model_index] = energy_consumption
        self.responses[model_index] = response

    def set_response_vote(self, victory_index: Literal[0, 1]) -> None:
        self._timestamp = now()

        # Next stage depends on the user's vote.
        energy_a, energy_b = self.energy_consumptions
        if (victory_index == 0 and energy_a <= energy_b) or (victory_index == 1 and energy_a >= energy_b):
            next_stage = "chose_less_energy_response"
        else:
            next_stage = "chose_more_energy_response"

        # Detect abnormal stage change.
        if self._user_stage != "prompted":
            self._abnormal_stage_change.append((self._user_stage, next_stage))

        self._user_stage = next_stage
        self.response_victory_index = victory_index

    def set_energy_vote(self, victory_index: Literal[0, 1]) -> None:
        self._timestamp = now()

        # Detect abnormal stage change.
        if self._user_stage != "chose_more_energy_response":
            self._abnormal_stage_change.append((self._user_stage, "voted_energy"))

        self._user_stage = "voted_energy"
        self.energy_victory_index = victory_index


class Controller:
    def __init__(
        self,
        max_new_tokens: int,
        background_task_interval: int,
        max_num_req_states: int,
        req_state_expiration_time: int,
        worker_service: WorkerService,
    ):
        self.max_new_tokens = max_new_tokens
        self.max_num_req_states = max_num_req_states
        self.request_states: BoundedExpiringDict[str, RequestState] = \
            BoundedExpiringDict(req_state_expiration_time)
        self.worker_service = worker_service

        self.background_task_handle = asyncio.create_task(
            self._background_task(background_task_interval),
        )

    def shutdown(self) -> None:
        """Shutdown the controller."""
        self.background_task_handle.cancel()

    async def _background_task(self, heartbeat_interval: int) -> None:
        """Periodically check if dead workers are alive again and do GC."""
        while True:
            await asyncio.sleep(heartbeat_interval)
            await self.worker_service.check_workers()
            self.request_states.cleanup()

    def response_vote(self, request_id: str, victory_index: Literal[0, 1]) -> RequestState | None:
        """Record the user's response vote and return the new state."""
        if (state := self.request_states.get(request_id)) is not None:
            state.set_response_vote(victory_index)
            request_logger.info(state.json())
            return state
        return None

    def energy_vote(self, request_id: str, victory_index: Literal[0, 1]) -> RequestState | None:
        """Record the user's energy vote and return the new state."""
        # Pop the state from the dict, since this is the last step.
        if (state := self.request_states.pop(request_id)) is not None:
            state.set_energy_vote(victory_index)
            request_logger.info(state.json())
            return state
        return None

    async def prompt(
        self,
        request_id: str,
        prompt: str,
        model_index: Literal[0, 1],
    ) -> AsyncGenerator[bytes, None]:
        # This method is called twice for the same request, once for each model.
        # If it's the first time this method is called, assign models to the request.
        if request_id not in self.request_states:
            workers = self.worker_service.choose_two()
            model_names = [worker.model_name for worker in workers]
            request_state = self.request_states[request_id] = RequestState(
                request_id=request_id,
                prompt=prompt,
                model_names=model_names,
            )
        request_state = self.request_states[request_id]
        model_name = request_state.model_names[model_index]
        try:
            worker = self.worker_service.get_worker(model_name)
        except KeyError:
            controller_logger.error("Worker %s not found.", model_name)
            raise
        except RuntimeError:
            controller_logger.error("Worker %s is dead.", model_name)
            raise
        prompt = add_system_prompt(
            system_prompt=get_system_prompt("chat"),
            prompt=prompt,
            model_name=worker.model_name,
        )

        # Request the model worker to stream the response to the user's prompt.
        response = ""
        energy = 0.0
        try:
            async for resp in worker.client.generate_stream(prompt=prompt, max_new_tokens=self.max_new_tokens):
                # Even special tokens consume energy when they're generated.
                energy += resp.token.energy
                if not resp.token.special:
                    response += resp.token.text
                    yield json.dumps(response.token.text).encode() + b"\0"
        except OverloadedError:
            # TODO: Define controller errors in spitfight.colosseum.controller.errors.
            raise HTTPException(status_code=429, detail="Model overloaded. Pleaes try again later.")
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # XXX: This part could be done in the background with BackgroundTasks.
        request_state.set_response_and_energy(model_index, response, energy)
        request_logger.info(request_state.json())


CONTROLLER: Controller | None = None

def init_global_controller(config: ControllerConfig) -> None:
    global CONTROLLER
    CONTROLLER = Controller(
        max_new_tokens=config.max_new_tokens,
        background_task_interval=config.background_task_interval,
        max_num_req_states=config.max_num_req_states,
        req_state_expiration_time=config.req_state_expiration_time,
        worker_service=WorkerService(config.deployment_yaml)
    )

def get_global_controller() -> Controller:
    global CONTROLLER
    assert CONTROLLER is not None
    return CONTROLLER
