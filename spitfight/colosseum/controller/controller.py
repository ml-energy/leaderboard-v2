from __future__ import annotations

import random
import asyncio
from uuid import UUID
from datetime import datetime
from typing import AsyncGenerator, Literal, Optional, TYPE_CHECKING

from pytz import timezone
from pydantic import BaseModel, UUID4, Field

from spitfight.log import setup_logger
from spitfight.utils import BoundedExpiringDict
from spitfight.colosseum.controller.worker import WorkerService

if TYPE_CHECKING:
    from uuid import UUID
    from spitfight.colosseum.controller.router import ControllerConfig

controller_logger = setup_logger("controller.log")
user_logger = setup_logger("user.log")


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

def now() -> datetime:
    return datetime.now(tz=timezone("US/Eastern"))


class RequestState(BaseModel):
    """Models the state of a Colosseum play.

    This model is also serialized as is and logged.
    """
    request_id: UUID4
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
        self.request_states: BoundedExpiringDict[UUID, RequestState] = \
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

    def _log_state(self, state: RequestState) -> None:
        """Log the state of a request."""
        controller_logger.info(state.json())

    def response_vote(self, request_id: UUID, victory_index: Literal[0, 1]) -> RequestState | None:
        """Record the user's response vote and return the new state."""
        if (state := self.request_states.get(request_id)) is not None:
            state.set_response_vote(victory_index)
            self._log_state(state)
            return state
        return None

    def energy_vote(self, request_id: UUID, victory_index: Literal[0, 1]) -> RequestState | None:
        """Record the user's energy vote and return the new state."""
        # Pop the state from the dict, since this is the last step.
        if (state := self.request_states.pop(request_id)) is not None:
            state.set_energy_vote(victory_index)
            self._log_state(state)
            return state
        return None

    async def prompt(
        self,
        request_id: UUID,
        prompt: str,
        model_index: Literal[0, 1],
    ) -> AsyncGenerator[str, None]:
        # This method is called twice for the same request, once for each model.
        # We first need to see whether this is the first time or the second time.
        # If it's the first time, we should assign models to this request.
        try:
            request_state = self.request_states[request_id]
            prompt = request_state.prompt
            model_name = request_state.model_names[model_index]
        except KeyError:
            # TODO: Use FastChat? Make part of worker?
            prompt = add_system_prompt(prompt)
            request_state = self.request_states[request_id] = RequestState(
                request_id=request_id,
                prompt=prompt,
                model_names=random.sample(self.workers.keys(), 2),  # TOCTTOU
            )

        # TODO: Handle TGI validation errors, e.g. sequence too long.


    def receive_request_stream(self, model_name, prompt, user_id):
        if model_name not in self.model_dest:
            return None
        worker_addr, worker_port = random.choice(list(self.model_dest[model_name]))
        if not worker_addr or not worker_port:
            yield None
        url = f'http://{worker_addr}:{worker_port}'
        client = Client(url)
        text = ""
        self.request_states[user_id]['prompt'] = prompt
        model_id = self.request_states[user_id]['model'].index(model_name)
        for response in client.generate_stream(prompt, max_new_tokens=self.max_new_tokens):
            if not response.token.special:
                text += response.token.text
                self.request_states[user_id]['energy'][model_id] += response.token.energy
                yield json.dumps(response.token.text).encode() + b"\0"

        controller_logger.info(f"User {user_id} request {prompt} from {model_name} "
                    f"with energy {self.request_states[user_id]['energy'][model_id]}. "
                    f"with response {text} ")
        if user_id in self.request_states:
            self.request_states[user_id]['response'][model_id] = text
        # yield text


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
