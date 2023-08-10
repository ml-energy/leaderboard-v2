from __future__ import annotations

import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Literal, Optional, TYPE_CHECKING

from pytz import timezone
from pydantic import BaseModel, Field

from spitfight.log import get_logger
from spitfight.utils import BoundedExpiringDict, create_task
from spitfight.colosseum.controller.worker import WorkerService
from spitfight.prompt import get_system_prompt, apply_model_characteristics

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
    extra_energy_was_worth: Optional[bool] = None

    # The time when the user's stage changed.
    timestamp: datetime = Field(default_factory=now)
    # The user's current stage.
    user_stage: UserStage = "prompted"
    # When the the user is not going through the aforementioned stages,
    # the user's stage transition is recorded here.
    abnormal_stage_change: list[tuple[UserStage, UserStage]] = []

    def set_response_and_energy(self, model_index: Literal[0, 1], response: str, energy_consumption: float) -> None:
        self.timestamp = now()
        self.energy_consumptions[model_index] = energy_consumption
        self.responses[model_index] = response

    def set_response_vote(self, victory_index: Literal[0, 1]) -> None:
        self.timestamp = now()

        # Next stage depends on the user's vote.
        energy_a, energy_b = self.energy_consumptions
        if (victory_index == 0 and energy_a <= energy_b) or (victory_index == 1 and energy_a >= energy_b):
            next_stage = "chose_less_energy_response"
        else:
            next_stage = "chose_more_energy_response"

        # Detect abnormal stage change.
        if self.user_stage != "prompted":
            self.abnormal_stage_change.append((self.user_stage, next_stage))

        self.user_stage = next_stage
        self.response_victory_index = victory_index

    def set_energy_vote(self, is_worth: bool) -> None:
        self.timestamp = now()

        # Detect abnormal stage change.
        if self.user_stage != "chose_more_energy_response":
            self.abnormal_stage_change.append((self.user_stage, "voted_energy"))

        self.user_stage = "voted_energy"
        self.extra_energy_was_worth = is_worth


class GenerationConfig(BaseModel):
    """Configuration for generation of prompts."""
    max_new_tokens: int
    do_sample: bool
    temperature: float
    repetition_penalty: float
    top_k: int
    top_p: float


class Controller:
    def __init__(
        self,
        background_task_interval: int,
        max_num_req_states: int,
        req_state_expiration_time: int,
        worker_service: WorkerService,
        generation_config: GenerationConfig,
    ):
        self.max_num_req_states = max_num_req_states
        self.request_states: BoundedExpiringDict[str, RequestState] = \
            BoundedExpiringDict(req_state_expiration_time)
        self.worker_service = worker_service

        self.generation_config = generation_config

        self.background_task_handle = create_task(
            self._background_task(background_task_interval),
        )

    def shutdown(self) -> None:
        """Shutdown the controller."""
        self.background_task_handle.cancel()

    async def _background_task(self, heartbeat_interval: int) -> None:
        """Periodically check if dead workers are alive again and do request state GC."""
        while True:
            await asyncio.sleep(heartbeat_interval)

            await self.worker_service.check_workers()

            prev_num_req_states = len(self.request_states)
            self.request_states.cleanup()
            controller_logger.info(
                "Request state garbage collection done: Removed %d reqeusts",
                prev_num_req_states - len(self.request_states),
            )

    def response_vote(self, request_id: str, victory_index: Literal[0, 1]) -> RequestState | None:
        """Record the user's response vote and return the new state."""
        if (state := self.request_states.get(request_id)) is not None:
            state.set_response_vote(victory_index)
            # Pop the state from the dict if the user has voted on energy.
            if state.user_stage == "chose_less_energy_response":
                self.request_states.pop(request_id)
            request_logger.info(state.json())
            return state
        return None

    def energy_vote(self, request_id: str, is_worth: bool) -> RequestState | None:
        """Record the user's energy vote and return the new state."""
        # Pop the state from the dict, since this is the last step in any case.
        if (state := self.request_states.pop(request_id)) is not None:
            state.set_energy_vote(is_worth)
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
            self.request_states[request_id] = RequestState(
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
        prompt, stop_str, stop_token_ids = apply_model_characteristics(
            system_prompt=get_system_prompt("chat"),
            prompt=prompt,
            model_name=worker.model_id,
        )

        # Request the model worker to stream the response to the user's prompt.
        response = ""
        energy = 0.0
        client = worker.get_client()
        buffer = ""
        async for resp in client.generate_stream(
            prompt=prompt,
            stop_sequences=[stop_str] if stop_str is not None else None,
            **self.generation_config.dict(),
        ):
            # Even special tokens consume energy when they're generated.
            energy += resp.token.energy

            # Stop tokens usually don't overlap with (human-readable) stop sequences.
            if resp.token.special or resp.token.id in stop_token_ids:
                # If the buffer is not empty (i.e., we had partial stop_str matches),
                # yield it to the user.
                if buffer:
                    response += buffer
                    yield json.dumps(buffer).encode() + b"\0"
                break

            # Giving TGI `stop_sequences` will still generate the entire `stop_str`.
            # Therefore, we still need to keep track of the possibility of stopping.
            # If we see generated tokens matching the prefix of `stop_str`, we do not
            # yield them to the user but keep it in `buffer`. Later, if `stop_str`
            # ends up fully matching, we break the loop and discard the `buffer`, which
            # would be identical to `stop_str`. Otherwise, which means that `buffer` is
            # no longer a prefix of `stop_str`, we yield the `buffer` to the user.
            if stop_str is not None:
                # Avoid string concatenation if the buffer is empty.
                if not buffer:
                    buffer = resp.token.text
                else:
                    buffer += resp.token.text

                # Full match. Finish generation.
                if buffer == stop_str:
                    break

                # Partial match. Keep buffering.
                if stop_str.startswith(buffer):
                    continue

                # No match. Yield and empty the buffer.
                response += buffer
                yield json.dumps(buffer).encode() + b"\0"
                buffer = ""

            # No stop sequence. Just yield the token.
            else:
                curr_text = resp.token.text
                response += curr_text
                yield json.dumps(curr_text).encode() + b"\0"

        request_state.set_response_and_energy(model_index, response, energy)
        request_logger.info(request_state.json())


CONTROLLER: Controller | None = None

def init_global_controller(config: ControllerConfig) -> None:
    global CONTROLLER
    CONTROLLER = Controller(
        background_task_interval=config.background_task_interval,
        max_num_req_states=config.max_num_req_states,
        req_state_expiration_time=config.req_state_expiration_time,
        worker_service=WorkerService(config.compose_files),
        generation_config=GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            top_k=config.top_k,
            top_p=config.top_p,
        ),
    )

def get_global_controller() -> Controller:
    global CONTROLLER
    assert CONTROLLER is not None
    return CONTROLLER
