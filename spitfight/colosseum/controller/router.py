import json

import uvicorn
from pydantic import BaseSettings
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from text_generation.errors import OverloadedError, ValidationError

from spitfight.log import get_logger, init_queued_root_logger, shutdown_queued_root_loggers
from spitfight.colosseum.common import (
    COLOSSEUM_PROMPT_ROUTE,
    COLOSSEUM_RESP_VOTE_ROUTE,
    COLOSSEUM_ENERGY_VOTE_ROUTE,
    COLOSSEUM_HEALTH_ROUTE,
    PromptRequest,
    ResponseVoteRequest,
    ResponseVoteResponse,
    EnergyVoteRequest,
    EnergyVoteResponse,
)
from spitfight.colosseum.controller.controller import (
    Controller,
    init_global_controller,
    get_global_controller,
)
from spitfight.utils import prepend_generator


class ControllerConfig(BaseSettings):
    """Controller settings automatically loaded from environment variables."""
    max_new_tokens: int = 512
    background_task_interval: int = 300
    max_num_req_states: int = 10000
    req_state_expiration_time: int = 600
    deployment_yaml: str = "deployment.yaml"
    controller_log_file: str = "controller.log"
    request_log_file: str = "requests.log"
    uvicorn_log_file: str = "uvicorn.log"


app = FastAPI()
settings = ControllerConfig()
logger = get_logger("spitfight.colosseum.controller.router")

@app.on_event("startup")
async def startup_event():
    init_queued_root_logger("uvicorn", settings.uvicorn_log_file)
    init_queued_root_logger("spitfight.colosseum.controller", settings.controller_log_file)
    init_queued_root_logger("colosseum_requests", settings.request_log_file)
    init_global_controller(settings)

@app.on_event("shutdown")
async def shutdown_event():
    get_global_controller().shutdown()
    shutdown_queued_root_loggers()

@app.post(COLOSSEUM_PROMPT_ROUTE)
async def prompt(
    request: PromptRequest,
    controller: Controller = Depends(get_global_controller),
):
    generator = controller.prompt(request.request_id, request.prompt, request.model_index)

    # First try to get the first token in order to catch TGI errors.
    try:
        first_token = await generator.__anext__()
    except OverloadedError:
        name = controller.request_states[request.request_id].model_names[request.model_index]
        logger.warning("Model %s is overloaded. Failed request: %s", name, repr(request))
        raise HTTPException(status_code=429, detail="Model overloaded. Pleaes try again later.")
    except ValidationError as e:
        logger.info("TGI returned validation error: %s. Failed request: %s", str(e), repr(request))
        raise HTTPException(status_code=422, detail=str(e))
    except StopAsyncIteration:
        logger.info("TGI returned empty response. Failed request: %s", repr(request))
        return StreamingResponse(
            iter([json.dumps("*The model generated an empty response.*").encode() + b"\0"]),
        )

    return StreamingResponse(prepend_generator(first_token, generator))

@app.post(COLOSSEUM_RESP_VOTE_ROUTE, response_model=ResponseVoteResponse)
async def response_vote(
    request: ResponseVoteRequest,
    controller: Controller = Depends(get_global_controller),
):
    if (state := controller.response_vote(request.request_id, request.victory_index)) is None:
        raise HTTPException(status_code=410, detail="Colosseum battle session timeout expired.")
    return ResponseVoteResponse(
        energy_consumptions=state.energy_consumptions,
        model_names=state.model_names,
    )

@app.post(COLOSSEUM_ENERGY_VOTE_ROUTE, response_model=EnergyVoteResponse)
async def energy_vote(
    request: EnergyVoteRequest,
    controller: Controller = Depends(get_global_controller),
):
    if (state := controller.energy_vote(request.request_id, request.is_worth)) is None:
        raise HTTPException(status_code=410, detail="Colosseum battle session timeout expired.")
    return EnergyVoteResponse(model_names=state.model_names)

@app.get(COLOSSEUM_HEALTH_ROUTE)
async def health():
    return "OK"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", log_config=None)
