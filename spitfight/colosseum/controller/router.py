from pydantic import BaseConfig
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException

from spitfight.log import init_queued_root_logger, shutdown_queued_root_loggers
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


class ControllerConfig(BaseConfig):
    max_new_tokens: int = 256
    background_task_interval: int = 300
    max_num_req_states: int = 10000
    req_state_expiration_time: int = 600
    deployment_yaml: str = "deployment.yaml"
    controller_log_file: str = "controller.log"
    request_log_file: str = "requests.log"


app = FastAPI()
settings = ControllerConfig()

@app.on_event("startup")
async def startup_event():
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
    return StreamingResponse(
        controller.prompt(request.request_id, request.prompt, request.model_index)
    )

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
    if (state := controller.energy_vote(request.request_id, request.victory_index)) is None:
        raise HTTPException(status_code=410, detail="Colosseum battle session timeout expired.")
    return EnergyVoteResponse(model_names=state.model_names)

@app.get(COLOSSEUM_HEALTH_ROUTE)
async def health():
    return "OK\n"
