import json
import time
import yaml
import random
import asyncio
import requests
from uuid import UUID
from typing import Literal, Optional
from collections import defaultdict

import uvicorn
from text_generation import Client
from pydantic import UUID4, BaseModel, BaseConfig
from fastapi import FastAPI, Request, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException

from spitfight.colosseum.common import (
    COLOSSEUM_PROMPT_ROUTE,
    COLOSSEUM_RESP_VOTE_ROUTE,
    COLOSSEUM_ENERGY_VOTE_ROUTE,
    PromptRequest,
    ResponseVoteRequest,
    ResponseVoteResponse,
    EnergyVoteRequest,
)
from spitfight.colosseum.controller.controller import (
    Controller,
    init_global_controller,
    get_global_controller,
)


class ControllerConfig(BaseConfig):
    max_new_tokens: int = 1024
    background_task_interval: int = 300
    max_num_req_states: int = 10000
    req_state_expiration_time: int = 600
    deployment_yaml: str = "deployment.yaml"


app = FastAPI()
settings = ControllerConfig()

@app.on_event("startup")
async def startup_event():
    init_global_controller(settings)

@app.on_event("shutdown")
async def shutdown_event():
    get_global_controller().shutdown()

@app.post(COLOSSEUM_PROMPT_ROUTE)
async def prompt(
    request: PromptRequest,
    controller: Controller = Depends(get_global_controller),
):
    return StreamingResponse(
        controller.prompt(request.request_id, request.prompt, request.index),
    )
"""
    system_prompt = "A chat between a human user and an assistant, who gives helpful and polite answers to the user's questions. "
    prompt = system_prompt + data["prompt"]
    index = data["index"]
    user_id = request.headers.get("X-User-ID")
    controller.random_assign_models(user_id)
    model_name = controller.request_states[user_id]['model'][index]

    return StreamingResponse(controller.receive_request_stream(model_name, prompt, user_id))
"""

# XXX: Why is this POST??
# @app.post("/get_models")
# async def get_models():
#     return controller.get_models()

@app.post(COLOSSEUM_RESP_VOTE_ROUTE, response_model=ResponseVoteResponse)
async def response_vote(
    request: ResponseVoteRequest,
    controller: Controller = Depends(get_global_controller),
):
    if (state := controller.response_vote(request.request_id, request.victory_index)) is None:
        raise HTTPException(status_code=410, detail="Colosseum battle session timeout expired.")
    return ResponseVoteResponse(
        model_names=state.model_names,
        energy_consumptions=state.energy_consumptions,
    )

"""
    user_id = request.headers.get("X-User-ID")
    data = await request.json()
    nlp_voting = data["nlp_voting"]
    controller_logger.info(f"User {user_id} return nlp_voting {nlp_voting} between models {controller.request_states[user_id]['model']}")
    controller.request_states[user_id]['nlp_voting'] = nlp_voting
    if user_id in controller.request_states:
        model_names = controller.request_states[user_id]['model']
        controller.request_states[user_id]['model'] = None
        return model_names + controller.request_states[user_id]['energy']
    else:
        return 'TIMEOUT' + 0
"""

@app.post(COLOSSEUM_ENERGY_VOTE_ROUTE)
async def energy_vote(
    request: EnergyVoteRequest,
    controller: Controller = Depends(get_global_controller),
):
    if controller.energy_vote(request.request_id, request.victory_index) is None:
        raise HTTPException(status_code=410, detail="Colosseum battle session timeout expired.")

"""
    user_id = request.headers.get("X-User-ID")
    data = await request.json()
    energy_voting = data["energy_voting"]
    if user_id in controller.request_states:
        controller.request_states[user_id]['energy_voting'] = energy_voting
        user_logger.info(controller.request_states[user_id])
        controller.remove_user(user_id)
    else:
        controller_logger.info(f"User {user_id} expired energy voting {energy_voting}")
"""
