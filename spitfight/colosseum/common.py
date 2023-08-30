from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

COLOSSEUM_MODELS_ROUTE = "/models"
COLOSSEUM_PROMPT_ROUTE = "/prompt"
COLOSSEUM_RESP_VOTE_ROUTE = "/response_vote"
COLOSSEUM_ENERGY_VOTE_ROUTE = "/energy_vote"
COLOSSEUM_HEALTH_ROUTE = "/health"


class ModelsResponse(BaseModel):
    available_models: list[str]


class PromptRequest(BaseModel):
    request_id: str
    prompt: str
    model_index: Literal[0, 1]
    model_preference: str


class ResponseVoteRequest(BaseModel):
    request_id: str
    victory_index: Literal[0, 1]


class ResponseVoteResponse(BaseModel):
    model_names: list[str]
    energy_consumptions: list[float]


class EnergyVoteRequest(BaseModel):
    request_id: str
    is_worth: bool


class EnergyVoteResponse(BaseModel):
    model_names: list[str]
