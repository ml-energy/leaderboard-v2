from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

COLOSSEUM_PROMPT_ROUTE = "/prompt"
COLOSSEUM_RESP_VOTE_ROUTE = "/response_vote"
COLOSSEUM_ENERGY_VOTE_ROUTE = "/energy_vote"


class PromptRequest(BaseModel):
    request_id: str
    prompt: str
    model_index: Literal[0, 1]


class ResponseVoteRequest(BaseModel):
    request_id: str
    victory_index: Literal[0, 1]


class ResponseVoteResponse(BaseModel):
    model_names: list[str]
    energy_consumptions: list[float]


class EnergyVoteRequest(BaseModel):
    request_id: str
    victory_index: Literal[0, 1]


class EnergyVoteResponse(BaseModel):
    model_names: list[str]
