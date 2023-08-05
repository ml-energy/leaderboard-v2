from __future__ import annotations

from typing import Literal

from pydantic import UUID4, BaseModel

COLOSSEUM_PROMPT_ROUTE = "/prompt"
COLOSSEUM_RESP_VOTE_ROUTE = "/response_vote"
COLOSSEUM_ENERGY_VOTE_ROUTE = "/energy_vote"


class PromptRequest(BaseModel):
    user_id: UUID4
    prompt: str
    index: Literal[0, 1]


class ResponseVoteRequest(BaseModel):
    user_id: UUID4
    victory_index: Literal[0, 1]


class ResponseVoteResponse(BaseModel):
    model_names: tuple[str, str]
    energy_consumptions: tuple[float, float]


class EnergyVoteRequest(BaseModel):
    user_id: UUID4
    victory_index: Literal[0, 1]
