from __future__ import annotations

import json
import logging
import unittest
from uuid import uuid4, UUID
from copy import deepcopy
from typing import Generator, Literal

import requests
import gradio as gr
from pydantic import BaseModel

from spitfight.common import (
    COLOSSEUM_PROMPT_ROUTE,
    COLOSSEUM_RESP_VOTE_ROUTE,
    COLOSSEUM_ENERGY_VOTE_ROUTE,
    PromptRequest,
    ResponseVoteRequest,
    ResponseVoteResponse,
    EnergyVoteRequest,
)

logger = logging.getLogger(__name__)


class ControllerClient:
    """Client for the Colosseum controller."""

    def __init__(self, controller_addr: str, timeout: int = 15, user_id: UUID | None = None) -> None:
        """Initialize the controller client."""
        self.controller_addr = controller_addr
        self.timeout = timeout
        self.user_id = user_id or uuid4()

    def __deepcopy__(self, memo: dict) -> ControllerClient:
        """Return a deepcopy of the client with a new user ID.

        This exploints the fact that gr.State simply deepcopies objects.
        """
        return ControllerClient(
            controller_addr=deepcopy(self.controller_addr, memo),
            timeout=deepcopy(self.timeout, memo),
            user_id=uuid4(),
        )

    def prompt(self, prompt: str, index: Literal[0, 1]) -> Generator[str, None, None]:
        """Generate the response of the `index`th model with the prompt."""
        prompt_request = PromptRequest(user_id=self.user_id, prompt=prompt, index=index)
        resp = requests.post(
            f"http://{self.controller_addr}{COLOSSEUM_PROMPT_ROUTE}",
            json=prompt_request.dict(),
            stream=True,
            timeout=self.timeout,
        )
        _check_response(prompt_request, resp)
        # XXX: Why can't the server just yield `text + "\n"` and here we just iter_lines?
        for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                yield json.loads(chunk.decode("utf-8"))

    def response_vote(self, victory_index: Literal[0, 1]) -> ResponseVoteResponse:
        """Notify the controller of the user's vote for the response."""
        response_vote_request = ResponseVoteRequest(user_id=self.user_id, victory_index=victory_index)
        resp = requests.post(
            f"http://{self.controller_addr}{COLOSSEUM_RESP_VOTE_ROUTE}",
            json=response_vote_request.dict(),
        )
        _check_response(response_vote_request, resp)
        return ResponseVoteResponse(**resp.json())

    def energy_vote(self, victory_index: Literal[0, 1]) -> None:
        """Notify the controller of the user's vote for energy."""
        energy_vote_request = EnergyVoteRequest(user_id=self.user_id, victory_index=victory_index)
        resp = requests.post(
            f"http://{self.controller_addr}{COLOSSEUM_ENERGY_VOTE_ROUTE}",
            json=energy_vote_request.dict(),
        )
        _check_response(energy_vote_request, resp)


def _check_response(request: BaseModel, response: requests.Response) -> None:
    if response.status_code != 200:
        logger.error(
            "Request to %s failed with code %d.\nRequest: %s\nResponse: %s",
            COLOSSEUM_RESP_VOTE_ROUTE,
            response.status_code,
            request.json(),
            response.text,
        )
        raise gr.Error("Failed to talked to the backend. Please try again later.")


class TestControllerClient(unittest.TestCase):
    def test_new_uuid_on_deepcopy(self):
        client = ControllerClient("http://localhost:8000")
        clients = [deepcopy(client) for _ in range(50)]
        user_ids = [client.user_id for client in clients]
        assert len(set(user_ids)) == len(user_ids)


if __name__ == "__main__":
    unittest.main()
