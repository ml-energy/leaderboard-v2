from __future__ import annotations

import json
import unittest
import contextlib
from uuid import uuid4, UUID
from copy import deepcopy
from typing import Generator, Literal

import requests
import gradio as gr

from spitfight.colosseum.common import (
    COLOSSEUM_PROMPT_ROUTE,
    COLOSSEUM_RESP_VOTE_ROUTE,
    COLOSSEUM_ENERGY_VOTE_ROUTE,
    PromptRequest,
    ResponseVoteRequest,
    ResponseVoteResponse,
    EnergyVoteRequest,
    EnergyVoteResponse,
)


class ControllerClient:
    """Client for the Colosseum controller, to be used by Gradio."""

    def __init__(self, controller_addr: str, timeout: int = 15, request_id: UUID | None = None) -> None:
        """Initialize the controller client."""
        self.controller_addr = controller_addr
        self.timeout = timeout
        self.request_id = str(request_id) or str(uuid4())

    def fork(self) -> ControllerClient:
        """Return a copy of the client with a new request ID."""
        return ControllerClient(
            controller_addr=self.controller_addr,
            timeout=self.timeout,
            request_id=uuid4(),
        )

    def prompt(self, prompt: str, index: Literal[0, 1]) -> Generator[str, None, None]:
        """Generate the response of the `index`th model with the prompt."""
        prompt_request = PromptRequest(request_id=self.request_id, prompt=prompt, model_index=index)
        with _catch_requests_exceptions():
            resp = requests.post(
                f"http://{self.controller_addr}{COLOSSEUM_PROMPT_ROUTE}",
                json=prompt_request.dict(),
                stream=True,
                timeout=self.timeout,
            )
        _check_response(resp)
        # XXX: Why can't the server just yield `text + "\n"` and here we just iter_lines?
        for chunk in resp.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                yield json.loads(chunk.decode("utf-8"))

    def response_vote(self, victory_index: Literal[0, 1]) -> ResponseVoteResponse:
        """Notify the controller of the user's vote for the response."""
        response_vote_request = ResponseVoteRequest(request_id=self.request_id, victory_index=victory_index)
        with _catch_requests_exceptions():
            resp = requests.post(
                f"http://{self.controller_addr}{COLOSSEUM_RESP_VOTE_ROUTE}",
                json=response_vote_request.dict(),
            )
        _check_response(resp)
        return ResponseVoteResponse(**resp.json())

    def energy_vote(self, is_worth: bool) -> EnergyVoteResponse:
        """Notify the controller of the user's vote for energy."""
        energy_vote_request = EnergyVoteRequest(request_id=self.request_id, is_worth=is_worth)
        with _catch_requests_exceptions():
            resp = requests.post(
                f"http://{self.controller_addr}{COLOSSEUM_ENERGY_VOTE_ROUTE}",
                json=energy_vote_request.dict(),
            )
        _check_response(resp)
        return EnergyVoteResponse(**resp.json())


@contextlib.contextmanager
def _catch_requests_exceptions():
    """Catch requests exceptions and raise gr.Error instead."""
    try:
        yield
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise gr.Error("Failed to connect to our the backend server. Please try again later.")


def _check_response(response: requests.Response) -> None:
    if 400 <= response.status_code < 500:
        raise gr.Error(response.json()["detail"])
    elif response.status_code >= 500:
        raise gr.Error("Failed to talk to our backend server. Please try again later.")


class TestControllerClient(unittest.TestCase):
    def test_new_uuid_on_deepcopy(self):
        client = ControllerClient("http://localhost:8000")
        clients = [deepcopy(client) for _ in range(50)]
        request_ids = [client.request_id for client in clients]
        assert len(set(request_ids)) == len(request_ids)


if __name__ == "__main__":
    unittest.main()
