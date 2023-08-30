import yaml
import random
import asyncio
from typing import Literal
from functools import cached_property

import httpx
from pydantic import BaseModel
from text_generation import AsyncClient

from spitfight.log import get_logger

logger = get_logger(__name__)


class Worker(BaseModel):
    """A worker that serves a model."""
    # Worker's container name, since we're using Overlay networks.
    hostname: str
    # For TGI, this would always be 80.
    port: int
    # User-friendly model name, e.g. "Llama2-7B".
    model_name: str
    # Hugging Face model ID, e.g. "metaai/Llama-2-13b-chat-hf".
    model_id: str
    # Whether the model worker container is good.
    status: Literal["up", "down"]

    class Config:
        keep_untouched = (cached_property,)

    @cached_property
    def url(self) -> str:
        return f"http://{self.hostname}:{self.port}"

    def get_client(self) -> AsyncClient:
        return AsyncClient(base_url=self.url)

    def audit(self) -> None:
        """Make sure the worker is running and information is as expected.

        Assumed to be called on app startup when workers are initialized.
        This method will just raise `ValueError`s if audit fails in order to
        prevent the controller from starting if anything is wrong.
        """
        try:
            response = httpx.get(self.url + "/info")
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise ValueError(f"Could not connect to {self!r}: {e!r}")
        if response.status_code != 200:
            raise ValueError(f"Could not get /info from {self!r}.")
        info = response.json()
        if info["model_id"] != self.model_id:
            raise ValueError(f"Model name mismatch: {info['model_id']} != {self.model_id}")
        self.status = "up"
        logger.info("%s is up.", repr(self))

    async def check_status(self) -> None:
        """Check worker status and update `self.status` accordingly."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.url + "/info")
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                self.status = "down"
                logger.warning("%s is down: %s", repr(self), repr(e))
                return
            if response.status_code != 200:
                self.status = "down"
                logger.warning("GET /info from %s returned %s.", repr(self), response.json())
                return
            info = response.json()
            if info["model_id"] != self.model_id:
                self.status = "down"
                logger.warning(
                    "Model name mismatch for %s: %s != %s",
                    repr(self),
                    info["model_id"],
                    self.model_id,
                )
                return
        logger.info("%s is up.", repr(self))
        self.status = "up"


class WorkerService:
    """A service that manages model serving workers.

    Worker objects are only created once and shared across the
    entire application. Especially, changing the status of a worker
    will immediately take effect on the result of `choose_two`.

    Attributes:
        workers (list[Worker]): The list of workers.
    """

    def __init__(self, compose_files: list[str]) -> None:
        """Initialize the worker service."""
        self.workers: list[Worker] = []
        worker_model_names = set()
        for compose_file in compose_files:
            spec = yaml.safe_load(open(compose_file))
            for model_name, service_spec in spec["services"].items():
                command = service_spec["command"]
                for i, cmd in enumerate(command):
                    if cmd == "--model-id":
                        model_id = command[i + 1]
                        break
                else:
                    raise ValueError(f"Could not find model ID in {command!r}")
                worker_model_names.add(model_name)
                worker = Worker(
                    hostname=service_spec["container_name"],
                    port=80,
                    model_name=model_name,
                    model_id=model_id,
                    status="down",
                )
                worker.audit()
                self.workers.append(worker)

        if len(worker_model_names) != len(self.workers):
            raise ValueError("Model names must be unique.")

    def get_worker(self, model_name: str) -> Worker:
        """Get a worker by model name."""
        for worker in self.workers:
            if worker.model_name == model_name:
                if worker.status == "down":
                    # This is an unfortunate case where, when the two models were chosen,
                    # the worker was up, but after that went down before the request
                    # completed. We'll just raise a 500 internal error and have the user
                    # try again. This won't be common.
                    raise RuntimeError(f"The worker with model name {model_name} is down.")
                return worker
        raise ValueError(f"Worker with model name {model_name} does not exist.")

    def choose_two(self) -> tuple[Worker, Worker]:
        """Choose two different workers.

        Good place to use the Strategy Pattern when we want to
        implement different strategies for choosing workers.
        """
        live_workers = [worker for worker in self.workers if worker.status == "up"]
        if len(live_workers) < 2:
            raise ValueError("Not enough live workers to choose from.")
        worker_a, worker_b = random.sample(live_workers, 2)
        return worker_a, worker_b

    def choose_based_on_preference(self, preference: str) -> tuple[Worker, Worker]:
        """Choose two different workers based on user preference.

        Specifically, if `preference` is `"Random"`, this is equivalent to
        choosing two models at random. Otherwise, if `preference` is a model
        name, this is equivalent to choosing that model and another model at
        random. In that case, the order of the two models is also randomized.
        """
        if preference == "Random":
            return self.choose_two()
        else:
            worker_a = self.get_worker(preference)
            worker_b = random.choice([worker for worker in self.workers if worker != worker_a])
            return tuple(random.sample([worker_a, worker_b], 2))

    async def check_workers(self) -> None:
        """Check the status of all workers."""
        await asyncio.gather(*[worker.check_status() for worker in self.workers])
