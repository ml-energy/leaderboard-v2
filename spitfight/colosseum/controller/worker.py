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
    hostname: str
    port: int
    model_name: str
    model_id: str
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

    async def check_status(self) -> None:
        """Check worker status and update `self.status` accordingly."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.url + "/info")
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                self.status = "down"
                logger.warning("Worker %s is down: %s", repr(self), repr(e))
                return
            if response.status_code != 200:
                self.status = "down"
                logger.warning("GET /info from %s returned %s.", repr(self), response.json())
                return
            info = response.json()
            if info["model_id"] != self.model_id:
                self.status = "down"
                logger.warning(
                    "Model name mismatch for worker %s: %s != %s",
                    repr(self),
                    info["model_id"],
                    self.model_id,
                )
                return
        logger.info("Worker %s is up.", repr(self))
        self.status = "up"


class WorkerService:
    """A service that manages model serving workers.

    Worker objects are only created once and shared across the
    entire application. Especially, changing the status of a worker
    will immediately take effect on the result of `choose_two`.

    Attributes:
        workers (list[Worker]): The list of workers.
    """

    def __init__(self, deployment_yaml: str) -> None:
        """Initialize the worker service."""
        deployment = yaml.safe_load(open(deployment_yaml))

        self.workers: list[Worker] = []
        worker_model_names = set()
        for model in deployment["workers"]:
            model_name = model["name"]
            worker_model_names.add(model_name)
            worker = Worker(
                hostname=model["docker_params"]["name"],
                port=model["tgi_params"]["port"],
                model_name=model_name,
                model_id=model["tgi_params"]["model_id"],
                status="up",
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

    async def check_workers(self) -> None:
        """Check the status of all workers."""
        await asyncio.gather(*[worker.check_status() for worker in self.workers])
