from __future__ import annotations

import time
import heapq
import asyncio
from typing import TypeVar, Generic, AsyncGenerator, Any, Coroutine

from fastapi.logger import logger

K = TypeVar('K')
V = TypeVar('V')


class BoundedExpiringDict(Generic[K, V]):
    def __init__(self, expiration_time: int) -> None:
        self.data_dict: dict[K, V] = {}
        self.timestamp_heap: list[tuple[float, K]] = []
        self.timeout = expiration_time

        # Without this, the controller is vulnerable to "user flood attacks,"
        # where someone can create a bunch of users by polling /request before
        # self.timeout expires and blow up memory.
        self.max_size = 10000

    def __getitem__(self, key: K) -> V:
        return self.data_dict[key]

    def __setitem__(self, key: K, value: V) -> None:
        if len(self.data_dict) >= self.max_size:
            self.cleanup()

        heapq.heappush(self.timestamp_heap, (time.monotonic(), key))
        self.data_dict[key] = value

    def __delitem__(self, key: K) -> None:
        # This is a bit inefficient, but it's not a common case operation.
        # We still need to do this to keep timestamp_heap in sync.
        del self.data_dict[key]
        for i, (_, existing_key) in enumerate(self.timestamp_heap):
            if existing_key == key:
                del self.timestamp_heap[i]
                break
        heapq.heapify(self.timestamp_heap)

    def __contains__(self, key: K) -> bool:
        return key in self.data_dict

    def __len__(self) -> int:
        return len(self.data_dict)

    def get(self, key: K, default: V | None = None) -> V | None:
        return self.data_dict.get(key, default)

    def pop(self, key: K, default: V | None = None) -> V | None:
        item = self.data_dict.pop(key, default)
        if item is not None:
            for i, (_, existing_key) in enumerate(self.timestamp_heap):
                if existing_key == key:
                    del self.timestamp_heap[i]
                    break
            heapq.heapify(self.timestamp_heap)
        return item

    def cleanup(self) -> None:
        now = time.monotonic()
        # After the while loop, the dictionary will be smaller than max_size
        # and all keys will have been accessed within the timeout.
        while (self.timestamp_heap and now - self.timestamp_heap[0][0] > self.timeout) or len(self.data_dict) > self.max_size:
            _, key = heapq.heappop(self.timestamp_heap)
            del self.data_dict[key]

        assert len(self.data_dict) == len(self.timestamp_heap)


T = TypeVar("T")


async def prepend_generator(
    first_item: T,
    generator: AsyncGenerator[T, None],
) -> AsyncGenerator[T, None]:
    """Prepend an item to an async generator."""
    yield first_item
    async for item in generator:
        yield item


def create_task(coroutine: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
    """Create an `asyncio.Task` but ensure that exceptions are logged.

    Reference: https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/
    """
    loop = asyncio.get_running_loop()
    task = loop.create_task(coroutine)
    task.add_done_callback(_handle_task_exception)
    return task


def _handle_task_exception(task: asyncio.Task) -> None:
    """Print out exception and tracebook when a task dies with an exception."""
    try:
        task.result()
    except asyncio.CancelledError:
        # Cancellation should not be logged as an error.
        pass
    except Exception:  # pylint: disable=broad-except
        # `logger.exception` automatically handles exception and traceback info.
        logger.exception("Job task died with an exception!")
