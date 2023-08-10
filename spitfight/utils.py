from __future__ import annotations

import time
import heapq
import asyncio
import unittest
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


class TokenGenerationBuffer:
    """A constant sized buffer for tokens, used to handle stop sequences.

    Attributes:
        token_buffer (str): Internal buffer for tokens.
        matched_stop_str (bool): Whether the stop string has been seen. When this
            is True, generation should stop and `pop` will always return None.
    """
    def __init__(self, stop_str: str | None = None) -> None:
        """Initialize the buffer.

        If `stop_str` is None, the buffer will just return all tokens as they come.
        """
        self.stop_str = stop_str
        self.token_len_list = []
        self.token_buffer = ""
        self.matched_stop_str = False

    def append(self, text: str) -> None:
        """Append a token to the buffer."""
        if self.stop_str is not None:
            self.token_len_list.append(len(text))
        self.token_buffer += text

    def _pop_one(self) -> str:
        """Remove and return the first token in the buffer."""
        token_len = self.token_len_list.pop(0)
        token, self.token_buffer = self.token_buffer[:token_len], self.token_buffer[token_len:]
        return token

    def pop(self) -> str | None:
        """Try to pop a token from the buffer.

        Return value None means that there is nothing to yield for now.
        Repeated calls to this method will always just return None before more
        tokens are appended to the buffer.
        """
        # A short circuit for no stop string.
        if self.stop_str is None:
            return_buffer = self.token_buffer or None
            self.token_buffer = ""
            return return_buffer

        if self.matched_stop_str:
            return None

        # The token buffer matched the stop string. We're done generating.
        if self.stop_str == self.token_buffer:
            self.matched_stop_str = True
            return None

        # The tokens in the buffer could potentially be part of the stop string.
        # We'll stay put until we see more tokens. This also covers the case of
        # empty token buffer.
        if self.stop_str.startswith(self.token_buffer):
            return None

        # We can return tokens from the beginning of the buffer until the buffer
        # is a prefix of the stop string.
        return_buffer = ""
        while self.token_buffer:
            return_buffer += self._pop_one()
            if self.stop_str == self.token_buffer:
                self.matched_stop_str = True
                break
            if self.stop_str.startswith(self.token_buffer):
                break

        return return_buffer or None



class TestTokenGenerationBuffer(unittest.TestCase):
    def test_basic1(self):
        buffer = TokenGenerationBuffer("stop")

        buffer.append("hello")
        self.assertEqual(buffer.pop(), "hello")
        self.assertEqual(buffer.pop(), None)
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("world")
        self.assertEqual(buffer.pop(), "world")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("stop")
        self.assertEqual(buffer.pop(), None)
        self.assertTrue(buffer.matched_stop_str)
        self.assertEqual(buffer.pop(), None)
        self.assertTrue(buffer.matched_stop_str)
        self.assertEqual(buffer.pop(), None)
        self.assertTrue(buffer.matched_stop_str)
        self.assertEqual(buffer.pop(), None)
        self.assertTrue(buffer.matched_stop_str)

    def test_basic2(self):
        buffer = TokenGenerationBuffer("stop")

        buffer.append("hi")
        self.assertEqual(buffer.pop(), "hi")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("stole")
        self.assertEqual(buffer.pop(), "stole")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("sto")
        self.assertEqual(buffer.pop(), None)
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("ic")
        self.assertEqual(buffer.pop(), "stoic")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("st")
        self.assertEqual(buffer.pop(), None)
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("opper")
        self.assertEqual(buffer.pop(), "stopper")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("sto")
        self.assertEqual(buffer.pop(), None)
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("p")
        self.assertEqual(buffer.pop(), None)
        self.assertTrue(buffer.matched_stop_str)

    def test_falcon1(self):
        buffer = TokenGenerationBuffer("\nUser")

        buffer.append("Hi")
        self.assertEqual(buffer.pop(), "Hi")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("!")
        self.assertEqual(buffer.pop(), "!")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("\n")
        self.assertEqual(buffer.pop(), None)
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("User")
        self.assertEqual(buffer.pop(), None)
        self.assertTrue(buffer.matched_stop_str)

    def test_falcon2(self):
        buffer = TokenGenerationBuffer("\nUser")

        buffer.append("\n")
        self.assertEqual(buffer.pop(), None)
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("\n")
        self.assertEqual(buffer.pop(), "\n")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("\n")
        self.assertEqual(buffer.pop(), "\n")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("\n")
        self.assertEqual(buffer.pop(), "\n")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("User")
        self.assertEqual(buffer.pop(), None)
        self.assertEqual(buffer.pop(), None)
        self.assertTrue(buffer.matched_stop_str)

    def test_no_stop_str(self):
        buffer = TokenGenerationBuffer()

        buffer.append("hello")
        self.assertEqual(buffer.pop(), "hello")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("world")
        self.assertEqual(buffer.pop(), "world")
        self.assertFalse(buffer.matched_stop_str)

        buffer.append("\n")
        self.assertEqual(buffer.pop(), "\n")
        self.assertFalse(buffer.matched_stop_str)




if __name__ == "__main__":
    unittest.main()
