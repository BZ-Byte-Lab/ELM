"""
OpenRouter API client with rate limiting and retry logic.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import threading
from collections import deque

from openai import OpenAI, AsyncOpenAI

from .utils import get_logger
from .config import SynthesisConfig, TaskConfig

logger = get_logger("api_client")


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, requests_per_second: float):
        """Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
        """
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """Wait until a request token is available."""
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

            time.sleep(0.1)


class AsyncRateLimiter:
    """Async token bucket rate limiter with concurrency control and event-based notification."""

    def __init__(self, requests_per_second: float, max_concurrent: int = 10):
        """Initialize async rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            max_concurrent: Maximum concurrent requests
        """
        self.rate = requests_per_second
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.condition = asyncio.Condition()  # Event-based notification
        self._refill_task = None
        self._running = False

    async def start(self) -> None:
        """Start the background token refill task."""
        if self._refill_task is None and not self._running:
            self._running = True
            self._refill_task = asyncio.create_task(self._refill_tokens())
            logger.debug("AsyncRateLimiter background refill task started")

    async def stop(self) -> None:
        """Stop the background token refill task."""
        if self._refill_task:
            self._running = False
            self._refill_task.cancel()
            try:
                await self._refill_task
            except asyncio.CancelledError:
                pass
            self._refill_task = None
            logger.debug("AsyncRateLimiter background refill task stopped")

    async def _refill_tokens(self) -> None:
        """Background task to refill tokens and notify waiters."""
        try:
            while self._running:
                await asyncio.sleep(1.0 / self.rate)  # Sleep until next token available
                async with self.condition:
                    now = time.time()
                    elapsed = now - self.last_update
                    self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
                    self.last_update = now
                    self.condition.notify(1)  # Wake one waiting task
        except asyncio.CancelledError:
            logger.debug("Token refill task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in token refill task: {e}")

    async def acquire(self) -> None:
        """Wait until a request token is available (check tokens BEFORE semaphore)."""
        # Start refill task lazily on first acquire
        if self._refill_task is None:
            await self.start()

        # Check for token first (eliminates polling delay)
        async with self.condition:
            while self.tokens < 1:
                await self.condition.wait()  # No polling - event-driven!

            # Consume token
            self.tokens -= 1

        # Then acquire semaphore to limit concurrency
        await self.semaphore.acquire()

    async def release(self) -> None:
        """Release the semaphore after request completes."""
        self.semaphore.release()


@dataclass
class GenerationResult:
    """Result of a single generation."""
    success: bool
    text: Optional[str]
    error: Optional[str]
    token_count: int
    latency: float


class OpenRouterClient:
    """OpenRouter API client with rate limiting."""

    def __init__(self, config: SynthesisConfig):
        """Initialize OpenRouter client.

        Args:
            config: Synthesis configuration
        """
        self.config = config

        # Sync client (for backward compatibility)
        self.client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.get_api_key(),
        )
        self.rate_limiter = RateLimiter(config.requests_per_second)

        # Async client (NEW)
        self.async_client = AsyncOpenAI(
            base_url=config.api_base_url,
            api_key=config.get_api_key(),
        )
        self.async_rate_limiter = AsyncRateLimiter(
            config.requests_per_second,
            max_concurrent=getattr(config, 'max_concurrent_requests', 10),
        )

        self.request_times: deque = deque(maxlen=100)

    def generate(
        self,
        prompt: str,
        task_config: TaskConfig,
        max_tokens: int = 500,
    ) -> GenerationResult:
        """Generate text using OpenRouter API.

        Args:
            prompt: Input prompt
            task_config: Task configuration with temperature/top_p
            max_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with success status and text
        """
        # Rate limiting
        self.rate_limiter.acquire()

        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=task_config.temperature,
                    top_p=task_config.top_p,
                    max_tokens=max_tokens,
                )

                text = response.choices[0].message.content
                token_count = response.usage.completion_tokens if response.usage else 0
                latency = time.time() - start_time

                # Track request time
                self.request_times.append(latency)

                logger.debug(f"Generated {token_count} tokens in {latency:.2f}s")

                return GenerationResult(
                    success=True,
                    text=text,
                    error=None,
                    token_count=token_count,
                    latency=latency,
                )

            except Exception as e:
                logger.warning(f"API error (attempt {attempt + 1}): {e}")

                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    return GenerationResult(
                        success=False,
                        text=None,
                        error=str(e),
                        token_count=0,
                        latency=time.time() - start_time,
                    )

    def generate_batch(
        self,
        prompts: List[str],
        task_config: TaskConfig,
        max_tokens: int = 500,
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts.

        Args:
            prompts: List of input prompts
            task_config: Task configuration
            max_tokens: Maximum tokens per generation

        Returns:
            List of GenerationResults
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, task_config, max_tokens)
            results.append(result)
        return results

    async def generate_async(
        self,
        prompt: str,
        task_config: TaskConfig,
        max_tokens: int = 500,
    ) -> GenerationResult:
        """Generate text using OpenRouter API (async version).

        Args:
            prompt: Input prompt
            task_config: Task configuration with temperature/top_p
            max_tokens: Maximum tokens to generate

        Returns:
            GenerationResult with success status and text
        """
        # Rate limiting (async) - acquire token and semaphore
        await self.async_rate_limiter.acquire()

        start_time = time.time()

        try:
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=task_config.temperature,
                        top_p=task_config.top_p,
                        max_tokens=max_tokens,
                    )

                    text = response.choices[0].message.content
                    token_count = response.usage.completion_tokens if response.usage else 0
                    latency = time.time() - start_time

                    # Track request time
                    self.request_times.append(latency)

                    logger.debug(f"Generated {token_count} tokens in {latency:.2f}s (async)")

                    return GenerationResult(
                        success=True,
                        text=text,
                        error=None,
                        token_count=token_count,
                        latency=latency,
                    )

                except Exception as e:
                    logger.warning(f"API error (attempt {attempt + 1}): {e}")

                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    else:
                        logger.error(f"Failed after {self.config.max_retries} retries: {e}")
                        return GenerationResult(
                            success=False,
                            text=None,
                            error=str(e),
                            token_count=0,
                            latency=time.time() - start_time,
                        )
        finally:
            # Always release semaphore, even on exception
            await self.async_rate_limiter.release()

    async def generate_batch_async(
        self,
        prompts: List[str],
        task_config: TaskConfig,
        max_tokens: int = 500,
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts in parallel (async version).

        Args:
            prompts: List of input prompts
            task_config: Task configuration
            max_tokens: Maximum tokens per generation

        Returns:
            List of GenerationResults
        """
        tasks = [
            self.generate_async(prompt, task_config, max_tokens)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics.

        Returns:
            Dictionary with latency and request statistics
        """
        if not self.request_times:
            return {"avg_latency": 0, "total_requests": 0}

        return {
            "avg_latency": sum(self.request_times) / len(self.request_times),
            "total_requests": len(self.request_times),
            "min_latency": min(self.request_times),
            "max_latency": max(self.request_times),
        }
