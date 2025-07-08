import time

import anyio
import pytest

from asyncmq.backends.memory import InMemoryBackend
from asyncmq.conf import monkay
from asyncmq.workers import Worker

pytestmark = pytest.mark.anyio


async def test_worker_heartbeat_registration_and_updates():
    backend = InMemoryBackend()
    monkay.settings.backend = backend
    worker = Worker("test_queue", heartbeat_interval=0.1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker._run_with_scope)
        await anyio.sleep(0.05)

        # Check that worker is registered
        workers = await backend.list_workers()
        assert len(workers) == 1
        assert workers[0].id == worker.id
        assert workers[0].queue == "test_queue"
        assert workers[0].concurrency == worker.concurrency

        # Store initial heartbeat timestamp
        initial_heartbeat = workers[0].heartbeat

        # Wait for at least one heartbeat cycle
        await anyio.sleep(0.15)

        # Check that heartbeat was updated
        workers = await backend.list_workers()
        assert len(workers) == 1
        assert workers[0].heartbeat > initial_heartbeat

        # Cancel the worker
        tg.cancel_scope.cancel()

    # After cancellation, worker should be deregistered
    workers = await backend.list_workers()
    assert len(workers) == 0

async def test_backend_returns_only_alive_workers():
    backend = InMemoryBackend()

    # Manually register a worker with an old timestamp
    old_timestamp = time.time() - (monkay.settings.heartbeat_ttl + 10)
    await backend.register_worker("expired_worker", "test_queue", 1, old_timestamp)

    # Register a current worker
    current_timestamp = time.time()
    await backend.register_worker("current_worker", "test_queue", 1, current_timestamp)

    workers = await backend.list_workers()
    assert len(workers) == 1
    assert workers[0].id == "current_worker"


async def test_multiple_workers_heartbeat():
    backend = InMemoryBackend()
    monkay.settings.backend = backend

    # Create multiple workers
    worker1 = Worker("queue1", heartbeat_interval=0.1)
    worker2 = Worker("queue2", heartbeat_interval=0.1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker1._run_with_scope)
        tg.start_soon(worker2._run_with_scope)

        await anyio.sleep(0.05)

        # Check both workers are registered
        workers = await backend.list_workers()
        assert len(workers) == 2

        worker_ids = {w.id for w in workers}
        assert worker1.id in worker_ids
        assert worker2.id in worker_ids

        # Check queue assignments
        worker_queues = {w.queue for w in workers}
        assert worker_queues == {"queue1", "queue2"}

        tg.cancel_scope.cancel()

    # Both workers should be deregistered
    workers = await backend.list_workers()
    assert len(workers) == 0


async def test_worker_heartbeat_survives_backend_restart():
    backend = InMemoryBackend()
    monkay.settings.backend = backend

    worker = Worker("test_queue", heartbeat_interval=0.1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(worker._run_with_scope)

        await anyio.sleep(0.05)

        # Verify worker is registered
        workers = await backend.list_workers()
        assert len(workers) == 1

        # Simulate backend restart by clearing worker registry
        backend._worker_registry.clear()

        # Worker should re-register on next heartbeat
        await anyio.sleep(0.15)

        workers = await backend.list_workers()
        assert len(workers) == 1
        assert workers[0].id == worker.id

        tg.cancel_scope.cancel()


async def test_worker_heartbeat_ttl_configuration():
    backend = InMemoryBackend()
    monkay.settings.backend = backend
    monkay.settings.heartbeat_ttl = 1

    # Register worker with current timestamp
    current_time = time.time()
    await backend.register_worker("test_worker", "test_queue", 1, current_time)

    # Should be listed as active
    workers = await backend.list_workers()
    assert len(workers) == 1

    # Register worker with timestamp just over TTL
    old_time = current_time - 1.1  # Just over 1 second TTL
    await backend.register_worker("old_worker", "test_queue", 1, old_time)

    # Only current worker should be listed
    workers = await backend.list_workers()
    assert len(workers) == 1
    assert workers[0].id == "test_worker"
