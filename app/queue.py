import json
import uuid
import redis.asyncio as aioredis
from redis.exceptions import ResponseError

from app.config import Config


async def push_job(redis: aioredis.Redis, config: Config, topic: str, session_id: str, output_format: str) -> str:
    """Adds a job payload to the Redis stream and returns its application-level job ID.

    Args:
        redis: Active asynchronous Redis client instance.
        config: Application configuration containing the target stream key.
        topic: Topic or domain associated with the job.
        session_id: Identifier for the originating user or session.
        output_format: Desired output format for the job result.

    Returns:
        The generated UUID string identifying the job.
    """
    job_id = str(uuid.uuid4())
    await redis.xadd(config.stream_key, {
        "job_id": job_id,
        "topic": topic,
        "session_id": session_id,
        "output_format": output_format,
    })
    return job_id


async def get_result(redis: aioredis.Redis, job_id: str) -> dict | None:
    """Retrieves a stored job result from Redis, if available.

    Args:
        redis: Active asynchronous Redis client instance.
        job_id: Application-level identifier of the job.

    Returns:
        The deserialized JSON result, or None if the result key is missing or its stored value is empty.
    """
    data = await redis.get(f"result:{job_id}")
    return json.loads(data) if data else None


async def set_result(redis: aioredis.Redis, config: Config, job_id: str, result: dict) -> None:
    """Stores a job result in Redis with the configured expiration time.

    Args:
        redis: Active asynchronous Redis client instance.
        config: Application configuration containing the result TTL setting.
        job_id: Application-level identifier of the job.
        result: Result payload to serialize as JSON.
    """
    await redis.setex(f"result:{job_id}", config.result_ttl, json.dumps(result))


async def ensure_group(redis: aioredis.Redis, config: Config) -> None:
    """Creates the consumer group for the Redis stream if it does not exist.

    Ensures multiple worker processes can share and distribute job processing across the target stream. Safely ignores duplicate group creation errors.

    Args:
        redis: Active asynchronous Redis client instance.
        config: Application configuration containing the stream key and consumer group name.
    """
    try:
        await redis.xgroup_create(
            config.stream_key,
            config.consumer_group,
            id="0",
            mkstream=True,
        )
    except ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def consume_jobs(redis: aioredis.Redis, config: Config) -> list[dict]:
    """Reads up to one new message from the Redis stream consumer group.

    Waits for up to five seconds for new messages when none are immediately available. Reads only messages that have not previously been delivered to a consumer in this group; pending messages are not recovered by this function.

    Args:
        redis: Active asynchronous Redis client instance.
        config: Application configuration containing the stream key, consumer group name, and consumer name.

    Returns:
        A list containing zero or one job entry. Each entry is a dictionary with:
            - 'msg_id': The Redis stream entry ID.
            - 'data': The message payload as a dictionary.
    """
    messages = await redis.xreadgroup(
        config.consumer_group,
        config.consumer_name,
        {config.stream_key: ">"},
        count=1,
        block=5000,
    )
    if not messages:
        return []

    jobs = []
    for _, entries in messages:
        for msg_id, data in entries:
            jobs.append({"msg_id": msg_id, "data": data})
    return jobs


async def ack_job(redis: aioredis.Redis, config: Config, msg_id: str) -> None:
    """Acknowledges a processed message in the Redis consumer group.

    Removes the message from the group's Pending Entries List (PEL) if it is pending. Does not delete the message from the stream.

    Args:
        redis: Active asynchronous Redis client instance.
        config: Application configuration containing the stream key and consumer group name.
        msg_id: Redis stream entry ID to acknowledge.
    """
    await redis.xack(
        config.stream_key,
        config.consumer_group,
        msg_id,
    )
