import difflib
import json
import asyncio
from datetime import datetime
import redis.asyncio as aioredis
from sentence_transformers import SentenceTransformer
from app.config import Config
from app.pool import get_pool

_model = SentenceTransformer("all-MiniLM-L6-v2")


async def session_add(redis: aioredis.Redis, config: Config, session_id: str, role: str, content: str) -> None:
    """Appends a message to the short-term session history in Redis.

    Pushes the message to a Redis list, truncates the history to the maximum
    configured length, and refreshes the key expiration TTL.

    Args:
        redis: Active asynchronous Redis connection instance.
        config: Application configuration containing session parameters.
        session_id: Unique identifier for the chat session.
        role: Message sender role (e.g., 'user', 'assistant').
        content: Text body of the chat message.
    """
    key = f"session:{session_id}"
    await redis.rpush(key, json.dumps({"role": role, "content": content}))
    await redis.ltrim(key, -config.session_max_messages, -1)
    await redis.expire(key, config.session_ttl)


async def session_get(redis: aioredis.Redis, session_id: str) -> list[dict]:
    """Retrieves the stored chat session history from Redis.

    Args:
        redis: Active asynchronous Redis connection instance.
        session_id: Unique identifier for the chat session.

    Returns:
        List of decoded message dictionaries in their stored order.
    """
    messages = await redis.lrange(f"session:{session_id}", 0, -1)
    return [json.loads(m) for m in messages]


async def db_migrate(config: Config) -> None:
    """Initializes the long-term memory database schema and supporting indexes in PostgreSQL.

    Ensures the pgvector extension is enabled, creates the 'reports' table if it does not already exist, and creates vector, topic, and creation-time indexes used by long-term memory operations.

    Args:
        config: Application configuration containing the IVFFLAT index list count.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id         TEXT PRIMARY KEY,
                topic      TEXT NOT NULL,
                report     TEXT NOT NULL,
                embedding  vector(384),
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS reports_embedding_idx
            ON reports USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {config.ivfflat_lists})
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS reports_topic_idx ON reports (topic)")
        await conn.execute("CREATE INDEX IF NOT EXISTS reports_created_idx ON reports (created_at DESC)")


async def ltm_store(report_id: str, topic: str, report: str) -> None:
    """Stores a report and its topic embedding in long-term memory.

    Generates a 384-dimensional embedding for the topic using all-MiniLM-L6-v2 in a background thread, then persists the report, topic, embedding, and creation timestamp in PostgreSQL. If the supplied report ID already exists, the insert is ignored.

    Args:
        report_id: Unique primary key for the report entry.
        topic: Topic text used to generate the report embedding.
        report: Full text content of the report.
    """
    embedding = await asyncio.to_thread(lambda: _model.encode(topic).tolist())
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO reports (id, topic, report, embedding, created_at)
            VALUES ($1, $2, $3, $4::vector, $5)
            ON CONFLICT (id) DO NOTHING
            """,
            report_id, topic, report, str(embedding), datetime.utcnow(),
        )


async def ltm_search(config: Config, topic: str) -> dict | None:
    """Searches for the most similar report that satisfies the configured filters.

    Generates an embedding for the supplied topic and compares it with stored report embeddings using cosine similarity. Only reports newer than the configured age cutoff and above the configured similarity threshold are considered. The most similar qualifying report is returned.

    Args:
        config: Application configuration containing the age cutoff and similarity threshold.
        topic: Input topic used to generate the search embedding.

    Returns:
        Dictionary containing the matching report record and its similarity score, or None if no report satisfies the configured filters.
    """
    embedding = await asyncio.to_thread(lambda: _model.encode(topic).tolist())
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, topic, report, created_at,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM reports
            WHERE created_at > NOW() - ($2 || ' days')::INTERVAL
              AND 1 - (embedding <=> $1::vector) > $3
            ORDER BY similarity DESC LIMIT 1
            """,
            str(embedding), str(config.ltm_days), config.ltm_threshold,
        )
        return dict(row) if row else None


async def ltm_search_related(config: Config, topic: str) -> str | None:
    """Finds a related report within a lower embedding-similarity band for background reference.

    Generates an embedding for the supplied topic and searches for reports whose cosine similarity is between 0.5 and ``config.ltm_threshold - 0.01``, inclusive. The most recently created report within that similarity range is returned. Unlike ``ltm_search``, this query does not apply the configured age cutoff.

    Args:
        config: Application configuration containing the primary similarity threshold.
        topic: Topic used to generate the embedding for the related-topic search.

    Returns:
        The report text of the most recently created matching report, or None if no report falls within the configured similarity range.
    """
    embedding = await asyncio.to_thread(lambda: _model.encode(topic).tolist())
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT report FROM reports
            WHERE 1 - (embedding <=> $1::vector) BETWEEN 0.5 AND $2
            ORDER BY created_at DESC LIMIT 1
            """,
            str(embedding), config.ltm_threshold - 0.01,
        )
        return row["report"] if row else None
