import hashlib
import json
import numpy as np
import redis.asyncio as aioredis
from sentence_transformers import SentenceTransformer
from app.config import Config

_model = SentenceTransformer("all-MiniLM-L6-v2")
_CACHE_PREFIX = "semantic:"
_EMB_PREFIX = "emb:"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate and return the cosine similarity between two embedding vectors."""
    # Convert the input lists to float32 NumPy arrays for efficient numerical computation
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)

    denominator = np.linalg.norm(va) * np.linalg.norm(vb)
    # Avoid division by zero when either vector has zero magnitude
    if denominator == 0:
        return 0.0

    similarity = np.dot(va, vb) / denominator
    # Clamp floating-point rounding errors to the valid cosine similarity range [-1.0, 1.0]
    return float(np.clip(similarity, -1.0, 1.0))


def _embed(text: str) -> list[float]:
    """Convert text into a sentence embedding."""
    return _model.encode(text).tolist()


def _get_key_suffix(text: str) -> str:
    """Generate a deterministic key suffix for the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


async def cache_get(redis: aioredis.Redis, config: Config, query: str) -> str | None:
    """Retrieve the first cached result whose similarity exceeds the configured threshold for the given query."""
    # Convert the user query into a vector embedding
    query_emb = _embed(query)

    # Iterate over embedding keys without blocking Redis
    async for key in redis.scan_iter(match=f"{_EMB_PREFIX}*"):
        # Convert the stored embedding in JSON string back to a list
        stored_emb = json.loads(await redis.get(key))

        # Check if the stored embedding is similar enough to the embedding of the user query
        if _cosine_similarity(query_emb, stored_emb) >= config.cache_similarity_threshold:
            # Transform the embedding key to the corresponding cache key
            cache_key = key.replace(_EMB_PREFIX, _CACHE_PREFIX)
            # Asynchronously retrieve the cached query result in Redis
            return await redis.get(cache_key)

    # No embedding is found to meet the similarity threshold
    return None


async def cache_set(redis: aioredis.Redis, config: Config, query: str, result: str) -> None:
    """Cache a query result and its embedding using the configured TTL."""
    # Generate a deterministic key suffix that links the cached result and its vector embedding together
    key_suffix = _get_key_suffix(query)

    # Asynchronously store the cached query result text in Redis
    await redis.setex(f"{_CACHE_PREFIX}{key_suffix}", config.cache_ttl, result)

    # Asynchronously store the JSON string of the vector embedding of the query in Redis
    await redis.setex(f"{_EMB_PREFIX}{key_suffix}", config.cache_ttl, json.dumps(_embed(query)))
