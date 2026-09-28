import asyncpg
from app.config import Config

# Define a global variable to store the database connection pool and set it to None
_pool: asyncpg.Pool | None = None


async def init_pool(config: Config) -> None:
    """Initialize a reusable PostgreSQL database connection pool to share across services and optimize system performance."""
    global _pool
    _pool = await asyncpg.create_pool(
        config.database_url,
        min_size=config.db_pool_min,
        max_size=config.db_pool_max,
    )


async def close_pool() -> None:
    """Close the database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Return the initialized database connection pool instance.

    Raises:
        RuntimeError: If the pool has not been initialized yet.
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    return _pool
