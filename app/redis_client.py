import redis.asyncio as redis
from app.config import settings

# Global Redis connection pool
redis_pool: redis.Redis | None = None


async def get_redis() -> redis.Redis:
    """Get or create the Redis connection pool."""
    global redis_pool
    if redis_pool is None:
        redis_pool = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return redis_pool


async def close_redis() -> None:
    """Close the Redis connection pool."""
    global redis_pool
    if redis_pool is not None:
        await redis_pool.close()
        redis_pool = None
