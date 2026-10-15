from fastapi import APIRouter, Request
from app.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)

router = APIRouter(tags=["Models"])


@router.post("/v1/embeddings")
async def create_embeddings(body: EmbeddingRequest):
    """Embeddings endpoint (stub).

    Returns a placeholder embedding. Use with frameworks that call
    the embeddings endpoint during initialization.
    """
    texts = body.input if isinstance(body.input, list) else [body.input]

    data = []
    for i, text in enumerate(texts):
        import random
        embedding = [random.uniform(-1, 1) for _ in range(1536)]
        norm = sum(x * x for x in embedding) ** 0.5
        embedding = [x / norm for x in embedding]
        data.append(EmbeddingData(embedding=embedding, index=i))

    prompt_tokens = sum(max(1, len(t) // 4) for t in texts)

    return EmbeddingResponse(
        data=data,
        model=body.model,
        usage=EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


@router.get("/v1/engines")
async def list_engines():
    """Legacy engines endpoint."""
    from app.routes.chat import list_models
    return await list_models()
