import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.middleware.auth import verify_admin_key
from app.models import AgenticSessionCreate, AgenticSessionResponse
from app.crud import (
    create_agentic_session,
    get_agentic_session,
    close_agentic_session,
)

router = APIRouter(prefix="/admin/agentic-sessions", tags=["Agentic Sessions"])


def _session_to_response(row) -> AgenticSessionResponse:
    return AgenticSessionResponse(
        id=str(row.id) if getattr(row, "id", None) is not None else "",
        session_id=row.session_id,
        combo_id=row.combo_id,
        messages=list(row.messages) if row.messages is not None else [],
        iteration_index=int(row.iteration_index or 0),
        step_history=list(row.step_history) if row.step_history is not None else [],
        created_at=str(row.created_at) if row.created_at else None,
        updated_at=str(row.updated_at) if row.updated_at else None,
        closed_at=str(row.closed_at) if row.closed_at else None,
    )


@router.post("", response_model=AgenticSessionResponse)
async def create_session(
    body: AgenticSessionCreate,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Create or update an agentic session.

    If session_id is omitted, one is generated as `sess-{uuid4hex12}`.
    """
    session_id = body.session_id
    if not session_id:
        session_id = f"sess-{uuid.uuid4().hex[:12]}"

    row = await create_agentic_session(db, session_id=session_id, combo_id=body.combo_id)
    await db.commit()
    return _session_to_response(row)


@router.get("/{session_id}", response_model=AgenticSessionResponse)
async def get_session(
    session_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve an agentic session by its session_id."""
    row = await get_agentic_session(db, session_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Agentic session '{session_id}' not found")
    return _session_to_response(row)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    _: str = Depends(verify_admin_key),
    db: AsyncSession = Depends(get_db),
):
    """Close (mark closed) an agentic session. Returns ok flag and session_id."""
    ok = await close_agentic_session(db, session_id)
    await db.commit()
    return {"ok": bool(ok), "session_id": session_id}
