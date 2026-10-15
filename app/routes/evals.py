"""Admin eval harness endpoints: golden tasks, run suite, results."""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.middleware.auth import verify_admin_key
from app.eval_harness import get_eval_result, list_eval_results, load_golden_tasks, run_eval_suite
from app.models import EvalRunRequest, EvalTaskInfo

router = APIRouter(prefix="/admin/evals", tags=["Evals"])


@router.get("/tasks", response_model=list[EvalTaskInfo])
async def list_eval_tasks(_: str = Depends(verify_admin_key)):
    """List golden tasks from evals/golden_tasks.json."""
    try:
        tasks = load_golden_tasks()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="golden_tasks.json not found")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load golden tasks: {exc}")
    return [
        EvalTaskInfo(
            id=t.id,
            name=t.name or t.id,
            messages=t.messages,
            expect_substrings=t.expect_substrings,
            expect_regex=t.expect_regex,
            tools_allowed=t.tools_allowed,
            max_iterations=t.max_iterations,
            loop_strategy=t.loop_strategy,
        )
        for t in tasks
    ]


@router.post("/run")
async def run_evals(body: EvalRunRequest | None = None, _: str = Depends(verify_admin_key)):
    """Run the golden-task suite (optionally a subset) and return the run result."""
    body = body or EvalRunRequest()
    try:
        result = await run_eval_suite(task_ids=body.task_ids)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Eval run failed: {exc}")
    return result


@router.get("/results")
async def eval_results(limit: int = Query(default=20, ge=1, le=50), _: str = Depends(verify_admin_key)):
    """List recent eval run summaries (newest first)."""
    items = await list_eval_results(limit=limit)
    return {"data": items, "total": len(items)}


@router.get("/results/{run_id}")
async def eval_result(run_id: str, _: str = Depends(verify_admin_key)):
    """Fetch a full eval run result by id."""
    item = await get_eval_result(run_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found")
    return item
