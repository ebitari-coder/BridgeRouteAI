import asyncio
import inspect
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.middleware.auth import verify_admin_key
from app.tools_registry import tool_schemas, tools_registry, execute_tool

router = APIRouter(prefix="/admin/tools", tags=["Tools"])


class ToolExecRequest(BaseModel):
    name: str
    params: dict = {}


def _get_tool_description(name: str) -> str:
    func = tools_registry.get(name)
    if not func:
        return ""
    doc = inspect.getdoc(func) or ""
    return doc.strip()


@router.get("")
async def list_tools(_: str = Depends(verify_admin_key)):
    """List all registered tools with schemas and descriptions."""
    data = []
    for name in sorted(tool_schemas.keys()):
        data.append({
            "name": name,
            "schema": tool_schemas[name],
            "description": _get_tool_description(name),
        })
    return {"data": data}


@router.get("/{name}")
async def get_tool(name: str, _: str = Depends(verify_admin_key)):
    """Get schema and description for a single registered tool."""
    if name not in tool_schemas:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")
    return {
        "name": name,
        "schema": tool_schemas[name],
        "description": _get_tool_description(name),
    }


def _cap_string_size(obj, max_bytes: int = 100 * 1024) -> dict:
    """Recursively trim string values in a result dict to stay under a total size cap."""
    if isinstance(obj, dict):
        return {k: _cap_string_size(v, max_bytes) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cap_string_size(v, max_bytes) for v in obj]
    if isinstance(obj, str):
        encoded = obj.encode("utf-8", errors="replace")
        if len(encoded) > max_bytes:
            return encoded[:max_bytes].decode("utf-8", errors="replace") + "...[trimmed]"
        return obj
    return obj


@router.post("/exec")
async def exec_tool(
    body: ToolExecRequest,
    _: str = Depends(verify_admin_key),
):
    """Admin-only dry-run execution of a registered tool.

    Enforces a 15s timeout and 100KB string-result size cap.
    """
    if body.name not in tools_registry:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{body.name}' not registered. Available: {sorted(tools_registry.keys())}",
        )

    try:
        result = await asyncio.wait_for(
            execute_tool(body.name, body.params or {}),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Tool '{body.name}' exceeded 15s execution timeout",
        )

    capped = _cap_string_size(result, 100 * 1024)
    return capped
