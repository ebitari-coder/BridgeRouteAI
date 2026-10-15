#!/usr/bin/env python3
"""Bridgeroute MCP server — expose the gateway's sandboxed tool registry to
any MCP-capable agent (Claude Code, ZCode, Cursor) over stdio.

Setup in the client:
  claude mcp add bridgeroute -- python mcp_server.py

Env:
  BRIDGEROUTE_BASE_URL  default http://127.0.0.1:8001
  BRIDGERoute_ADMIN_KEY or BRIDGEROUTE_ADMIN_KEY  admin credential for tool exec
  BRIDGEROUTE_API_KEY   optional user key for chat through the gateway
"""
import json
import os
import sys

import httpx

BASE = os.environ.get("BRIDGEROUTE_BASE_URL", "http://127.0.0.1:8001")
ADMIN_KEY = os.environ.get("BRIDGEROUTE_ADMIN_KEY") or ""
USER_KEY = os.environ.get("BRIDGEROUTE_API_KEY") or ""
CHAT_MODEL = os.environ.get("BRIDGEROUTE_CHAT_MODEL", "")
# tool_exec_mode: "direct" (admin /admin/tools/exec) or "chat" (user key ->
# /v1/chat agentic loop: billed to the key's budget + full analytics)
TOOL_EXEC_MODE = os.environ.get("BRIDGEROUTE_TOOL_EXEC_MODE", "chat" if USER_KEY else "direct")
TIMEOUT = 60.0

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "bridgeroute", "version": "1.0.0"}


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if ADMIN_KEY:
        h["Authorization"] = f"Bearer {ADMIN_KEY}"
    return h


def tool_schemas() -> list[dict]:
    """Fetch the registered tools (name, description, JSON schema)."""
    try:
        r = httpx.get(f"{BASE}/admin/tools", headers=_headers(), timeout=15.0)
        if r.status_code != 200:
            return []
        return [
            {
                "name": t["name"],
                "description": t.get("description") or t["name"],
                "inputSchema": t.get("schema") or {"type": "object", "properties": {}},
            }
            for t in r.json().get("data", [])
        ]
    except Exception:
        return []


def tool_exec_via_chat(name: str, args: dict) -> dict:
    """Route the tool call through /v1/chat with the user key: the gateway
    executes it under the keys budget, rate limits, and analytics."""
    payload = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": (
            f"Call the {name} tool with these exact arguments: "
            f"{json.dumps(args)}. Execute it, then reply with the JSON result only."
        )}],
        "tool_choice": "required",
    }
    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        json=payload,
        headers={**_headers(), "Authorization": f"Bearer {USER_KEY or ADMIN_KEY}"},
        timeout=TIMEOUT * 3,
    )
    if r.status_code != 200:
        return {"isError": True, "content": [{"type": "text", "text": f"HTTP {r.status_code}: {r.text[:500]}"}]}
    body = r.json()
    text_out = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return {"content": [{"type": "text", "text": text_out}], "isError": False}


def tool_exec(name: str, args: dict) -> dict:
    if TOOL_EXEC_MODE == "chat" and USER_KEY:
        return tool_exec_via_chat(name, args)
    r = httpx.post(
        f"{BASE}/admin/tools/exec",
        json={"name": name, "params": args},
        headers=_headers(),
        timeout=TIMEOUT,
    )
    if r.status_code != 200:
        return {"isError": True, "content": [{"type": "text", "text": f"HTTP {r.status_code}: {r.text[:500]}"}]}
    return {
        "content": [{"type": "text", "text": json.dumps(r.json(), ensure_ascii=False)}],
        "isError": False,
    }


def handle(msg: dict) -> dict | None:
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method in ("ping",):
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
        }}
    if method == "notifications/initialized":
        return None  # notification: no response
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tool_schemas()}}
    if method == "tools/call":
        name = params.get("name")
        raw_args = params.get("arguments") or {}
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            result = tool_exec(name, args)
            result = result
        except Exception as exc:
            result = {"isError": True, "content": [{"type": "text", "text": str(exc)}]}
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
