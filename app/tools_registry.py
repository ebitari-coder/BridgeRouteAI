"""Tool registry for agentic AI engine.

This module provides tool schema discovery and registration for the agentic loop.
Tools are async functions that take a dict of parameters and return serializable results.
"""
import json
import importlib
import inspect
import logging
import math
import os as _os
import shlex
from pathlib import Path as _Path
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

tools_registry: dict[str, Callable[[dict], Any]] = {}
tool_schemas: dict[str, dict] = {}
tool_descriptions: dict[str, str] = {}
tools_usage_metrics: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "success": 0, "error": 0})

# Sentinel for "the tool policy could not be loaded" — deny everything (S3).
# Compared by identity so it stays distinct from an ordinary empty list, which
# means "no allow-list configured" (unrestricted).
TOOLS_POLICY_UNAVAILABLE: list[str] = []

_DEFAULT_TOOL_ROOTS = (
    _Path.cwd() / "workspace",
    _Path.cwd() / "tmp",
    _Path("/tmp/opencode"),
)


def _allowed_tool_roots() -> list[_Path]:
    roots: list[_Path] = []
    env = _os.environ.get("TOOL_FILE_ROOTS", "")
    for part in env.split(";"):
        part = part.strip()
        if part:
            roots.append(_Path(part).resolve())
    if not roots:
        # NOTE: the process cwd is deliberately NOT a jail root — otherwise
        # read_file/write_file could reach .env and every other project file.
        roots = [r.resolve() for r in _DEFAULT_TOOL_ROOTS if r.exists()]
    return roots


def resolve_safe_path(path_str: str) -> _Path:
    """Resolve a user-supplied path and ensure it stays under allowed roots.

    Both the target and the roots are fully resolved (symlinks expanded) before
    the containment check, so a symlink planted inside a jail root cannot point
    the tool at a file outside it.

    Raises ValueError when the path escapes the allow-list or is empty.
    """
    if not path_str or not str(path_str).strip():
        raise ValueError("path is required")

    candidate = _Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (_Path.cwd() / candidate)

    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        raise ValueError(f"invalid path: {e}") from e

    roots = _allowed_tool_roots()
    if not roots:
        raise ValueError("no allowed file roots configured")

    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    raise ValueError(f"path outside allowed roots: {path_str}")


def get_tool_usage_metrics() -> dict:
    """Get usage metrics for all registered tools."""
    return dict(tools_usage_metrics)


def register_tool(name: str, schema: dict | None = None):
    """Decorator to register a tool function.
    
    Args:
        name: Tool name (must be unique)
        schema: Optional JSON schema for tool parameters
        
    Example:
        @register_tool("calculate", {"type": "object", "properties": {...}})
        async def calculate(params):
            return {"result": eval(params["expression"])}
    """
    def decorator(func: Callable[[dict], Any]):
        tools_registry[name] = func
        if schema:
            tool_schemas[name] = schema
        # Capture the function docstring as the tool description so models
        # receive a meaningful explanation of what each tool does.
        doc = inspect.getdoc(func)
        if doc:
            tool_descriptions[name] = doc
        return func
    return decorator


def get_tool_description(name: str) -> str:
    """Get the human-readable description for a registered tool."""
    return tool_descriptions.get(name, "")


def get_tool_schema(name: str) -> dict | None:
    """Get the schema for a registered tool."""
    return tool_schemas.get(name)


def get_all_tools() -> list[str]:
    """Get list of all registered tool names."""
    return list(tools_registry.keys())


def validate_tool_exists(name: str) -> bool:
    """Check if a tool is registered."""
    return name in tools_registry


def sanitize_tool_args(schema: dict, params: dict) -> dict:
    """Sanitize tool arguments against a schema.

    - Keep only params keys that exist in schema properties (allow-listing).
    - For each kept key: if schema property declares maxLength and value is str, truncate.
    - Coerce types: integer via int(), number via float(), boolean via bool().
    """
    if not schema or not params:
        return params if params else {}

    properties = schema.get("properties", {})
    if not properties:
        return params

    sanitized = {}
    dropped: list[str] = []
    truncated: dict[str, tuple[int, int]] = {}
    coerced: dict[str, tuple[str, str]] = {}
    for key, value in params.items():
        if key not in properties:
            dropped.append(key)
            continue
        prop_schema = properties[key]
        original_value = value

        if isinstance(value, str) and "maxLength" in prop_schema:
            try:
                max_len = int(prop_schema["maxLength"])
                if len(value) > max_len:
                    value = value[:max_len]
                    truncated[key] = (len(original_value), max_len)
            except (TypeError, ValueError):
                pass

        expected_type = prop_schema.get("type")
        if expected_type == "integer" and not isinstance(value, bool):
            if isinstance(value, str):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    pass
            elif isinstance(value, float):
                value = int(value)
        elif expected_type == "number" and not isinstance(value, bool):
            if isinstance(value, str):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass
            elif isinstance(value, int):
                value = float(value)
        elif expected_type == "boolean":
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes"):
                    value = True
                elif value.lower() in ("false", "0", "no"):
                    value = False

        if value is not original_value and type(value) is not type(original_value):
            coerced[key] = (type(original_value).__name__, type(value).__name__)

        sanitized[key] = value

    # A sanitize that actually changed the caller's arguments is worth a
    # warning: silently dropping a key or coercing a type can turn a
    # well-formed tool call into a surprising result, and the model has no
    # other signal that its arguments were rewritten before execution.
    if dropped or truncated or coerced:
        logger.warning(
            "sanitize_tool_args rewrote arguments: dropped=%s truncated=%s coerced=%s",
            dropped or None,
            {k: f"{old}->{new}" for k, (old, new) in truncated.items()} or None,
            {k: f"{old}->{new}" for k, (old, new) in coerced.items()} or None,
        )

    return sanitized


def register_tools_from_module(module_path: str) -> dict[str, dict]:
    """Discover and register tools from a module.

    Uses discover_tools_from_module and additionally auto-registers functions
    matching the async (dict) -> dict pattern.
    Returns dict of {tool_name: schema}.
    """
    discovered = discover_tools_from_module(module_path)

    try:
        module = importlib.import_module(module_path)
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name in tools_registry:
                continue
            try:
                sig = inspect.signature(obj)
                params_count = len(sig.parameters)
                if inspect.iscoroutinefunction(obj) and params_count >= 1:
                    existing_schema = discovered.get(name) or tool_schemas.get(name)
                    if existing_schema:
                        tools_registry[name] = obj
                        if name not in tool_schemas:
                            tool_schemas[name] = existing_schema
            except (TypeError, ValueError):
                continue
    except Exception as e:
        print(f"Error auto-registering tools from {module_path}: {e}")

    return {name: tool_schemas[name] for name in tools_registry if name in discovered or name in tool_schemas}


# ── Built-in Tools ───────────────────────────────────────────────────────────

@register_tool("web_search", {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "default": 5},
    },
    "required": ["query"],
})
async def web_search(params: dict) -> dict:
    """Search the web for information."""
    # TODO: Implement real web search
    return {
        "query": params["query"],
        "results": [],
        "note": "Web search not yet implemented - placeholder",
    }


@register_tool("read_file", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to read"},
        "encoding": {"type": "string", "default": "utf-8"},
    },
    "required": ["path"],
})
async def read_file(params: dict) -> dict:
    """Read a file from disk and return its contents. Use this to inspect source
    code, config, or any file before modifying it. Path must be under an allowed
    root (workspace, tmp, or /tmp/opencode)."""
    import asyncio
    encoding = params.get("encoding", "utf-8")

    try:
        path = resolve_safe_path(params["path"])
    except ValueError as e:
        return {"error": str(e), "success": False}

    try:
        # Run file read in executor to avoid blocking
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, lambda: open(path, 'r', encoding=encoding).read())
        return {
            "path": str(path),
            "content": content[:10000],  # Truncate for safety
            "size": len(content),
            "success": True,
        }
    except FileNotFoundError:
        return {"error": f"File not found: {path}", "success": False}
    except PermissionError:
        return {"error": f"Permission denied: {path}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


@register_tool("write_file", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "File path to write"},
        "content": {"type": "string", "description": "Content to write"},
        "mode": {"type": "string", "default": "w"},
    },
    "required": ["path", "content"],
})
async def write_file(params: dict) -> dict:
    """Write content to a file, creating parent directories as needed. Use this
    to create or overwrite source code, config, or any text file. Path must be
    under an allowed root (workspace, tmp, or /tmp/opencode)."""
    import asyncio
    content = params["content"]
    mode = params.get("mode", "w")

    try:
        path = resolve_safe_path(params["path"])
    except ValueError as e:
        return {"error": str(e), "success": False}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            lambda: open(path, mode, encoding='utf-8').write(content)
        )
        return {
            "path": str(path),
            "bytes_written": len(content.encode('utf-8')),
            "success": True,
        }
    except PermissionError:
        return {"error": f"Permission denied: {path}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


# Shell metacharacters are rejected outright: the command is executed via
# create_subprocess_exec (no shell), but refusing them keeps the intent of the
# allow-list obvious and avoids surprising literal-argument semantics.
_SHELL_METACHARACTERS = frozenset(";|&$`<>{}()\n\r\\")

@register_tool("bash_command", {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "number", "default": 30.0},
    },
    "required": ["command"],
})
async def bash_command(params: dict) -> dict:
    """Execute a shell command and return its output. Only a safe allow-list of
    commands is permitted (ls, cat, pwd, echo, date, whoami, uname, hostname).
    No shell metacharacters; arguments are parsed without a shell."""
    import asyncio

    command = params["command"]
    timeout = params.get("timeout", 30.0)

    # Security warnings - only allow safe commands
    safe_commands = ['ls', 'cat', 'pwd', 'echo', 'date', 'whoami', 'uname', 'hostname']

    if not isinstance(command, str) or not command.strip():
        return {"error": "command is required", "success": False}

    # Parse once; shlex removes quoting so `echo "a b"` yields one argument.
    try:
        argv = shlex.split(command, posix=True)
    except ValueError as e:
        return {"error": f"invalid command: {e}", "command": command, "success": False}

    if not argv:
        return {"error": "command is empty", "success": False}

    cmd_base = argv[0]
    if cmd_base not in safe_commands:
        return {
            "error": f"Command '{cmd_base}' not in safe list. Allowed: {safe_commands}",
            "command": command,
            "success": False,
        }

    for token in argv[1:]:
        if any(ch in _SHELL_METACHARACTERS for ch in token):
            return {
                "error": f"Shell metacharacters are not allowed: {token!r}",
                "command": command,
                "success": False,
            }

    try:
        timeout = min(max(float(timeout), 1.0), 120.0)
    except (TypeError, ValueError):
        timeout = 30.0

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"error": f"Command timed out after {timeout}s", "success": False}

        return {
            "command": command,
            "stdout": stdout.decode('utf-8', errors='replace')[:10000],
            "stderr": stderr.decode('utf-8', errors='replace')[:10000],
            "returncode": proc.returncode,
            "success": proc.returncode == 0,
        }
    except Exception as e:
        return {"error": str(e), "success": False}


@register_tool("calculate", {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
    },
    "required": ["expression"],
})
async def calculate(params: dict) -> dict:
    """Safely evaluate a mathematical expression."""
    import ast
    import operator
    
    expression = params["expression"]

    # Bounds that keep `9**9**9` (or any long mul-chain) from becoming a CPU/RAM bomb.
    MAX_POW_BASE = 10 ** 64
    MAX_POW_EXPONENT = 1000
    MAX_RESULT_BITS = 4096  # ~1233 decimal digits

    # Safe eval for math expressions only
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def check_magnitude(value):
        if isinstance(value, bool):
            raise ValueError("Only numbers allowed")
        if isinstance(value, int):
            if value.bit_length() > MAX_RESULT_BITS:
                raise ValueError("result too large")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("non-finite result")
            return value
        raise ValueError("Only numbers allowed")

    def safe_eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return check_magnitude(node.value)
            raise ValueError("Only numbers allowed")
        if isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Operator not allowed: {type(node.op).__name__}")
            if op is operator.pow:
                if abs(left) > MAX_POW_BASE or abs(right) > MAX_POW_EXPONENT:
                    raise ValueError(
                        f"exponent too large (|base| <= 10^64, |exponent| <= {MAX_POW_EXPONENT})"
                    )
            return check_magnitude(op(left, right))
        raise ValueError("Expression too complex")
    
    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        return {"expression": expression, "result": result, "success": True}
    except Exception as e:
        return {"error": str(e), "expression": expression, "success": False}


@register_tool("sleep", {
    "type": "object",
    "properties": {
        "seconds": {"type": "number", "description": "Seconds to sleep"},
    },
    "required": ["seconds"],
})
async def sleep(params: dict) -> dict:
    """Sleep for a specified number of seconds."""
    import asyncio
    seconds = min(float(params["seconds"]), 60)
    await asyncio.sleep(seconds)
    return {"slept_seconds": seconds, "success": True}


@register_tool("echo_test", {
    "type": "object",
    "properties": {
        "message": {"type": "string", "maxLength": 200},
    },
    "required": ["message"],
})
async def echo_test(params: dict) -> dict:
    """Test tool that echoes back the provided message."""
    return {"message": params["message"], "echoed": True, "success": True}


@register_tool("http_request", {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "HTTPS URL to request"},
        "method": {"type": "string", "default": "GET", "description": "HTTP method"},
        "headers": {"type": "object", "default": {}, "description": "Request headers"},
        "body": {"type": "string", "description": "Request body (optional)"},
    },
    "required": ["url"],
})
async def http_request(params: dict) -> dict:
    """Make an HTTPS request. Only HTTPS scheme allowed. Response body capped at 50KB."""
    import httpx
    MAX_BODY = 50000
    ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}
    try:
        url = params.get("url", "")
        if not url.lower().startswith("https://"):
            return {"success": False, "error": "Only https:// URLs are allowed", "url": url}
        method = (params.get("method") or "GET").upper()
        if method not in ALLOWED_METHODS:
            return {"success": False, "error": f"Method not allowed: {method}", "allowed": sorted(ALLOWED_METHODS)}
        headers = params.get("headers") or {}
        if not isinstance(headers, dict):
            headers = {}
        body = params.get("body")

        async with httpx.AsyncClient(timeout=10.0) as client:
            kwargs: dict = {"headers": headers}
            if body is not None and method in {"POST", "PUT", "PATCH"}:
                kwargs["content"] = body
            resp = await client.request(method, url, **kwargs)
            resp_headers = {k: v for k, v in dict(resp.headers).items()}
            trimmed_headers = {}
            header_len = 0
            for k, v in resp_headers.items():
                entry = f"{k}: {v}"
                if header_len + len(entry) > 4000:
                    break
                trimmed_headers[k] = v
                header_len += len(entry)
            raw_body = resp.text
            if len(raw_body) > MAX_BODY:
                raw_body = raw_body[:MAX_BODY]
            return {
                "success": True,
                "status_code": resp.status_code,
                "headers": trimmed_headers,
                "body": raw_body,
                "url": url,
                "method": method,
            }
    except Exception as e:
        return {"success": False, "error": str(e), "url": params.get("url", "")}


# ── Tool Discovery Function ───────────────────────────────────────────────────

def discover_tools_from_module(module_path: str) -> dict[str, dict]:
    """Discover tools registered in a module.
    
    Returns dict of {tool_name: schema} for all discovered tools.
    """
    import importlib
    import inspect
    
    discovered = {}
    
    try:
        module = importlib.import_module(module_path)
        
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if hasattr(obj, '__wrapped__') or name in tools_registry:
                schema = get_tool_schema(name)
                if schema:
                    discovered[name] = schema
                    
    except Exception as e:
        print(f"Error discovering tools from {module_path}: {e}")
    
    return discovered


def _schemas_for_names(names: list[str] | None) -> list[dict]:
    if names is TOOLS_POLICY_UNAVAILABLE:
        items: list[tuple[str, dict]] = []
    elif names is not None and len(names) > 0:
        items = [(n, tool_schemas[n]) for n in names if n in tool_schemas]
    else:
        items = list(tool_schemas.items())
    return [
        {"type": "function", "function": {"name": name, "description": "", "parameters": schema}}
        for name, schema in items
    ]


async def get_tools_for_combo(combo_id: str) -> list[dict]:
    """Async: get enabled tools for a combo from DB (tools_allowed column)."""
    allowed_list: list[str] | None = None
    if combo_id:
        try:
            from app.crud import get_combo_tools
            from app.database import async_session
            async with async_session() as db:
                allowed_list = await get_combo_tools(db, combo_id)
        except Exception:
            # Fail closed: advertise nothing rather than every tool (S3).
            logger.error("Failed to load tool policy for combo %s — no tools advertised", combo_id, exc_info=True)
            return _schemas_for_names(TOOLS_POLICY_UNAVAILABLE)
    return _schemas_for_names(allowed_list)


def get_tools_for_combo_sync(combo_id: str) -> list[dict]:
    """Sync wrapper (for non-async callers / fallbacks)."""
    allowed_list: list[str] | None = None
    if combo_id:
        try:
            from app.crud import get_combo_tools
            from app.database import async_session
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    allowed_list = pool.submit(
                        lambda: asyncio.run(_fetch_combo_tools(combo_id))
                    ).result()
            else:
                allowed_list = asyncio.run(_fetch_combo_tools(combo_id))
        except Exception:
            logger.error("Failed to load tool policy for combo %s — no tools advertised", combo_id, exc_info=True)
            return _schemas_for_names(TOOLS_POLICY_UNAVAILABLE)
    return _schemas_for_names(allowed_list)


async def _fetch_combo_tools(combo_id: str) -> list[str] | None:
    from app.crud import get_combo_tools
    from app.database import async_session
    async with async_session() as db:
        return await get_combo_tools(db, combo_id)


# ── Utility Functions ──────────────────────────────────────────────────────────

async def execute_tool(tool_name: str, args: dict) -> dict:
    """Execute a tool by name with arguments.

    Sanitizes args before calling, and tracks usage metrics.
    Caches successful results for idempotent tools (see app.tool_cache).
    Returns the tool result or error.
    """
    tools_usage_metrics[tool_name]["calls"] += 1

    if tool_name not in tools_registry:
        tools_usage_metrics[tool_name]["error"] += 1
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found",
            "available": list(tools_registry.keys()),
        }

    try:
        schema = tool_schemas.get(tool_name)
        sanitized_args = sanitize_tool_args(schema, args) if schema else args

        from app.tool_cache import get_or_none, put as cache_put
        cached_raw = await get_or_none(tool_name, sanitized_args)
        if cached_raw is not None:
            import json as _json
            try:
                cached = _json.loads(cached_raw)
                if isinstance(cached, dict):
                    tools_usage_metrics[tool_name]["success"] += 1
                    return cached
            except _json.JSONDecodeError:
                pass

        result = await tools_registry[tool_name](sanitized_args)
        tools_usage_metrics[tool_name]["success"] += 1
        if isinstance(result, dict) and result.get("success") is not False:
            import json as _json
            try:
                await cache_put(tool_name, sanitized_args, _json.dumps(result, default=str))
            except (TypeError, ValueError):
                pass
        return result
    except Exception as e:
        tools_usage_metrics[tool_name]["error"] += 1
        return {
            "success": False,
            "error": str(e),
            "tool": tool_name,
        }