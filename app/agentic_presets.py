STRATEGY_PRESETS: dict[str, dict] = {
    "research": {
        "name": "research",
        "description": "Deep research loop: parallel model fan-out with web search + HTTP tools for fact-gathering",
        "max_iterations": 8,
        "loop_strategy": "parallel",
        "compression_target_tokens": 6000,
        "recommended_tools": ["web_search", "http_request", "calculate", "bash_command"],
    },
    "code-review": {
        "name": "code-review",
        "description": "Best-of-N code review across top models with filesystem tools",
        "max_iterations": 6,
        "loop_strategy": "best_of",
        "compression_target_tokens": 5000,
        "recommended_tools": ["read_file", "calculate", "write_file", "bash_command"],
    },
    "debug": {
        "name": "debug",
        "description": "Deep sequential debug loop with bash + read introspection",
        "max_iterations": 10,
        "loop_strategy": "sequential",
        "compression_target_tokens": 4000,
        "recommended_tools": ["bash_command", "read_file", "calculate", "http_request"],
    },
}


def get_preset(name: str) -> dict | None:
    return STRATEGY_PRESETS.get(name)


def list_presets() -> list[dict]:
    return list(STRATEGY_PRESETS.values())


def apply_preset_defaults(body_fields: dict) -> dict:
    merged = dict(body_fields)
    preset_name = merged.get("strategy_preset")
    if not preset_name:
        return merged

    preset = get_preset(preset_name)
    if not preset:
        return merged

    agentic_config = merged.get("agentic_config")
    if agentic_config is None:
        agentic_config = {}
        merged["agentic_config"] = agentic_config

    if isinstance(agentic_config, dict):
        if agentic_config.get("max_iterations") is None:
            agentic_config["max_iterations"] = preset["max_iterations"]
        if agentic_config.get("loop_strategy") is None:
            agentic_config["loop_strategy"] = preset["loop_strategy"]
        if agentic_config.get("compression_target_tokens") is None:
            agentic_config["compression_target_tokens"] = preset["compression_target_tokens"]
    else:
        if getattr(agentic_config, "max_iterations", None) is None:
            agentic_config.max_iterations = preset["max_iterations"]
        if getattr(agentic_config, "loop_strategy", None) is None:
            agentic_config.loop_strategy = preset["loop_strategy"]
        if getattr(agentic_config, "compression_target_tokens", None) is None:
            agentic_config.compression_target_tokens = preset["compression_target_tokens"]

    tools_allowed = merged.get("tools_allowed")
    if not tools_allowed:
        merged["tools_allowed"] = list(preset["recommended_tools"])

    return merged
