"""Interactive brokerage analytics chatbot powered by MCP-style tools."""

import importlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

import utils.brokerage_mcp as brokerage_mcp_module
from utils.openai_commentary import get_openai_client


load_dotenv()

st.set_page_config(page_title="Brokerage GPT", page_icon="🧠", layout="wide")


SYSTEM_PROMPT = (
    "You are an expert brokerage analyst assisting with Vietnamese securities firms. "
    "You must rely on the provided tools for all factual data, metrics, valuations, and commentary. "
    "Use get_data_availability first to learn the latest completed quarter overall and per ticker (see latest_quarters_by_ticker). "
    "After using tools, summarise insights clearly and mention any data gaps."
)


SUPPORTED_MODELS = [
    "gpt-5-mini",
    "gpt-4o-mini",
    "gpt-4o",
]


def _default_model_name() -> str:
    env_model = os.getenv("OPENAI_MODEL")
    if env_model:
        return env_model
    for key in ("model", "OPENAI_MODEL"):
        try:
            value = st.secrets["openai"].get(key) if key == "model" else st.secrets.get(key)
            if value:
                return str(value)
        except Exception:  # noqa: BLE001
            continue
    return SUPPORTED_MODELS[0]


def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    pricing = {
        "gpt-5": {"input": 0.03, "output": 0.06},
        "gpt-5-mini": {"input": 0.0015, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    }
    model_pricing = pricing.get(model, pricing["gpt-4"])
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    return input_cost + output_cost


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = st.session_state.brokerage_tool_cache.get(key)
    if not entry:
        return None
    age = (datetime.utcnow() - entry["timestamp"]).total_seconds()
    if age > st.session_state.brokerage_tool_cache_ttl:
        return None
    return entry["result"]


def _cache_set(key: str, result: Dict[str, Any]) -> None:
    st.session_state.brokerage_tool_cache[key] = {
        "result": result,
        "timestamp": datetime.utcnow(),
    }


def compact_tool_result_for_llm(result: Dict[str, Any], max_rows: Optional[int] = None) -> Dict[str, Any]:
    try:
        limit = max_rows or int(os.getenv("LLM_TOOL_MAX_ROWS", "200"))
    except Exception:  # noqa: BLE001
        limit = 200
    if not isinstance(result, dict):
        return {"data": result}
    payload = json.loads(json.dumps(result, default=str))
    data = payload.get("data")
    if isinstance(data, dict):
        rows = data.get("rows")
        if isinstance(rows, list) and len(rows) > limit:
            try:
                head_n = max(5, min(20, int(os.getenv("LLM_TOOL_HEAD_ROWS", "10"))))
                tail_n = max(0, min(10, int(os.getenv("LLM_TOOL_TAIL_ROWS", "5"))))
            except Exception:  # noqa: BLE001
                head_n = 10
                tail_n = 5
            payload["data_head"] = rows[:head_n]
            payload["data_tail"] = rows[-tail_n:] if tail_n else []
            cleaned = {k: v for k, v in data.items() if k != "rows"}
            cleaned["rows"] = []
            payload["data"] = cleaned
    return payload


def compress_assistant_response(response: str, tool_calls: List[str], user_message: str) -> Dict[str, Any]:
    compressed: Dict[str, Any] = {
        "tickers": [],
        "periods": [],
        "metrics": {},
        "tools": tool_calls[:5],
        "summary": "",
    }
    tickers = re.findall(r"\b[A-Z]{3,5}\b", f"{response} {user_message}")
    compressed["tickers"] = sorted({t for t in tickers})[:10]
    periods = re.findall(r"\b\d{4}-Q\d\b|\bQ\d\s*\d{4}\b|\b\dQ\d{2}\b|\b20\d{2}\b", response)
    compressed["periods"] = sorted({p for p in periods})[:5]
    roe_match = re.search(r"ROE\s*([0-9.]+)%", response)
    if roe_match:
        compressed["metrics"]["ROE"] = f"{roe_match.group(1)}%"
    pe_match = re.search(r"PE\s*([0-9.]+)", response)
    if pe_match:
        compressed["metrics"]["PE"] = pe_match.group(1)
    if compressed["tickers"] and compressed["periods"]:
        compressed["summary"] = f"{compressed['tickers'][0]} {compressed['periods'][0]}"
    elif compressed["tickers"]:
        compressed["summary"] = f"Focused on {compressed['tickers'][0]}"
    else:
        compressed["summary"] = "Brokerage insight"
    return compressed


def reconstruct_context(history: List[Dict[str, Any]]) -> str:
    if not history:
        return ""
    parts: List[str] = []
    for item in history[-3:]:
        if item.get("role") == "user":
            content = item.get("content", "")
            snippet = f"{content[:120]}…" if len(content) > 120 else content
            parts.append(f"User asked: {snippet}")
        elif item.get("role") == "assistant_compressed":
            data = item.get("data", {})
            segments: List[str] = []
            tickers = data.get("tickers")
            periods = data.get("periods")
            if tickers:
                segments.append(f"Discussed {', '.join(tickers[:3])}")
            if periods:
                segments.append(f"for {', '.join(periods[:2])}")
            if segments:
                parts.append(" ".join(segments))
    return " | ".join(parts)


def _ensure_openai_client() -> None:
    if "brokerage_openai_client" in st.session_state:
        return
    try:
        st.session_state.brokerage_openai_client = get_openai_client()
        st.session_state.brokerage_openai_error = None
    except Exception as exc:  # noqa: BLE001
        st.session_state.brokerage_openai_client = None
        st.session_state.brokerage_openai_error = str(exc)


def _ensure_tool_system(force_reload: bool = False) -> None:
    global brokerage_mcp_module
    needs_init = "brokerage_tool_system" not in st.session_state or force_reload
    if not needs_init:
        return
    try:
        if st.session_state.brokerage_developer_mode or force_reload:
            brokerage_mcp_module = importlib.reload(brokerage_mcp_module)
        st.session_state.brokerage_tool_system = brokerage_mcp_module.BrokerageMCP()
        st.session_state.brokerage_tool_system_error = None
    except Exception as exc:  # noqa: BLE001
        st.session_state.brokerage_tool_system = None
        st.session_state.brokerage_tool_system_error = str(exc)


def _init_session() -> None:
    if "brokerage_messages" not in st.session_state:
        st.session_state.brokerage_messages = []
    if "brokerage_display_messages" not in st.session_state:
        st.session_state.brokerage_display_messages = []
    if "brokerage_tool_executions" not in st.session_state:
        st.session_state.brokerage_tool_executions = []
    if "brokerage_tool_cache" not in st.session_state:
        st.session_state.brokerage_tool_cache = {}
    if "brokerage_compressed_history" not in st.session_state:
        st.session_state.brokerage_compressed_history = []
    if "brokerage_selected_model" not in st.session_state:
        st.session_state.brokerage_selected_model = _default_model_name()
    if "brokerage_developer_mode" not in st.session_state:
        st.session_state.brokerage_developer_mode = False
    if "brokerage_tool_cache_ttl" not in st.session_state:
        st.session_state.brokerage_tool_cache_ttl = int(os.getenv("UI_TOOL_CACHE_TTL", "300"))
    if "brokerage_usage" not in st.session_state:
        st.session_state.brokerage_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }
    _ensure_openai_client()
    _ensure_tool_system(force_reload=st.session_state.brokerage_developer_mode)


def _render_history() -> None:
    for message in st.session_state.brokerage_display_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def _update_usage_totals(usage: Any, model: str) -> None:
    if usage is None:
        return
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or prompt_tokens + completion_tokens
    st.session_state.brokerage_usage["prompt_tokens"] += prompt_tokens
    st.session_state.brokerage_usage["completion_tokens"] += completion_tokens
    st.session_state.brokerage_usage["total_tokens"] += total_tokens
    st.session_state.brokerage_usage["estimated_cost"] += calculate_cost(
        prompt_tokens,
        completion_tokens,
        model,
    )


def chat_with_brokerage(user_message: str) -> str:
    client = st.session_state.brokerage_openai_client
    tool_system = st.session_state.brokerage_tool_system
    if not client or not tool_system:
        return "Tooling or OpenAI client is unavailable."
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    context = reconstruct_context(st.session_state.brokerage_compressed_history)
    if context:
        messages.append({"role": "system", "content": f"Recent context: {context}"})
    messages.extend(st.session_state.brokerage_messages)
    tool_status_container = st.container()
    tool_calls_made: List[str] = []
    max_rounds = 20
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        try:
            completion = client.chat.completions.create(
                model=st.session_state.brokerage_selected_model,
                messages=messages,
                tools=tool_system.tool_specs,
                tool_choice="auto",
            )
        except Exception as exc:  # noqa: BLE001
            err_name = exc.__class__.__name__
            err_msg = getattr(exc, "message", None) or str(exc)
            sanitized = err_msg if err_msg else "unknown error"
            fallback = (
                f"OpenAI request failed ({err_name}). Check the selected model/quota. "
                f"Details: {sanitized}"
            )
            assistant_entry = {"role": "assistant", "content": fallback}
            st.session_state.brokerage_messages.append(assistant_entry)
            st.session_state.brokerage_compressed_history.append(
                {
                    "role": "assistant_compressed",
                    "data": {
                        "summary": "OpenAI error",
                        "tools": [],
                        "tickers": [],
                        "periods": [],
                        "metrics": {},
                    },
                }
            )
            if len(st.session_state.brokerage_compressed_history) > 20:
                st.session_state.brokerage_compressed_history = st.session_state.brokerage_compressed_history[-20:]
            st.session_state.brokerage_tool_executions.append(
                {
                    "tool": "openai_chat",
                    "status": "failed",
                    "error": err_msg,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return fallback

        _update_usage_totals(getattr(completion, "usage", None), st.session_state.brokerage_selected_model)

        message = completion.choices[0].message
        assistant_content = message.content or ""
        tool_calls = message.tool_calls or []

        assistant_payload: Dict[str, Any] = {"role": "assistant", "content": assistant_content or None}
        if tool_calls:
            assistant_payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]

        messages.append(assistant_payload)
        st.session_state.brokerage_messages.append(assistant_payload)

        if tool_calls:
            status_lines: List[str] = []
            for call in tool_calls:
                tool_name = call.function.name or ""
                raw_args = call.function.arguments or "{}"
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls_made.append(tool_name)
                cache_key = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
                cached = _cache_get(cache_key)
                if cached is None:
                    result = tool_system.execute_tool(tool_name, arguments)
                    _cache_set(cache_key, result)
                    cached_ui = False
                else:
                    result = cached
                    cached_ui = True
                log_entry: Dict[str, Any] = {
                    "tool": tool_name,
                    "arguments": arguments,
                    "timestamp": datetime.utcnow().isoformat(),
                    "status": result.get("status"),
                    "cached_tool": result.get("cached"),
                    "cached_ui": cached_ui,
                }
                if result.get("error"):
                    log_entry["error"] = result["error"]
                data_block = result.get("data")
                if isinstance(data_block, dict):
                    rows = data_block.get("rows")
                    if isinstance(rows, list):
                        log_entry["rows"] = len(rows)
                    log_entry["data_keys"] = list(data_block.keys())[:5]
                st.session_state.brokerage_tool_executions.append(log_entry)
                status_text = (
                    f"✓ {tool_name}" if result.get("status") == "success" else f"✗ {tool_name}: {result.get('error', 'failed')}"
                )
                status_lines.append(status_text)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(compact_tool_result_for_llm(result), ensure_ascii=False, default=str),
                }
                messages.append(tool_message)
                st.session_state.brokerage_messages.append(tool_message)
            with tool_status_container:
                for line in status_lines:
                    st.caption(line)
            continue

        final_text = assistant_content.strip() or "No direct response returned."
        st.session_state.brokerage_compressed_history.append(
            {
                "role": "assistant_compressed",
                "data": compress_assistant_response(final_text, tool_calls_made, user_message),
            }
        )
        if len(st.session_state.brokerage_compressed_history) > 20:
            st.session_state.brokerage_compressed_history = st.session_state.brokerage_compressed_history[-20:]
        return final_text

    fallback_text = "Unable to complete the request."
    assistant_entry = {"role": "assistant", "content": fallback_text}
    st.session_state.brokerage_messages.append(assistant_entry)
    return fallback_text


def main() -> None:
    _init_session()

    st.title("🧠 Brokerage Analyst Chatbot")
    st.caption("Ask anything about Vietnamese brokerage firms, financials, or valuations.")

    if st.session_state.brokerage_openai_client is None:
        st.error("OpenAI API key not configured.")
        st.info("Add OPENAI_API_KEY via environment or Streamlit secrets.")
        return
    if st.session_state.brokerage_tool_system is None:
        st.error("Tool system failed to initialise.")
        if st.session_state.get("brokerage_tool_system_error"):
            st.warning(st.session_state.brokerage_tool_system_error)
        return

    with st.sidebar:
        st.header("Configuration")
        dev_toggle = st.toggle(
            "Developer mode (reload tools)",
            value=st.session_state.brokerage_developer_mode,
            help="Reload brokerage MCP module on every run.",
        )
        if dev_toggle != st.session_state.brokerage_developer_mode:
            st.session_state.brokerage_developer_mode = dev_toggle
            _ensure_tool_system(force_reload=True)
            st.experimental_rerun()
        model_options = SUPPORTED_MODELS.copy()
        if st.session_state.brokerage_selected_model not in model_options:
            model_options.insert(0, st.session_state.brokerage_selected_model)
        selected_model = st.selectbox(
            "Select model",
            options=model_options,
            index=model_options.index(st.session_state.brokerage_selected_model),
        )
        if selected_model != st.session_state.brokerage_selected_model:
            st.session_state.brokerage_selected_model = selected_model
        if st.button("Clear conversation"):
            st.session_state.brokerage_messages = []
            st.session_state.brokerage_display_messages = []
            st.session_state.brokerage_tool_executions = []
            st.session_state.brokerage_tool_cache = {}
            st.session_state.brokerage_compressed_history = []
            st.session_state.brokerage_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost": 0.0,
            }
            st.experimental_rerun()
        st.divider()
        st.subheader("Tools")
        tool_specs = {
            spec["function"]["name"]: spec["function"].get("description", "")
            for spec in st.session_state.brokerage_tool_system.tool_specs
        }
        st.json(tool_specs)
        st.divider()
        st.subheader("Usage")
        usage = st.session_state.brokerage_usage
        st.metric("Prompt tokens", usage["prompt_tokens"])
        st.metric("Completion tokens", usage["completion_tokens"])
        st.metric("Estimated cost (USD)", f"{usage['estimated_cost']:.4f}")
        st.divider()
        if st.button("Clear tool history"):
            st.session_state.brokerage_tool_executions = []
            st.experimental_rerun()
        if st.session_state.brokerage_tool_executions:
            st.download_button(
                "Download tool history",
                json.dumps(st.session_state.brokerage_tool_executions, indent=2, default=str),
                "brokerage_tool_history.json",
                "application/json",
            )

    st.info(
        "**Guidance:** Be specific about metrics or valuation focus. Available dimensions include historical metrics, forecasts, valuations, stock performance, commentary, and coverage gaps."
    )

    _render_history()

    if st.session_state.brokerage_tool_executions:
        with st.expander(
            f"Tool execution history ({len(st.session_state.brokerage_tool_executions)} records)",
            expanded=False,
        ):
            for entry in reversed(st.session_state.brokerage_tool_executions[-12:]):
                status = entry.get("status", "?")
                icon = "✅" if status == "success" else "⚠️" if status else "ℹ️"
                cached_flags = []
                if entry.get("cached_tool"):
                    cached_flags.append("tool-cache")
                if entry.get("cached_ui"):
                    cached_flags.append("ui-cache")
                suffix = f" ({', '.join(cached_flags)})" if cached_flags else ""
                st.markdown(f"{icon} **{entry.get('tool')}**{suffix}")
                st.markdown(f"• Status: `{status}`")
                st.markdown(f"• Arguments: `{entry.get('arguments')}`")
                if entry.get("rows") is not None:
                    st.markdown(f"• Rows: `{entry['rows']}`")
                if entry.get("data_keys"):
                    st.markdown(f"• Data keys: `{entry['data_keys']}`")
                if entry.get("error"):
                    st.markdown(f"• Error: `{entry['error']}`")
                st.markdown(f"• Timestamp: `{entry.get('timestamp')}`")
                st.markdown("---")

    query = st.chat_input("Ask about brokers, metrics, or valuations…")
    if not query:
        return

    user_entry = {"role": "user", "content": query}
    st.session_state.brokerage_messages.append(user_entry)
    st.session_state.brokerage_display_messages.append(user_entry)
    st.session_state.brokerage_compressed_history.append({"role": "user", "content": query})
    if len(st.session_state.brokerage_compressed_history) > 20:
        st.session_state.brokerage_compressed_history = st.session_state.brokerage_compressed_history[-20:]

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            response_text = chat_with_brokerage(query)
        st.write(response_text)

    st.session_state.brokerage_display_messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
