"""Interactive brokerage analytics chatbot powered by MCP-style tools."""

import json
import os
from typing import Any, Dict, List

import streamlit as st

from utils.brokerage_mcp import BrokerageMCP
from utils.openai_commentary import get_openai_client


st.set_page_config(page_title="Brokerage GPT", page_icon="🧠", layout="wide")


SYSTEM_PROMPT = (
    "You are an expert brokerage analyst. Use the available tools to fetch data "
    "and deliver concise, well-structured answers. If a tool returns no data, "
    "explain the limitation and suggest alternative analyses."
)


def _init_session() -> None:
    if "brokerage_chat_history" not in st.session_state:
        st.session_state.brokerage_chat_history = []
    if "brokerage_tool_executions" not in st.session_state:
        st.session_state.brokerage_tool_executions = []
    if "brokerage_mcp" not in st.session_state:
        st.session_state.brokerage_mcp = BrokerageMCP()


def _get_model_name() -> str:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        try:
            model = st.secrets["openai"].get("model", "gpt-4o-mini")
        except Exception:  # noqa: BLE001
            model = "gpt-4o-mini"
    return model


def _append_history(role: str, content: str) -> None:
    st.session_state.brokerage_chat_history.append({"role": role, "content": content})


def _append_tool_log(entry: Dict[str, Any]) -> None:
    st.session_state.brokerage_tool_executions.append(entry)


def _render_history() -> None:
    for message in st.session_state.brokerage_chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def _handle_tool_calls(
    mcp: BrokerageMCP,
    response_message,
    messages: List[Dict[str, str]],
) -> None:
    tool_calls = response_message.tool_calls or []
    for call in tool_calls:
        tool_name = call.function.name
        try:
            arguments = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            arguments = {}
        result = mcp.execute_tool(tool_name, arguments)
        payload = json.dumps(result, ensure_ascii=False)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": payload,
            }
        )
        _append_tool_log(
            {
                "tool": tool_name,
                "arguments": arguments,
                "result": result,
            }
        )


def _run_chat_cycle(prompt: str) -> str:
    client = get_openai_client()
    mcp: BrokerageMCP = st.session_state.brokerage_mcp
    model_name = _get_model_name()

    messages: List[Dict[str, Any]] = (
        [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.brokerage_chat_history
    )
    messages.append({"role": "user", "content": prompt})

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=mcp.tool_specs,
    )

    message = completion.choices[0].message

    if message.tool_calls:
        _handle_tool_calls(mcp, message, messages)
        follow_up = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        final_message = follow_up.choices[0].message
        return final_message.content or ""

    return message.content or ""


def main() -> None:
    _init_session()

    st.title("🧠 Brokerage Analyst Chatbot")
    st.caption("Ask anything about Vietnamese brokerage firms, financials, or valuations.")

    with st.sidebar:
        st.subheader("Session")
        if st.button("Clear conversation"):
            st.session_state.brokerage_chat_history = []
            st.session_state.brokerage_tool_executions = []
            st.experimental_rerun()

        st.divider()
        st.subheader("Tools")
        st.json({spec["function"]["name"]: spec["function"]["description"] for spec in st.session_state.brokerage_mcp.tool_specs})

        if st.checkbox("Show tool execution log"):
            st.json(st.session_state.brokerage_tool_executions)

    _render_history()

    query = st.chat_input("Ask about brokers, metrics, or valuations…")
    if not query:
        return

    _append_history("user", query)
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            try:
                response_text = _run_chat_cycle(query)
            except Exception as exc:  # noqa: BLE001
                response_text = (
                    "I couldn't complete that request because the AI backend reported an error: "
                    f"{exc}."
                )
        st.write(response_text)

    _append_history("assistant", response_text)


if __name__ == "__main__":
    main()

