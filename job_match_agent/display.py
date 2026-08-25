import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel


def get_final_text(agent_result: dict) -> str:
    final_message = agent_result["messages"][-1]
    content = getattr(final_message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
            else:
                parts.append(str(block))
        return "".join(parts)

    return str(content)


def print_message_trace(agent_result: dict) -> None:
    for index, message in enumerate(agent_result["messages"], start=1):
        print(f"\n--- Message {index}: {message.type} ---")

        if isinstance(message, HumanMessage):
            print("Human input:")
            print(message.content)
            continue

        if isinstance(message, AIMessage):
            if message.content:
                print("AI content:")
                print(message.content)

            if message.tool_calls:
                print("AI tool calls:")
                for tool_call in message.tool_calls:
                    print(f"- name: {tool_call.get('name')}")
                    print(f"  args: {tool_call.get('args')}")
                    print(f"  id: {tool_call.get('id')}")
            continue

        if isinstance(message, ToolMessage):
            print(f"Tool result for tool_call_id={message.tool_call_id}:")
            print(message.content)
            continue

        content = getattr(message, "content", "")
        if content:
            print(content)

        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls:
            print("raw tool_calls:")
            for tool_call in raw_tool_calls:
                print(tool_call)


def print_structured_response(agent_result: dict) -> None:
    print("--- Structured Response ---")

    if "structured_response" not in agent_result:
        print("No structured_response found in agent result.")
        return

    structured_response = agent_result["structured_response"]
    if isinstance(structured_response, BaseModel):
        print(structured_response.model_dump_json(indent=2))
        return

    print(json.dumps(structured_response, ensure_ascii=False, indent=2))

