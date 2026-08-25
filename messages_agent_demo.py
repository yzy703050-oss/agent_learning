from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from job_match_agent.llm import create_llm


@tool
def calculate_skill_overlap(job_skills: list[str], resume_skills: list[str]) -> dict:
    """Calculate the overlap between job skills and resume skills."""
    job_set = {skill.lower().strip() for skill in job_skills}
    resume_set = {skill.lower().strip() for skill in resume_skills}
    matched = sorted(job_set & resume_set)
    missing = sorted(job_set - resume_set)

    if not job_set:
        score = 0
    else:
        score = round(len(matched) / len(job_set) * 100)

    return {
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
    }


TOOLS = [calculate_skill_overlap]


def chatbot_node(state: MessagesState) -> dict:
    llm_with_tools = create_llm().bind_tools(TOOLS)
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def create_messages_agent_graph():
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_node("tools", ToolNode(TOOLS))

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder.compile()


def print_messages(messages) -> None:
    for index, message in enumerate(messages, start=1):
        print(f"\n--- Message {index}: {message.type} ---")
        print(message.content)

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            print("Tool calls:")
            for tool_call in tool_calls:
                print(f"- {tool_call}")


def main() -> None:
    graph = create_messages_agent_graph()
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "我的简历技能是 Python, SQL, Tableau。"
                        "岗位要求 Python, SQL, Power BI, A/B testing。"
                        "请调用工具计算匹配度，然后用中文解释结果。"
                    )
                )
            ]
        }
    )
    print_messages(result["messages"])


if __name__ == "__main__":
    main()
