import json

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from .llm import create_llm
from .models import JobInfo, JobMatchResponse, JobMatchResult
from .prompts import (
    JOB_INFO_EXTRACTION_PROMPT,
    JOB_MATCH_PROMPT,
    JOB_MATCH_SUMMARY_PROMPT,
    build_job_match_system_message,
)
from .tools import calculate_fit_score
from .tracing import build_trace_config


def prompt_value_to_agent_input(prompt_value) -> dict:
    return {"messages": prompt_value.to_messages()}


def create_job_info_chain():
    llm = create_llm()
    structured_llm = llm.with_structured_output(JobInfo)
    return JOB_INFO_EXTRACTION_PROMPT | structured_llm


def extract_job_info_langchain(
    jd: str,
    tracing: bool = False,
    trace_tokens: bool = False,
) -> JobInfo:
    chain = create_job_info_chain()
    trace_config = build_trace_config(tracing, run_name="job-info-extractor")
    job_info = chain.invoke({"job_description": jd}, config=trace_config)
    if isinstance(job_info, JobInfo):
        return job_info

    return JobInfo.model_validate(job_info)


def parse_job_match_result(agent_result: dict) -> JobMatchResult:
    match_result = agent_result["structured_response"]
    if isinstance(match_result, JobMatchResult):
        return match_result
    if isinstance(match_result, JobMatchResponse):
        return match_result.match_result

    if isinstance(match_result, dict) and "match_result" in match_result:
        return JobMatchResponse.model_validate(match_result).match_result

    return JobMatchResult.model_validate(match_result)


def parse_job_match_response(agent_result: dict) -> JobMatchResponse:
    response = agent_result["structured_response"]
    if isinstance(response, JobMatchResponse):
        return response

    if isinstance(response, JobMatchResult):
        return JobMatchResponse(match_result=response, final_answer="")

    return JobMatchResponse.model_validate(response)


def create_job_match_agent(streaming: bool = False):
    llm = create_llm(streaming=streaming)

    return create_agent(
        model=llm,
        tools=[calculate_fit_score],
        response_format=ToolStrategy(JobMatchResponse),
        system_prompt=build_job_match_system_message(),
    )


def create_job_match_chain(streaming: bool = False):
    return (
        JOB_MATCH_PROMPT
        | RunnableLambda(prompt_value_to_agent_input)
        | create_job_match_agent(streaming=streaming)
    )


def create_job_match_summary_chain(streaming: bool = False):
    llm = create_llm(streaming=streaming)
    structured_llm = llm.with_structured_output(JobMatchResponse)
    return JOB_MATCH_SUMMARY_PROMPT | structured_llm


def build_job_match_variables(
    job_info: JobInfo,
    resume_skills: list[str],
    retrieved_context: str = "No retrieved context provided.",
    memory_context: str = "No memory context provided.",
) -> dict:
    return {
        "job_info": job_info.model_dump_json(indent=2),
        "resume_skills": json.dumps(resume_skills, ensure_ascii=False),
        "memory_context": memory_context,
        "retrieved_context": retrieved_context,
    }


def print_stream_message(message) -> None:
    if isinstance(message, HumanMessage):
        print("\n[stream] Human message sent to agent.")
        return

    if isinstance(message, AIMessage):
        if message.content:
            print("\n[stream] AI content:")
            print(message.content)

        if message.tool_calls:
            print("\n[stream] AI tool calls:")
            for tool_call in message.tool_calls:
                print(f"- name: {tool_call.get('name')}")
                print(f"  args: {tool_call.get('args')}")
        return

    if isinstance(message, ToolMessage):
        print(f"\n[stream] Tool result for tool_call_id={message.tool_call_id}:")
        print(message.content)
        return

    content = getattr(message, "content", "")
    if content:
        print("\n[stream] Message content:")
        print(content)


def stream_job_match_agent(
    job_info: JobInfo,
    resume_skills: list[str],
    retrieved_context: str = "No retrieved context provided.",
    memory_context: str = "No memory context provided.",
    tracing: bool = False,
    trace_tokens: bool = False,
) -> dict:
    agent = create_job_match_agent(streaming=True)
    trace_config = build_trace_config(tracing, run_name="job-match-agent-stream")
    prompt_value = JOB_MATCH_PROMPT.invoke(
        build_job_match_variables(
            job_info,
            resume_skills,
            retrieved_context,
            memory_context,
        )
    )
    agent_input = prompt_value_to_agent_input(prompt_value)

    final_result = {}
    seen_messages = 0

    print("\nStreaming Job Match Agent:")
    for state in agent.stream(agent_input, config=trace_config, stream_mode="values"):
        if not isinstance(state, dict):
            print(state)
            continue

        final_result = state
        messages = state.get("messages", [])
        new_messages = messages[seen_messages:]

        for message in new_messages:
            print_stream_message(message)

        seen_messages = len(messages)

        if "structured_response" in state:
            print("\n[stream] Structured response received.")

    return final_result
