# Please install OpenAI SDK first: `pip3 install openai`
import os
from datetime import datetime
from openai import OpenAI
import json


def create_deepseek_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the DEEPSEEK_API_KEY environment variable first.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


def ask_llm(
    user_prompt,
    stream_output=True,
    system_prompt="You are a helpful assistant",
    model="deepseek-v4-pro",
    thinking=True,
    reasoning_effort="high",
    show_reasoning=False,
    print_output=True,
    json_mode=False,
):
    client = create_deepseek_client()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    request_args = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"{system_prompt}\nCurrent local time: {current_time}",
            },
            {"role": "user", "content": user_prompt},
        ],
        "stream": stream_output,
        "extra_body": {
            "thinking": {"type": "enabled" if thinking else "disabled"},
        },
    }

    if json_mode:
        request_args["response_format"] = {"type": "json_object"}

    if thinking:
        request_args["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**request_args)

    if stream_output:
        full_content = ""
        full_reasoning = ""
        printed_reasoning = False
        printed_answer = False

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            reasoning_content = getattr(delta, "reasoning_content", None)
            content = getattr(delta, "content", None)

            if reasoning_content:
                full_reasoning += reasoning_content
                if show_reasoning:
                    if not printed_reasoning:
                        print("[reasoning]")
                        printed_reasoning = True
                    print(reasoning_content, end="", flush=True)
                continue

            if content:
                if show_reasoning and printed_reasoning and not printed_answer:
                    print("\n[answer]")
                    printed_answer = True

                full_content += content
                if print_output:
                    print(content, end="", flush=True)

        if print_output:
            print()
        if not full_content:
            if print_output:
                print("No final answer returned. Try setting thinking=False.")
        return full_content

    content = response.choices[0].message.content
    if not content:
        reasoning_content = getattr(response.choices[0].message, "reasoning_content", "")
        if show_reasoning and reasoning_content:
            print(reasoning_content)
        if print_output:
            print("No final answer returned. Try setting thinking=False.")
        return ""

    if print_output:
        print(content)
    return content
def calculate_fit_score(required_skills: list[str], resume_skills: list[str]) -> dict:
    required_set = {skill.lower().strip() for skill in required_skills}
    resume_set = {skill.lower().strip() for skill in resume_skills}

    matched = required_set & resume_set
    missing = required_set - resume_set

    if len(required_set) == 0:
        score = 0.0
    else:
        score = len(matched) / len(required_set)

    return {
        "score": round(score, 2),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing)
    }


def extract_json_object(text: str) -> dict:
    decoder = json.JSONDecoder()
    stripped_text = text.strip()

    try:
        result, _ = decoder.raw_decode(stripped_text)
        if isinstance(result, dict):
            return result

        raise ValueError("JSON root must be an object.")
    except json.JSONDecodeError:
        pass

    start = None
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
            continue

        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == "\"":
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:index + 1])

    raise json.JSONDecodeError("No valid JSON object found.", text, 0)


def validate_job_info(job_info: dict) -> dict:
    if not isinstance(job_info, dict):
        raise ValueError(f"job_info must be dict, got {type(job_info).__name__}.")

    required_fields = {
        "job_title": str,
        "required_skills": list,
        "experience_level": str,
        "authorization_risk": str,
        "entry_level_fit": str,
    }

    for field, expected_type in required_fields.items():
        if field not in job_info:
            raise ValueError(f"Missing field: {field}")
        if not isinstance(job_info[field], expected_type):
            raise ValueError(
                f"Field {field} must be {expected_type.__name__}, "
                f"got {type(job_info[field]).__name__}."
            )

    job_info["required_skills"] = [
        str(skill).strip()
        for skill in job_info["required_skills"]
        if str(skill).strip()
    ]

    return job_info


def extract_job_info_with_retry(jd: str, max_attempts: int = 3) -> dict:
    last_error = None

    for attempt in range(1, max_attempts + 1):
        prompt = build_job_extraction_prompt(jd)

        if last_error:
            prompt += f"""

            Previous attempt failed with this error:
            {last_error}

            Please fix the output. Return only one valid JSON object with all required fields.
            """

        answer = ask_llm(
            prompt,
            stream_output=False,
            thinking=False,
            print_output=False,
            json_mode=True,
        )

        try:
            job_info = extract_json_object(answer)
            return validate_job_info(job_info)
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error
            print(f"[extract_job_info retry {attempt}] {error}")

    raise RuntimeError(f"Failed to extract valid job info after {max_attempts} attempts: {last_error}")


def build_job_extraction_prompt(jd: str) -> str:
    return f"""
    Extract the following job description into valid JSON.

    Job Description:
    {jd}

    Return only valid JSON. Do not include markdown. Do not include explanation.

    JSON format:
    {{
      "job_title": "",
      "required_skills": [],
      "experience_level": "",
      "authorization_risk": "",
      "entry_level_fit": ""
    }}
    """


def get_allowed_actions(state: dict) -> list[str]:
    allowed_actions = []

    if state["job_info"] is None:
        allowed_actions.append("extract_job_info")

    if state["job_info"] is not None and state["fit_result"] is None:
        allowed_actions.append("calculate_fit_score")

    if (
        state["job_info"] is not None
        and state["fit_result"] is not None
        and state["final_answer"] is None
    ):
        allowed_actions.append("final_answer")

    return allowed_actions


def rule_based_next_action(state: dict) -> str:
    allowed_actions = get_allowed_actions(state)
    if not allowed_actions:
        raise ValueError("No allowed actions available.")

    return allowed_actions[0]


def summarize_state_for_planner(state: dict) -> dict:
    job_info = state["job_info"] or {}
    fit_result = state["fit_result"] or {}

    return {
        "job_info_ready": state["job_info"] is not None,
        "fit_result_ready": state["fit_result"] is not None,
        "final_answer_ready": state["final_answer"] is not None,
        "allowed_actions": get_allowed_actions(state),
        "completed_steps": [step["action"] for step in state["steps"]],
        "recent_errors": state["errors"][-2:],
        "job_title": job_info.get("job_title"),
        "required_skills": job_info.get("required_skills"),
        "fit_score": fit_result.get("score"),
    }


def build_tool_call_planner_messages(state: dict) -> list[dict]:
    state_summary = summarize_state_for_planner(state)

    return [
        {
            "role": "system",
            "content": (
                "You are the planner for a job-matching agent. "
                "Choose the next action by calling exactly one of the provided tools. "
                "Do not answer in natural language."
            ),
        },
        {
            "role": "user",
            "content": (
                "Current state:\n"
                f"{json.dumps(state_summary, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def build_planner_tool_schema(action: str) -> dict:
    descriptions = {
        "extract_job_info": "Extract structured job information from the job description.",
        "calculate_fit_score": "Compare required skills with resume skills.",
        "final_answer": "Create the final user-facing summary.",
    }

    return {
        "type": "function",
        "function": {
            "name": action,
            "description": descriptions[action],
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why this is the correct next action.",
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    }


def build_planner_tools(state: dict) -> list[dict]:
    return [
        build_planner_tool_schema(action)
        for action in get_allowed_actions(state)
    ]


def validate_tool_call_decision(tool_call, state: dict) -> dict:
    if tool_call.type != "function":
        raise ValueError(f"Unsupported tool call type: {tool_call.type}")

    action = tool_call.function.name
    allowed_actions = get_allowed_actions(state)

    if not isinstance(action, str) or not action.strip():
        raise ValueError("Planner tool call must include a non-empty function name.")

    action = action.strip()
    if action not in TOOL_REGISTRY:
        raise ValueError(f"Unknown planner tool: {action}")

    if action not in allowed_actions:
        raise ValueError(
            f"Action {action} is not allowed. Allowed actions: {allowed_actions}"
        )

    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Planner tool arguments are not valid JSON: {error}") from error

    reason = arguments.get("reason", "")

    return {
        "action": action,
        "reason": str(reason),
        "source": "tool_call",
        "tool_call_id": tool_call.id,
    }


def decide_next_action_with_tool_call(state: dict) -> dict:
    tools = build_planner_tools(state)
    if not tools:
        raise ValueError("No planner tools available.")

    client = create_deepseek_client()
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=build_tool_call_planner_messages(state),
        tools=tools,
        tool_choice="required",
        stream=False,
        extra_body={
            "thinking": {"type": "disabled"},
        },
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        raise ValueError("Planner did not return a tool call.")

    if len(tool_calls) > 1:
        raise ValueError("Planner returned more than one tool call.")

    return validate_tool_call_decision(tool_calls[0], state)


def decide_next_action(state: dict, use_llm: bool = True) -> dict:
    if use_llm:
        try:
            return decide_next_action_with_tool_call(state)
        except Exception as error:
            fallback_action = rule_based_next_action(state)
            return {
                "action": fallback_action,
                "reason": f"Tool-calling planner failed, used rule-based fallback: {error}",
                "source": "fallback",
                "error": str(error),
            }

    return {
        "action": rule_based_next_action(state),
        "reason": "Rule-based planner is enabled.",
        "source": "rule",
    }


def build_final_summary(job_info: dict, fit_result: dict) -> str:
    score = fit_result["score"]
    if score >= 0.75:
        fit_level = "比较匹配"
    elif score >= 0.5:
        fit_level = "部分匹配"
    else:
        fit_level = "暂时不太匹配"

    return (
        f"结论：{fit_level}。\n"
        f"职位：{job_info['job_title']}\n"
        f"匹配分数：{score}\n"
        f"已匹配技能：{', '.join(fit_result['matched_skills']) or '无'}\n"
        f"缺失技能：{', '.join(fit_result['missing_skills']) or '无'}\n"
        f"经验要求：{job_info['experience_level']}\n"
        f"入门适合度：{job_info['entry_level_fit']}"
    )


def tool_extract_job_info(state: dict):
    state["job_info"] = extract_job_info_with_retry(state["jd"])
    return "Job information extracted."


def tool_calculate_fit_score(state: dict):
    state["fit_result"] = calculate_fit_score(
        required_skills=state["job_info"]["required_skills"],
        resume_skills=state["resume_skills"],
    )
    return state["fit_result"]


def tool_final_answer(state: dict):
    state["final_answer"] = build_final_summary(
        state["job_info"],
        state["fit_result"],
    )
    return "Final answer generated."


TOOL_REGISTRY = {
    "extract_job_info": tool_extract_job_info,
    "calculate_fit_score": tool_calculate_fit_score,
    "final_answer": tool_final_answer,
}


def run_tool(action: str, state: dict):
    tool = TOOL_REGISTRY.get(action)
    if tool is None:
        raise ValueError(f"Unknown action: {action}")

    return tool(state)


def job_agent_loop(
    jd: str,
    resume_skills: list[str],
    max_steps: int = 5,
    use_llm_planner: bool = True,
) -> dict:
    state = {
        "jd": jd,
        "resume_skills": resume_skills,
        "job_info": None,
        "fit_result": None,
        "final_answer": None,
        "steps": [],
        "errors": [],
    }

    for step in range(1, max_steps + 1):
        action = "decide_next_action"
        decision = {
            "source": "unknown",
            "reason": "",
        }
        try:
            decision = decide_next_action(state, use_llm=use_llm_planner)
            action = decision["action"]
            print(f"[agent step {step}] action = {action} ({decision['source']})")

            if "error" in decision:
                state["errors"].append(
                    {
                        "step": step,
                        "action": "decide_next_action",
                        "error": decision["error"],
                    }
                )

            observation = run_tool(action, state)
            state["steps"].append(
                {
                    "action": action,
                    "observation": observation,
                    "planner": {
                        "source": decision["source"],
                        "reason": decision["reason"],
                    },
                }
            )

            if action == "final_answer":
                return state

            continue
        except Exception as error:
            error_info = {
                "step": step,
                "action": action,
                "error": str(error),
            }
            state["errors"].append(error_info)
            state["steps"].append(
                {
                    "action": action,
                    "observation": "Action failed.",
                    "error": str(error),
                    "planner": {
                        "source": decision["source"],
                        "reason": decision["reason"],
                    },
                }
            )
            state["final_answer"] = f"分析失败：{error}"
            return state

    state["errors"].append(
        {
            "step": max_steps,
            "action": "max_steps",
            "error": "Agent loop reached max_steps before finishing.",
        }
    )
    state["final_answer"] = "分析失败：Agent loop reached max_steps before finishing."
    return state


if __name__ == "__main__":
    resume_skills = [
        "Python",
        "SQL",
        "Tableau",
        "Power BI",
        "A/B testing",
        "Excel",
        "Pandas"
    ]

    job_descriptions = [
        """
        We are looking for a Data Analyst with experience in Python, SQL, Tableau, and A/B testing.
        0-2 years of experience preferred. No sponsorship is available.
        """,
        """
        We are hiring a Business Analyst with Excel, SQL, Power BI, and stakeholder communication skills.
        Entry-level candidates are welcome.
        """,
        """
        We need a Senior Data Engineer with Spark, Airflow, AWS, and 5+ years of experience.
        """
    ]

    for i, jd in enumerate(job_descriptions, start=1):
        print(f"\n========== Job {i} ==========")

        result = job_agent_loop(jd, resume_skills)
        print(result["final_answer"])
