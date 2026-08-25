# Please install first:
# pip install -U langchain langchain-deepseek
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Literal, TypedDict


try:
    import langsmith as ls
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langchain.tools import tool
    from langchain_deepseek import ChatDeepSeek
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "Missing LangChain packages. Install them with: "
        "pip install -U langchain langchain-deepseek langgraph"
    ) from exc


JOB_INFO_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You extract structured information from job descriptions. "
                "Extract only facts explicitly stated or strongly implied. "
                "Use 'Unknown' when job title or experience level is not stated. "
                "For authorization_risk, use high if the job description says no sponsorship, "
                "no visa sponsorship, US citizen required, green card required, "
                "or permanent resident required. "
                "Use medium if the job says must be authorized to work in the US, "
                "but does not clearly say no sponsorship. "
                "Use low if OPT, CPT, sponsorship, visa support, or work visa support "
                "is clearly allowed. "
                "Use unknown if work authorization is not mentioned. "
                "For entry_level_fit, use Yes for 0-2 years, junior, new grad, "
                "or entry-level roles. "
                "Use No for senior or 5+ years roles. "
                "Use Maybe if unclear. "
                "Return the structured response using the required schema. "
                "Do not add commentary."
            ),
        ),
        (
            "user",
            "Extract structured job information from this job description:\n{job_description}",
        )
    ]
)


JOB_MATCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            """
Analyze this structured job information against my resume skills.

Job Info:
{job_info}

Resume Skills:
{resume_skills}

Memory Context:
{memory_context}

Retrieved Context:
{retrieved_context}

Requirements:
1. Use the calculate_fit_score tool with Job Info.required_skills and Resume Skills.
2. Give a concise Chinese summary with job title, matched skills, missing skills, and fit score from 0 to 100.
3. Mention experience level.
4. Mention authorization risk.
5. Mention whether the role is entry-level friendly.
6. Use Memory Context only for stable user preferences or prior analysis background.
7. Use Retrieved Context as external career advice, but do not override the structured Job Info.
""",
        )
    ]
)


JOB_MATCH_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a job matching assistant. "
                "Create a structured JobMatchResult. "
                "Use the provided fit_score, matched_skills, and missing_skills exactly. "
                "Use the provided Job Info fields exactly. "
                "Write final_answer as a concise Chinese summary."
            ),
        ),
        (
            "user",
            """
Job Info:
{job_info}

Resume Skills:
{resume_skills}

Fit Score:
{fit_score}

Matched Skills:
{matched_skills}

Missing Skills:
{missing_skills}

Memory Context:
{memory_context}

Retrieved Context:
{retrieved_context}

Visa Warning:
{visa_warning}

Requirements:
1. Mention job title, matched skills, missing skills, and fit score.
2. Mention experience level.
3. Mention authorization risk.
4. Mention whether the role is entry-level friendly.
5. If Visa Warning is not empty, include it clearly in final_answer.
6. Use Memory Context only for stable user preferences or prior analysis background.
7. Use Retrieved Context as external career advice, but do not override Job Info or Fit Score.
""",
        ),
    ]
)


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "career_knowledge_base"


def build_job_match_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "You are a job matching assistant. "
            "You will receive structured job information and resume skills. "
            "Call calculate_fit_score using the required_skills from the job information. "
            "After the tool returns, create a structured JobMatchResult response. "
            "Use the tool result for fit_score, matched_skills, and missing_skills. "
            "Use the job information for job_title, experience_level, authorization_risk, "
            "and entry_level_fit. "
            "Write final_answer as a concise Chinese summary."
        )
    )


def prompt_value_to_agent_input(prompt_value) -> dict:
    return {"messages": prompt_value.to_messages()}


def get_langsmith_project_name() -> str:
    return os.environ.get("LANGSMITH_PROJECT", "job-agent-demo")


def build_trace_config(enable_tracing: bool, run_name: str) -> dict | None:
    if not enable_tracing:
        return None

    return {
        "run_name": run_name,
        "tags": ["job-agent-demo"],
        "metadata": {
            "project": get_langsmith_project_name(),
            "demo": "job-agent",
        },
    }


class JobInfo(BaseModel):
    job_title: str = Field(
        description="The explicit job title. Use 'Unknown' if not stated."
    )
    required_skills: list[str] = Field(
        description="Only explicit required or preferred skills mentioned in the job description."
    )
    experience_level: str = Field(
        description="The explicit experience requirement. Use 'Unknown' if not stated."
    )
    authorization_risk: Literal["low", "medium", "high", "unknown"] = Field(
        description=(
            "Visa, sponsorship, or work authorization risk. "
            "Use high if the job says no sponsorship, no visa sponsorship, "
            "US citizen required, green card required, or permanent resident required. "
            "Use medium if the job says must be authorized to work in the US, "
            "but does not clearly say no sponsorship. "
            "Use low if OPT, CPT, sponsorship, visa support, or work visa support is clearly allowed. "
            "Use unknown if work authorization is not mentioned."
        )
    )
    entry_level_fit: Literal["Yes", "No", "Maybe"] = Field(
        description=(
            "Whether this role is suitable for an entry-level candidate. "
            "Use Yes for 0-2 years, junior, new grad, or entry-level roles. "
            "Use No for senior roles or 5+ years of experience. "
            "Use Maybe if unclear."
        )
    )


class JobMatchResult(BaseModel):
    job_title: str = Field(description="The job title from JobInfo.")
    fit_score: int = Field(description="The 0 to 100 score returned by calculate_fit_score.")
    matched_skills: list[str] = Field(description="Skills found in both the job and resume.")
    missing_skills: list[str] = Field(description="Required job skills missing from the resume.")
    experience_level: str = Field(description="The experience level from JobInfo.")
    authorization_risk: Literal["low", "medium", "high", "unknown"] = Field(
        description="The authorization risk from JobInfo."
    )
    entry_level_fit: Literal["Yes", "No", "Maybe"] = Field(
        description="The entry-level fit from JobInfo."
    )
    final_answer: str = Field(
        description="A concise Chinese summary for the user."
    )


class JobMatchMemory:
    """Small explicit memory store for learning how context enters a prompt."""

    def __init__(self, max_history: int = 3):
        self.max_history = max_history
        self.resume_skills: list[str] = []
        self.user_preferences: dict[str, str] = {}
        self.analysis_history: list[dict[str, Any]] = []
        self.compressed_history: dict[str, Any] = {
            "older_jobs_count": 0,
            "older_high_risk_count": 0,
            "best_fit_score": None,
            "best_fit_job_title": None,
            "missing_skill_counts": {},
        }

    def update_user_profile(
        self,
        resume_skills: list[str] | None = None,
        visa_preference: str | None = None,
        target_role: str | None = None,
    ) -> None:
        if resume_skills:
            self.resume_skills = resume_skills

        if visa_preference:
            self.user_preferences["visa_preference"] = visa_preference

        if target_role:
            self.user_preferences["target_role"] = target_role

    def remember_analysis(self, job_info: JobInfo, match_result: JobMatchResult) -> None:
        self.analysis_history.append(
            {
                "job_title": job_info.job_title,
                "fit_score": match_result.fit_score,
                "authorization_risk": job_info.authorization_risk,
                "entry_level_fit": job_info.entry_level_fit,
                "missing_skills": match_result.missing_skills,
            }
        )
        self.compress_old_history()

    def compress_old_history(self) -> None:
        while len(self.analysis_history) > self.max_history:
            old_item = self.analysis_history.pop(0)
            self.compressed_history["older_jobs_count"] += 1

            if old_item["authorization_risk"] == "high":
                self.compressed_history["older_high_risk_count"] += 1

            best_score = self.compressed_history["best_fit_score"]
            if best_score is None or old_item["fit_score"] > best_score:
                self.compressed_history["best_fit_score"] = old_item["fit_score"]
                self.compressed_history["best_fit_job_title"] = old_item["job_title"]

            missing_skill_counts = self.compressed_history["missing_skill_counts"]
            for skill in old_item["missing_skills"]:
                missing_skill_counts[skill] = missing_skill_counts.get(skill, 0) + 1

    def build_context(self) -> str:
        lines = []

        if self.resume_skills:
            lines.append(f"Remembered resume skills: {', '.join(self.resume_skills)}")

        if self.user_preferences:
            preferences = [
                f"{key}: {value}" for key, value in self.user_preferences.items()
            ]
            lines.append(f"Remembered user preferences: {'; '.join(preferences)}")

        if self.compressed_history["older_jobs_count"]:
            lines.append(
                "Compressed older analyses: "
                f"{self.compressed_history['older_jobs_count']} older jobs, "
                f"{self.compressed_history['older_high_risk_count']} high authorization risk."
            )

            if self.compressed_history["best_fit_job_title"]:
                lines.append(
                    "Best older fit: "
                    f"{self.compressed_history['best_fit_job_title']} "
                    f"with score {self.compressed_history['best_fit_score']}."
                )

            missing_skill_counts = self.compressed_history["missing_skill_counts"]
            if missing_skill_counts:
                common_missing_skills = sorted(
                    missing_skill_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
                formatted_skills = [
                    f"{skill} ({count})" for skill, count in common_missing_skills
                ]
                lines.append(
                    "Common missing skills from older analyses: "
                    f"{', '.join(formatted_skills)}"
                )

        if self.analysis_history:
            lines.append("Recent job analyses:")
            for item in self.analysis_history:
                lines.append(
                    "- "
                    f"{item['job_title']}: fit_score={item['fit_score']}, "
                    f"authorization_risk={item['authorization_risk']}, "
                    f"entry_level_fit={item['entry_level_fit']}"
                )

        if not lines:
            return "No memory context yet."

        return "\n".join(lines)


def create_llm(streaming: bool = False):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("Please set the DEEPSEEK_API_KEY environment variable first.")

    return ChatDeepSeek(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
        streaming=streaming,
    )


@tool
def calculate_fit_score(required_skills: list[str], resume_skills: list[str]) -> dict:
    """Calculate how well resume skills match the job's required skills."""
    required_set = {skill.lower().strip() for skill in required_skills}
    resume_set = {skill.lower().strip() for skill in resume_skills}

    matched = required_set & resume_set
    missing = required_set - resume_set

    if len(required_set) == 0:
        score = 0
    else:
        score = round(len(matched) / len(required_set) * 100)

    return {
        "fit_score": score,
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
    }


def create_job_info_chain():
    llm = create_llm()
    structured_llm = llm.with_structured_output(JobInfo)
    return JOB_INFO_EXTRACTION_PROMPT | structured_llm


class LocalHashEmbeddings(Embeddings):
    """Small local embedding model for demos without another API key or package."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9+#./-]+", text.lower())

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        length = math.sqrt(sum(value * value for value in vector))
        if length == 0:
            return vector

        return [value / length for value in vector]


class LocalVectorStore:
    """Tiny in-process vector store for learning the RAG data flow."""

    def __init__(self, embedding: Embeddings):
        self.embedding = embedding
        self._items = []

    def add_documents(self, documents: list[Document]) -> None:
        vectors = self.embedding.embed_documents(
            [document.page_content for document in documents]
        )
        for document, vector in zip(documents, vectors):
            self._items.append(
                {
                    "document": document,
                    "vector": vector,
                }
            )

    def similarity_search(self, query: str, k: int = 2) -> list[Document]:
        query_vector = self.embedding.embed_query(query)
        scored_items = [
            (self._cosine_similarity(query_vector, item["vector"]), item["document"])
            for item in self._items
        ]
        scored_items.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored_items[:k]]

    def as_retriever(self, search_kwargs: dict | None = None):
        search_kwargs = search_kwargs or {}
        k = search_kwargs.get("k", 2)
        return RunnableLambda(lambda query: self.similarity_search(query, k=k))

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        return sum(left_value * right_value for left_value, right_value in zip(left, right))


def extract_markdown_title(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()

    return fallback


def load_markdown_documents(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> list[Document]:
    documents = []
    for path in sorted(kb_dir.glob("*.md")):
        markdown_text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=markdown_text,
                metadata={
                    "source": str(path),
                    "title": extract_markdown_title(markdown_text, path.stem),
                },
            )
        )

    if not documents:
        raise RuntimeError(
            f"No Markdown knowledge files found in {kb_dir}. "
            "Add one or more .md files before running RAG."
        )

    return documents


def split_document(document: Document, chunk_size: int = 700, overlap: int = 120) -> list[Document]:
    text = document.page_content.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            paragraph_break = text.rfind("\n\n", start, end)
            if paragraph_break > start + chunk_size // 2:
                end = paragraph_break

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_index = len(chunks)
            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk": chunk_index,
                    },
                )
            )

        if end >= len(text):
            break

        start = max(end - overlap, 0)

    return chunks


def build_vector_store(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> LocalVectorStore:
    documents = load_markdown_documents(kb_dir)
    chunks = []
    for document in documents:
        chunks.extend(split_document(document))

    vector_store = LocalVectorStore(embedding=LocalHashEmbeddings())
    vector_store.add_documents(chunks)
    return vector_store


def build_job_context_query(job_info: JobInfo, resume_skills: list[str]) -> str:
    return " ".join(
        [
            job_info.job_title,
            job_info.experience_level,
            job_info.authorization_risk,
            job_info.entry_level_fit,
            " ".join(job_info.required_skills),
            " ".join(resume_skills),
        ]
    )


def format_retrieved_documents(documents: list[Document]) -> str:
    if not documents:
        return "No relevant context found."

    formatted_docs = []
    for document in documents:
        title = document.metadata.get("title", "Untitled")
        source = Path(document.metadata.get("source", "")).name
        content = " ".join(document.page_content.split())
        formatted_docs.append(f"- {title} ({source}): {content}")

    return "\n".join(formatted_docs)


def retrieve_job_context(job_info: JobInfo, resume_skills: list[str], top_k: int = 2) -> str:
    vector_store = build_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    query_text = build_job_context_query(job_info, resume_skills)
    documents = retriever.invoke(query_text)
    return format_retrieved_documents(documents)


def create_job_context_retriever_chain():
    return RunnableLambda(
        lambda inputs: retrieve_job_context(
            inputs["job_info"],
            inputs["resume_skills"],
        )
    ).with_config({"run_name": "job-context-retriever"})


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

    return JobMatchResult.model_validate(match_result)


def create_job_match_agent(streaming: bool = False):
    llm = create_llm(streaming=streaming)

    return create_agent(
        model=llm,
        tools=[calculate_fit_score],
        response_format=ToolStrategy(JobMatchResult),
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
    structured_llm = llm.with_structured_output(JobMatchResult)
    return JOB_MATCH_SUMMARY_PROMPT | structured_llm


def ensure_job_info(value: JobInfo | dict) -> JobInfo:
    if isinstance(value, JobInfo):
        return value

    return JobInfo.model_validate(value)


def ensure_job_match_result(value: JobMatchResult | dict) -> JobMatchResult:
    if isinstance(value, JobMatchResult):
        return value

    return JobMatchResult.model_validate(value)


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


class JobMatchGraphState(TypedDict, total=False):
    jd: str
    resume_skills: list[str]
    memory_context: str
    tracing: bool
    trace_tokens: bool
    job_info: JobInfo
    retrieved_context: str
    fit_score: int
    matched_skills: list[str]
    missing_skills: list[str]
    visa_warning: str
    human_approved: bool
    human_feedback: str
    interrupted: bool
    match_result: JobMatchResult
    final_answer: str


def extract_job_info_node(state: JobMatchGraphState) -> dict:
    job_info = extract_job_info_langchain(
        state["jd"],
        tracing=state.get("tracing", False),
        trace_tokens=state.get("trace_tokens", False),
    )
    return {"job_info": job_info.model_dump()}


def retrieve_context_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    retrieved_context = retrieve_job_context(
        job_info,
        state["resume_skills"],
    )
    return {"retrieved_context": retrieved_context}


def calculate_score_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    score_result = calculate_fit_score.invoke(
        {
            "required_skills": job_info.required_skills,
            "resume_skills": state["resume_skills"],
        }
    )
    return {
        "fit_score": score_result["fit_score"],
        "matched_skills": score_result["matched_skills"],
        "missing_skills": score_result["missing_skills"],
    }


def route_authorization_risk(state: JobMatchGraphState) -> str:
    job_info = ensure_job_info(state["job_info"])
    if job_info.authorization_risk == "high":
        return "visa_warning"

    return "generate_summary"


def visa_warning_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    warning = (
        "This job has high authorization risk because the job description indicates "
        "no sponsorship, citizenship, green card, or permanent-resident-style constraints. "
        "For a candidate who needs OPT, CPT, H-1B sponsorship, or visa support, "
        "this should be treated as an important application risk."
    )
    human_decision = interrupt(
        {
            "reason": "high_authorization_risk",
            "question": "This job is high authorization risk. Continue generating the final summary?",
            "job_title": job_info.job_title,
            "fit_score": state["fit_score"],
            "visa_warning": warning,
            "expected_resume_value": {
                "approved": True,
                "feedback": "Continue, but make the visa risk clear.",
            },
        }
    )

    if isinstance(human_decision, dict):
        approved = bool(human_decision.get("approved", False))
        feedback = str(human_decision.get("feedback", ""))
    else:
        approved = bool(human_decision)
        feedback = ""

    return {
        "visa_warning": warning,
        "human_approved": approved,
        "human_feedback": feedback,
    }


def route_after_human_review(state: JobMatchGraphState) -> str:
    if state.get("human_approved", False):
        return "generate_summary"

    return "skip_application_summary"


def skip_application_summary_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    match_result = JobMatchResult(
        job_title=job_info.job_title,
        fit_score=state["fit_score"],
        matched_skills=state["matched_skills"],
        missing_skills=state["missing_skills"],
        experience_level=job_info.experience_level,
        authorization_risk=job_info.authorization_risk,
        entry_level_fit=job_info.entry_level_fit,
        final_answer=(
            "该岗位签证/工卡风险较高，且人工审核选择不继续生成完整申请总结。"
            f"当前匹配分数为 {state['fit_score']}，可先优先考虑更签证友好的岗位。"
        ),
    )
    return {
        "match_result": match_result.model_dump(),
        "final_answer": match_result.final_answer,
    }


def generate_summary_node(state: JobMatchGraphState) -> dict:
    job_info = ensure_job_info(state["job_info"])
    chain = create_job_match_summary_chain(streaming=state.get("trace_tokens", False))
    trace_config = build_trace_config(
        state.get("tracing", False),
        run_name="langgraph-generate-summary",
    )
    match_result = chain.invoke(
        {
            "job_info": job_info.model_dump_json(indent=2),
            "resume_skills": json.dumps(state["resume_skills"], ensure_ascii=False),
            "fit_score": state["fit_score"],
            "matched_skills": json.dumps(
                state["matched_skills"],
                ensure_ascii=False,
            ),
            "missing_skills": json.dumps(
                state["missing_skills"],
                ensure_ascii=False,
            ),
            "memory_context": state.get(
                "memory_context",
                "No memory context provided.",
            ),
            "retrieved_context": state.get(
                "retrieved_context",
                "No retrieved context provided.",
            ),
            "visa_warning": state.get("visa_warning", ""),
        },
        config=trace_config,
    )
    if not isinstance(match_result, JobMatchResult):
        match_result = JobMatchResult.model_validate(match_result)

    return {
        "match_result": match_result.model_dump(),
        "final_answer": match_result.final_answer,
    }


def create_job_match_graph(checkpointer=None):
    graph_builder = StateGraph(JobMatchGraphState)
    graph_builder.add_node("extract_job_info", extract_job_info_node)
    graph_builder.add_node("retrieve_context", retrieve_context_node)
    graph_builder.add_node("calculate_score", calculate_score_node)
    graph_builder.add_node("visa_warning", visa_warning_node)
    graph_builder.add_node("skip_application_summary", skip_application_summary_node)
    graph_builder.add_node("generate_summary", generate_summary_node)

    graph_builder.add_edge(START, "extract_job_info")
    graph_builder.add_edge("extract_job_info", "retrieve_context")
    graph_builder.add_edge("retrieve_context", "calculate_score")
    graph_builder.add_conditional_edges(
        "calculate_score",
        route_authorization_risk,
        {
            "visa_warning": "visa_warning",
            "generate_summary": "generate_summary",
        },
    )
    graph_builder.add_conditional_edges(
        "visa_warning",
        route_after_human_review,
        {
            "generate_summary": "generate_summary",
            "skip_application_summary": "skip_application_summary",
        },
    )
    graph_builder.add_edge("skip_application_summary", END)
    graph_builder.add_edge("generate_summary", END)

    return graph_builder.compile(checkpointer=checkpointer)


def build_langgraph_thread_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
        }
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


def analyze_job_with_agent(
    jd: str,
    resume_skills: list[str] | None,
    memory: JobMatchMemory | None = None,
    visa_preference: str | None = None,
    target_role: str | None = None,
    stream: bool = False,
    tracing: bool = False,
    trace_tokens: bool = False,
) -> dict:
    if tracing and not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")

    project_name = get_langsmith_project_name()
    if tracing:
        print(f"LangSmith tracing enabled. Project: {project_name}")

    if memory:
        memory.update_user_profile(
            resume_skills=resume_skills,
            visa_preference=visa_preference,
            target_role=target_role,
        )

    if resume_skills is None:
        if memory and memory.resume_skills:
            resume_skills = memory.resume_skills
        else:
            raise ValueError("resume_skills is required when memory has no saved skills.")

    memory_context = memory.build_context() if memory else "No memory context provided."

    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            job_info = extract_job_info_langchain(
                jd,
                tracing=tracing,
                trace_tokens=trace_tokens,
            )
            retriever_chain = create_job_context_retriever_chain()
            retriever_config = build_trace_config(tracing, run_name="job-context-retriever")
            retrieved_context = retriever_chain.invoke(
                {
                    "job_info": job_info,
                    "resume_skills": resume_skills,
                },
                config=retriever_config,
            )

            if stream:
                result = stream_job_match_agent(
                    job_info,
                    resume_skills,
                    retrieved_context=retrieved_context,
                    memory_context=memory_context,
                    tracing=tracing,
                    trace_tokens=trace_tokens,
                )
            else:
                chain = create_job_match_chain(streaming=trace_tokens)
                trace_config = build_trace_config(tracing, run_name="job-match-chain")
                result = chain.invoke(
                    build_job_match_variables(
                        job_info,
                        resume_skills,
                        retrieved_context,
                        memory_context,
                    ),
                    config=trace_config,
                )

            match_result = parse_job_match_result(result)
            if memory:
                memory.remember_analysis(job_info, match_result)

            return {
                "job_info": job_info,
                "retrieved_context": retrieved_context,
                "memory_context": memory_context,
                "match_result": match_result,
                "final_answer": match_result.final_answer,
                "agent_result": result,
            }
    finally:
        if tracing:
            wait_for_all_tracers()


def analyze_job_with_langgraph(
    jd: str,
    resume_skills: list[str] | None,
    memory: JobMatchMemory | None = None,
    checkpointer=None,
    thread_id: str = "job-match-demo",
    visa_preference: str | None = None,
    target_role: str | None = None,
    stream: bool = False,
    tracing: bool = False,
    trace_tokens: bool = False,
) -> dict:
    if tracing and not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")

    project_name = get_langsmith_project_name()
    if tracing:
        print(f"LangSmith tracing enabled. Project: {project_name}")

    if memory:
        memory.update_user_profile(
            resume_skills=resume_skills,
            visa_preference=visa_preference,
            target_role=target_role,
        )

    if resume_skills is None:
        if memory and memory.resume_skills:
            resume_skills = memory.resume_skills
        else:
            raise ValueError("resume_skills is required when memory has no saved skills.")

    memory_context = memory.build_context() if memory else "No memory context provided."
    graph = create_job_match_graph(checkpointer=checkpointer)
    graph_config = build_langgraph_thread_config(thread_id)
    graph_input: JobMatchGraphState = {
        "jd": jd,
        "resume_skills": resume_skills,
        "memory_context": memory_context,
        "tracing": tracing,
        "trace_tokens": trace_tokens,
    }

    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            if stream:
                final_state: JobMatchGraphState = graph_input.copy()
                print("\nStreaming LangGraph workflow:")
                for update in graph.stream(
                    graph_input,
                    config=graph_config,
                    stream_mode="updates",
                ):
                    print(json.dumps(update, ensure_ascii=False, default=str, indent=2))
                    for node_update in update.values():
                        final_state.update(node_update)
            else:
                final_state = graph.invoke(graph_input, config=graph_config)

            if "__interrupt__" in final_state:
                checkpoint_state = graph.get_state(graph_config) if checkpointer else None
                return {
                    "interrupted": True,
                    "interrupts": final_state["__interrupt__"],
                    "graph_state": final_state,
                    "checkpoint_state": checkpoint_state,
                    "thread_id": thread_id,
                }

            match_result = ensure_job_match_result(final_state["match_result"])
            job_info = ensure_job_info(final_state["job_info"])
            if memory:
                memory.remember_analysis(job_info, match_result)

            checkpoint_state = None
            if checkpointer:
                checkpoint_state = graph.get_state(graph_config)

            return {
                "job_info": job_info,
                "retrieved_context": final_state["retrieved_context"],
                "memory_context": memory_context,
                "visa_warning": final_state.get("visa_warning", ""),
                "match_result": match_result,
                "final_answer": final_state["final_answer"],
                "graph_state": final_state,
                "checkpoint_state": checkpoint_state,
                "interrupted": False,
                "thread_id": thread_id,
            }
    finally:
        if tracing:
            wait_for_all_tracers()


def resume_job_match_after_human_review(
    checkpointer,
    thread_id: str,
    approved: bool,
    feedback: str = "",
    memory: JobMatchMemory | None = None,
    tracing: bool = False,
) -> dict:
    if tracing and not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")

    project_name = get_langsmith_project_name()
    graph = create_job_match_graph(checkpointer=checkpointer)
    graph_config = build_langgraph_thread_config(thread_id)
    resume_value = {
        "approved": approved,
        "feedback": feedback,
    }

    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            final_state = graph.invoke(
                Command(resume=resume_value),
                config=graph_config,
            )

            match_result = ensure_job_match_result(final_state["match_result"])
            job_info = ensure_job_info(final_state["job_info"])
            if memory:
                memory.remember_analysis(job_info, match_result)

            checkpoint_state = graph.get_state(graph_config)

            return {
                "interrupted": False,
                "job_info": job_info,
                "retrieved_context": final_state["retrieved_context"],
                "memory_context": final_state.get(
                    "memory_context",
                    "No memory context provided.",
                ),
                "visa_warning": final_state.get("visa_warning", ""),
                "human_approved": final_state.get("human_approved", False),
                "human_feedback": final_state.get("human_feedback", ""),
                "match_result": match_result,
                "final_answer": final_state["final_answer"],
                "graph_state": final_state,
                "checkpoint_state": checkpoint_state,
                "thread_id": thread_id,
            }
    finally:
        if tracing:
            wait_for_all_tracers()


def create_job_match_workflow_tool(
    memory: JobMatchMemory,
    checkpointer,
    thread_id: str,
    tracing: bool = False,
    trace_tokens: bool = False,
):
    @tool
    def analyze_job_posting(
        job_description: str,
        continue_high_risk_analysis: bool = True,
    ) -> str:
        """Analyze a job posting with the Job Match LangGraph workflow."""
        result = analyze_job_with_langgraph(
            job_description,
            resume_skills=None,
            memory=memory,
            checkpointer=checkpointer,
            thread_id=thread_id,
            stream=False,
            tracing=tracing,
            trace_tokens=trace_tokens,
        )

        if result["interrupted"]:
            if not continue_high_risk_analysis:
                result = resume_job_match_after_human_review(
                    checkpointer=checkpointer,
                    thread_id=result["thread_id"],
                    approved=False,
                    feedback="The chat agent chose not to continue high-risk analysis.",
                    memory=memory,
                    tracing=tracing,
                )
            else:
                result = resume_job_match_after_human_review(
                    checkpointer=checkpointer,
                    thread_id=result["thread_id"],
                    approved=True,
                    feedback="Continue, but make the visa risk clear.",
                    memory=memory,
                    tracing=tracing,
                )

        return json.dumps(
            {
                "job_title": result["job_info"].job_title,
                "fit_score": result["match_result"].fit_score,
                "matched_skills": result["match_result"].matched_skills,
                "missing_skills": result["match_result"].missing_skills,
                "authorization_risk": result["match_result"].authorization_risk,
                "entry_level_fit": result["match_result"].entry_level_fit,
                "visa_warning": result["visa_warning"],
                "final_answer": result["final_answer"],
            },
            ensure_ascii=False,
        )

    return analyze_job_posting


def build_career_chat_system_message() -> SystemMessage:
    return SystemMessage(
        content=(
            "You are a career assistant chat agent. "
            "You are the conversational entry point, not the whole job matching system. "
            "When the user asks you to analyze, evaluate, match, or review a job posting, "
            "call the analyze_job_posting tool. "
            "The tool runs the inner Job Match LangGraph workflow, including RAG, scoring, "
            "authorization-risk review, and final structured summary. "
            "After the tool returns, explain the result to the user in concise Chinese. "
            "If the user is only asking a general career question without a job description, "
            "answer directly and do not call the tool."
        )
    )


def create_career_chat_agent(
    memory: JobMatchMemory,
    checkpointer,
    thread_id: str,
    streaming: bool = False,
    tracing: bool = False,
    trace_tokens: bool = False,
):
    llm = create_llm(streaming=streaming)
    job_match_tool = create_job_match_workflow_tool(
        memory=memory,
        checkpointer=checkpointer,
        thread_id=thread_id,
        tracing=tracing,
        trace_tokens=trace_tokens,
    )
    return create_agent(
        model=llm,
        tools=[job_match_tool],
        system_prompt=build_career_chat_system_message(),
    )


def run_career_chat_agent(
    user_message: str,
    memory: JobMatchMemory,
    checkpointer,
    thread_id: str = "career-chat-demo-user-1",
    tracing: bool = False,
    trace_tokens: bool = False,
) -> dict:
    if tracing and not os.environ.get("LANGSMITH_API_KEY"):
        raise RuntimeError("Please set the LANGSMITH_API_KEY environment variable first.")

    project_name = get_langsmith_project_name()
    agent = create_career_chat_agent(
        memory=memory,
        checkpointer=checkpointer,
        thread_id=thread_id,
        streaming=trace_tokens,
        tracing=tracing,
        trace_tokens=trace_tokens,
    )
    trace_config = build_trace_config(tracing, run_name="career-chat-agent")

    try:
        with ls.tracing_context(project_name=project_name, enabled=tracing):
            return agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=user_message),
                    ]
                },
                config=trace_config,
            )
    finally:
        if tracing:
            wait_for_all_tracers()


if __name__ == "__main__":
    RUN_CHAT_AGENT_DEMO = True
    STREAM_OUTPUT = True
    TRACE_OUTPUT = False
    TRACE_TOKENS = False

    resume_skills = [
        "Python",
        "SQL",
        "Tableau",
        "Power BI",
        "A/B testing",
        "Excel",
        "Pandas",
    ]

    jd = """
    We are looking for a Data Analyst with experience in Python, SQL, Tableau,
    and A/B testing. 0-2 years of experience preferred. No sponsorship is available.
    """

    memory = JobMatchMemory()
    checkpointer = MemorySaver()
    memory.update_user_profile(
        resume_skills=resume_skills,
        visa_preference="I need OPT, CPT, H-1B sponsorship, or visa-friendly roles.",
        target_role="Entry-level data analyst",
    )

    if RUN_CHAT_AGENT_DEMO:
        chat_result = run_career_chat_agent(
            user_message=(
                "请帮我分析这个岗位是否适合我，并特别注意签证风险：\n"
                f"{jd}"
            ),
            memory=memory,
            checkpointer=checkpointer,
            thread_id="job-match-demo-user-1",
            tracing=TRACE_OUTPUT,
            trace_tokens=TRACE_TOKENS,
        )

        print("Career Chat Agent Message Trace:")
        print_message_trace(chat_result)
        print("\nMemory Context After Chat Run:")
        print(memory.build_context())
    else:
        result = analyze_job_with_langgraph(
            jd,
            resume_skills=None,
            memory=memory,
            checkpointer=checkpointer,
            thread_id="job-match-demo-user-1",
            stream=STREAM_OUTPUT,
            tracing=TRACE_OUTPUT,
            trace_tokens=TRACE_TOKENS,
        )

        if result["interrupted"]:
            print("\nLangGraph interrupted for human review:")
            for item in result["interrupts"]:
                print(item.value)

            result = resume_job_match_after_human_review(
                checkpointer=checkpointer,
                thread_id=result["thread_id"],
                approved=True,
                feedback="Continue, but make the visa risk clear.",
                memory=memory,
                tracing=TRACE_OUTPUT,
            )

        print("Structured Job Info:")
        print(result["job_info"].model_dump_json(indent=2))
        print("\nMemory Context Used:")
        print(result["memory_context"])
        print("\nRetrieved Context:")
        print(result["retrieved_context"])
        print("\nVisa Warning:")
        print(result["visa_warning"] or "No visa warning.")
        print("\nStructured Match Result:")
        print(result["match_result"].model_dump_json(indent=2))
        print("\nFinal Answer:")
        print(result["final_answer"])
        print("\nGraph State Keys:")
        print(sorted(result["graph_state"].keys()))
        print("\nCheckpoint State Keys:")
        print(sorted(result["checkpoint_state"].values.keys()))
        print("\nMemory Context After This Run:")
        print(memory.build_context())
