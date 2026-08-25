import hashlib
import math
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from .models import JobInfo


KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "career_knowledge_base"


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
        return sum(
            left_value * right_value for left_value, right_value in zip(left, right)
        )


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


def split_document(
    document: Document,
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[Document]:
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


def retrieve_job_context(
    job_info: JobInfo,
    resume_skills: list[str],
    top_k: int = 2,
) -> str:
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

