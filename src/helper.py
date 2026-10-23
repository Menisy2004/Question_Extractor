import logging
import os
import re
import time

from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

from src.prompt import prompt_template

load_dotenv(find_dotenv())

# Accept either common Google AI Studio key name in local and hosted environments.
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

logger = logging.getLogger(__name__)
CHAT_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
)


def _is_temporary_model_error(error):
    message = str(error).upper()
    return any(
        marker in message
        for marker in (
            "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED",
            "TIMEOUT", "TIMED OUT", "DEADLINE EXCEEDED",
        )
    )


def _invoke_with_model_fallback(prompt, values):
    for index, model_name in enumerate(CHAT_MODELS):
        llm = ChatGoogleGenerativeAI(
            model=model_name, temperature=0.3, timeout=45, max_retries=0
        )
        try:
            return (prompt | llm | StrOutputParser()).invoke(values)
        except Exception as exc:
            if not _is_temporary_model_error(exc) or index == len(CHAT_MODELS) - 1:
                raise
            logger.warning("Model %s is at capacity; trying %s", model_name, CHAT_MODELS[index + 1])


class GroundedAnswerer:
    """Answer from relevant PDF chunks using Gemini embeddings and Gemini Flash."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self._chunk_embeddings = None
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "Answer the question using only the supplied study material. "
                "If the material does not contain the answer, say so clearly.\n\n"
                "Study material:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
        )

    def run(self, question):
        if self._chunk_embeddings is None:
            self._chunk_embeddings = self.embeddings.embed_documents(self.chunks)
        question_embedding = self.embeddings.embed_query(question)

        def similarity(vector):
            dot = sum(a * b for a, b in zip(vector, question_embedding))
            left_norm = sum(value * value for value in vector) ** 0.5
            right_norm = sum(value * value for value in question_embedding) ** 0.5
            if not left_norm or not right_norm:
                return 0.0
            return dot / (left_norm * right_norm)

        ranked_chunks = sorted(
            enumerate(self.chunks),
            key=lambda item: (similarity(self._chunk_embeddings[item[0]]), -item[0]),
            reverse=True,
        )
        selected = [chunk for _, chunk in ranked_chunks[:4]]
        context = "\n\n".join(selected)
        return _invoke_with_model_fallback(
            self.prompt, {"context": context, "question": question}
        )


def file_processing(file_path):
    pages = PyPDFLoader(file_path).load()
    text = "\n\n".join(page.page_content for page in pages if page.page_content.strip())
    if not text.strip():
        raise ValueError("No selectable text was found in the PDF.")

    question_splitter = TokenTextSplitter(
        model_name="gpt-3.5-turbo", chunk_size=10000, chunk_overlap=200
    )
    question_documents = [
        Document(page_content=chunk) for chunk in question_splitter.split_text(text)
    ]
    if len(question_documents) > 4:
        # Bound prompt size while sampling content across the whole PDF.
        sample_indices = [round(i * (len(question_documents) - 1) / 3) for i in range(4)]
        question_documents = [question_documents[index] for index in sample_indices]

    answer_splitter = TokenTextSplitter(
        model_name="gpt-3.5-turbo", chunk_size=1000, chunk_overlap=100
    )
    answer_chunks = answer_splitter.split_text(text)
    return question_documents, answer_chunks


def llm_pipeline(file_path):
    question_documents, answer_chunks = file_processing(file_path)
    logger.info(
        "Starting question generation for %d document chunks",
        len(question_documents),
    )
    started_at = time.monotonic()

    question_prompt = PromptTemplate(
        template=prompt_template, input_variables=["text"]
    )
    generation_context = "\n\n".join(
        document.page_content for document in question_documents
    )
    generated_text = _invoke_with_model_fallback(
        question_prompt, {"text": generation_context}
    )
    logger.info("Question model completed in %.1f seconds", time.monotonic() - started_at)

    questions = [
        line.strip()
        for line in generated_text.splitlines()
        if re.match(r"^\s*\d+[.)]\s+.+\?\s*$", line)
    ]
    if not questions:
        questions = [
            line.strip()
            for line in generated_text.splitlines()
            if line.strip().endswith("?")
        ]

    # Keep PDF chunks in the session and defer answer generation until reveal.
    answerer = GroundedAnswerer(answer_chunks)
    return answerer, questions
