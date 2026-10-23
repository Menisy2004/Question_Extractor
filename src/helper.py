import logging
import os
import re
import time

from dotenv import find_dotenv, load_dotenv
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

from src.prompt import prompt_template, refine_template

load_dotenv(find_dotenv())

# Accept either common Google AI Studio key name in local and hosted environments.
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

logger = logging.getLogger(__name__)


class GroundedAnswerer:
    """Answer from relevant PDF chunks without making embeddings during upload."""

    STOP_WORDS = {
        "about", "after", "again", "also", "among", "because", "before", "being",
        "between", "could", "does", "from", "have", "into", "more", "most", "other",
        "should", "that", "their", "there", "these", "they", "this", "those", "through",
        "under", "using", "what", "when", "where", "which", "while", "with", "would",
        "your", "explain", "describe", "according", "document", "material",
    }

    def __init__(self, chunks):
        self.chunks = chunks
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", temperature=0.3, timeout=45, max_retries=1
        )
        self.chain = (
            PromptTemplate(
                input_variables=["context", "question"],
                template=(
                    "Answer the question using only the supplied study material. "
                    "If the material does not contain the answer, say so clearly.\n\n"
                    "Study material:\n{context}\n\nQuestion: {question}\nAnswer:"
                ),
            )
            | self.llm
            | StrOutputParser()
        )

    @classmethod
    def _terms(cls, text):
        return {
            word for word in re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
            if word not in cls.STOP_WORDS
        }

    def run(self, question):
        question_terms = self._terms(question)
        ranked_chunks = sorted(
            enumerate(self.chunks),
            key=lambda item: (
                len(self._terms(item[1]) & question_terms),
                -item[0],
            ),
            reverse=True,
        )
        selected = [chunk for _, chunk in ranked_chunks[:4]]
        context = "\n\n".join(selected)
        return self.chain.invoke({"context": context, "question": question})


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
        # Keep serial refine calls bounded on serverless workers while sampling
        # content across the whole PDF instead of taking only its first pages.
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

    question_llm = ChatGroq(
        model="openai/gpt-oss-120b", temperature=0.3, timeout=45, max_retries=0
    )
    question_prompt = PromptTemplate(
        template=prompt_template, input_variables=["text"]
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"], template=refine_template
    )
    question_chain = load_summarize_chain(
        llm=question_llm,
        chain_type="refine",
        verbose=False,
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
    )
    generated_text = question_chain.invoke({"input_documents": question_documents})[
        "output_text"
    ]
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

    # Build the answer model and keep PDF chunks in the session, but defer every
    # Google API request until a learner actually reveals an answer.
    answerer = GroundedAnswerer(answer_chunks)
    return answerer, questions
