import os 
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_classic.chains.summarize import load_summarize_chain
from google import genai
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.prompt import *

load_dotenv(find_dotenv())

def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen+= page.page_content

    splitter_ques_gen = TokenTextSplitter(
    model_name="gpt-3.5-turbo",
    chunk_size=10000,
    chunk_overlap= 200
    )

    # split raw string to token chunks
    chunk_ques_gen = splitter_ques_gen.split_text(question_gen)
    document_ques_gen = [Document(page_content= t) for t in chunk_ques_gen]

    splitter_ans_gen = TokenTextSplitter(
    model_name = 'gpt-3.5-turbo',
    chunk_size = 1000,
    chunk_overlap = 100
    )

    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_answer_gen, document_ques_gen

def llm_pipeline(file_path):
    # file_processing returns small answer chunks first and larger question
    # chunks second. Keep the larger chunks for question generation to avoid
    # making an unnecessary Gemini refine call for every small answer chunk.
    document_answer_gen, document_ques_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.3)

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=['text'])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template = refine_template
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline,
                                      chain_type="refine",
                                      verbose = True,
                                      question_prompt=PROMPT_QUESTIONS,
                                      refine_prompt= REFINE_PROMPT_QUESTIONS)
    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)
    llm_answer_gen = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.3)

    ques_list = [
    line.strip()
    for line in ques.split("\n")
    if line.strip().endswith("?") and line.strip()[0].isdigit()
    ]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                      chain_type="stuff",
                                                      retriever=vector_store.as_retriever())

    return answer_generation_chain, ques_list
