prompt_template = """
You are an expert at creating practice questions based on study material and documentation.
Your goal is to prepare a student or researcher for their upcoming exams and tests.
You do this by asking questions about the text below:

--------
{text}
--------

Create 10 clear, direct questions about the content of the text that will prepare the student for their test.
Make sure not to lose any important information.
IMPORTANT FORMATTING RULE: Output ONLY a numbered list of questions (1. , 2. , etc.), with one question per line. Do NOT include any introductory text, section headers, categories, or explanations.

QUESTIONS:
"""


refine_template = """
You are an expert at creating practice questions based on study material and documentation.
Your goal is to help a student or researcher prepare for an upcoming test.
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones (only if necessary) with some more context below.
-------------
{text}
-------------

Given the new context, refine the original questions in English. If the context is not helpful, please provide the original questions.
IMPORTANT FORMATTING RULE: Output ONLY a numbered list of questions (1. , 2. , etc.), with one question per line. Do NOT include any introductory text, section headers, categories, or explanations.

QUESTIONS:
"""