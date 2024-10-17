import os
import json 
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging

# importing necessary packages from the langchain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
#from langchain.schema import RunnableSequence
from langchain.chains import SequentialChain
from langchain.chains import LLMChain

# load the variables from the .env file
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=key, model="gpt-3.5-turbo", temperature=0.4)

template = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template
)

quiz_chain = LLMChain(llm = llm,prompt = quiz_generation_prompt,output_key = "quiz")

template2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students,\
you need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
If the quiz is not appropriate for the students' abilities, update the questions accordingly.
Quiz_MCQs:
{quiz}
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["quiz", "subject"],
    template=template2
)

review_chain = LLMChain(llm = llm,prompt = quiz_evaluation_prompt,output_key = "review")


generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"]
)
