import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
import streamlit as st
#from langchain.callbacks import get_openai_callback
#from langchain_community.callbacks.manager import get_openai_callback
from src.mcqgenerator.MCQGENERATOR import generate_evaluate_chain
from src.mcqgenerator.logger import logging
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#print("my name is Abhay")
# loading response_json
with open(r'C:\Users\rajpu\mcqgenn\Response.json','r') as file:
    RESPONSE_JSON = json.load(file)
# now let us create the web application using streamlit
st.title("MCQ Creator Application with Langchain 🔗🔗🔗🐦🔗 🔗🐦")
# create a form using st.form
with st.form("user_inputs"):
    # file upload
    uploaded_file = st.file_uploader("Upload a pdf or text file")
    # input fields
    mcq_count = st.number_input("Number of mcqs",min_value = 3,max_value = 50)
    # subject
    subject = st.text_input("Insert Subject",max_chars = 20)
    # quiz_tone
    tone = st.text_input("Complexity level of question",max_chars = 20,placeholder="simple")
    # add button
    button = st.form_submit_button("Create MCQs")
    # check if the button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                response = generate_evaluate_chain(
                    {
                        "text":text,
                        "number":mcq_count,
                        "subject":subject,
                        "tone":tone,
                        "response_json":json.dumps(RESPONSE_JSON)
                    }
                )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
            else:
                if isinstance(response,dict):
                    quiz = response.get("quiz",None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            # display the review in a text box as well
                            st.text_area(label = "Review",value = response["review"])
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)
                    






