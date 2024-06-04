"""Python file to serve as the frontend"""
import os
import pickle

import streamlit as st
from streamlit_chat import message
from query_data import _template, QA_PROMPT, CONDENSE_QUESTION_PROMPT, get_chain

os.environ["OPENAI_API_KEY"] = "API_KEY"


st.set_page_config(page_title="Custom Document Assistant", page_icon=":robot:")
st.header("Custom Document Assistant")

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

chain = get_chain(vectorstore)

placeholder = st.empty()

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", value="", key="input")
    return input_text


user_input = get_text()


if user_input:
    docs = vectorstore.similarity_search(user_input)
    output = chain.run(input=user_input, vectorstore=vectorstore, context=docs[:2],
                       chat_history=[], question=user_input, QA_PROMPT=QA_PROMPT,
                       CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
