"""
File name: oracle_chat_with_memory.py
Author: Luigi Saetta
Date created: 2023-12-04
Date last modified: 2024-03-23
Python Version: 3.11

Description:
    This module provides the chatbot UI for the RAG demo 

Usage:
    streamlit run oracle_chat_with_memory.py

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all the data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

import logging
import time
import traceback
import streamlit as st

# to use the create_query_engine
import prepare_chain_4_chat


#
# Configs
#
from config import ADD_REFERENCES, STREAM_CHAT, VERBOSE, SHOW_COUNT_TOKENS


# when push the button
def reset_conversation():
    """
    reset the chat history
    """
    st.session_state.messages = []

    # stored in the session to enable reset
    st.session_state.chat_engine, st.session_state.token_counter = create_chat_engine(
        verbose=False
    )
    # clear message chat history
    st.session_state.chat_engine.reset()

    # reset # questions counter
    st.session_state.question_count = 0


# defined here to avoid import of streamlit in other module
# Don't use cache anymore... one for session
@st.cache_resource
def create_chat_engine(verbose=False):
    """ "
    function to create the chat engine
    local function here to enable caching
    """
    chat_engine, token_counter = prepare_chain_4_chat.create_chat_engine(
        verbose=verbose
    )

    # token_counter keeps track of the num. of tokens
    return chat_engine, token_counter


# case no streaming: to format output with references
def no_stream_output(response):
    """
    prepare the output in caso of no_stream
    """
    output = response.response

    if ADD_REFERENCES and len(response.source_nodes) > 0:
        output += "\n\n Ref.:\n\n"

        for node in response.source_nodes:
            output += str(node.metadata).replace("{", "").replace("}", "") + "  \n"

    st.markdown(output)

    return output


# case streaming
def stream_output(response):
    """
    prepare the output in case of streaming
    """
    # stream the words as soon they arrive
    text_placeholder = st.empty()
    output = ""

    for text in response.response_gen:
        output += text
        text_placeholder.markdown(output, unsafe_allow_html=True)

    if ADD_REFERENCES:
        output += "\n\n Ref.:\n\n"

        for node in response.source_nodes:
            output += str(node.metadata).replace("{", "").replace("}", "") + "  \n"

        text_placeholder.markdown(output, unsafe_allow_html=True)

    return output


#
# Main
#

# Configure logging
# I have changed the way I config logger to solve some problems with
# PY 3.11

logger = logging.getLogger("ConsoleLogger")

# to avoid duplication of logging
if not logger.handlers:
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.propagate = False

# st.title("Knowledge Assistant with Oracle AI Vector Search")
st.title("Assistente IA per ...")

# Added reset button
st.sidebar.button("Clear Chat History", on_click=reset_conversation)

# Initialize session, chat history
if "messages" not in st.session_state:
    reset_conversation()

# init RAG
# TODO: review init logic
with st.spinner("Initializing RAG chain..."):
    # I have added the token counter to count token
    # I've done this way because I marked the function with @cache
    # but there was a problem with the counter. It works if it is created in the other module
    # and returned here where I print the results for each query

    # here we create the query engine
    st.session_state.chat_engine, st.session_state.token_counter = create_chat_engine(
        verbose=VERBOSE
    )


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("Ciao, come posso aiutarti?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(question)

    try:

        with st.spinner("Waiting..."):
            time_start = time.time()

            st.session_state.question_count += 1
            logger.info("")
            logger.info("Question n. %s", st.session_state.question_count)

            # added streaming if available
            if STREAM_CHAT:
                response = st.session_state.chat_engine.stream_chat(question)
            else:
                response = st.session_state.chat_engine.chat(question)

            time_elapsed = time.time() - time_start

        # count the number of questions done
        logger.info("Elapsed time: %s sec.", round(time_elapsed, 1))

        # display num. of input/output token
        # count are incrementals
        if SHOW_COUNT_TOKENS:
            prompt_token_count = st.session_state.token_counter.prompt_llm_token_count
            completion_token_count = (
                st.session_state.token_counter.completion_llm_token_count
            )

            logger.info("LLM Prompt Tokens: %s", prompt_token_count)
            logger.info("LLM Completion Tokens: %s", completion_token_count)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            if STREAM_CHAT:
                # it handles references inside
                output = stream_output(response)

            else:
                # no streaming
                output = no_stream_output(response)

        # Add used and assistant responses to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": output})

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        stack_trace = traceback.format_exc()
        logger.error(stack_trace)
        st.error("An error occurred: " + str(e))
