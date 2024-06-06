"""
File name: prepare_chain_4_chat.py
Author: Luigi Saetta
Date created: 2023-12-04
Date last modified: 2024-05-11
Python Version: 3.11

Description:
    This module provides all factyory methods (create_llms..) 
    and a function to initialize the RAG chain 
    for chatbot using message history

    This is part of the porting to Llamaindex 10+ 

Usage:
    Import this module into other scripts to use its functions. 
    Example:
        from prepare_chain_4_chat import create_chat_engine

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle AI Vector Search,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all the data (text + embeddings) are stored in Oracle DB 23c 

    Now it can use for LLM: OCI (Cohere & LLama2), Mistral 8x7B, Cohere Command-R

Warnings:
    This module is in development, may change in future versions.
"""

import os
import logging

from tokenizers import Tokenizer

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

# integrations
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.cohere import Cohere
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.embeddings.oci_genai import OCIGenAIEmbeddings
from llama_index.llms.oci_genai import OCIGenAI

from llama_index.core.memory import ChatMemoryBuffer

# Phoenix traces
from llama_index.core.callbacks.global_handlers import set_global_handler

# for reranker if in OCI DS
import ads

# COHERE_KEY is used for reranker, LLM
# MISTRAL_KEY for LLM
from config_private import (
    COMPARTMENT_OCID,
    ENDPOINT,
    ENDPOINT_EMBED,
    MISTRAL_API_KEY,
    COHERE_API_KEY,
)

#
# all the configuration is controlled by parameters
# in config.py
#
from config import (
    VERBOSE,
    EMBED_MODEL_TYPE,
    EMBED_MODEL,
    TOKENIZER,
    GEN_MODEL,
    OCI_GEN_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_K,
    TOP_N,
    ADD_RERANKER,
    RERANKER_MODEL,
    RERANKER_ID,
    CHAT_MODE,
    MEMORY_TOKEN_LIMIT,
    ADD_PHX_TRACING,
    PHX_PORT,
    PHX_HOST,
    # to enable approximate query with HNSW indexes
    LA2_ENABLE_INDEX,
    STREAM_CHAT,
)

from oci_utils import load_oci_config, print_configuration, check_value_in_list
from oracle_vector_db import OracleVectorStore

from oci_baai_reranker import OCIBAAIReranker
from oci_llama_reranker import OCILLamaReranker

# logging
logger = logging.getLogger("ConsoleLogger")

# added Arize Phoenix tracing
if ADD_PHX_TRACING:
    import phoenix as px

#
# This module now expose directly the factory methods for all the single components (llm, etc)
# the philosophy of the factory methods is that they're taking all the infos from the config
# module... so as few parameters as possible
#


#
# enables to plug different GEN_MODELS
# for now: OCI, LLAMA2 70 B, MISTRAL, COMMAND-R
#
def create_cohere_llm():
    """
    create the client for Cohere command-r
    """
    llm = Cohere(
        model="command-r",
        api_key=COHERE_API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return llm


def create_mistral_llm():
    """
    create the client for Mistral large
    """
    llm = MistralAI(
        api_key=MISTRAL_API_KEY,
        model="mistral-large-latest",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    return llm


# to call a model deployed on VM as VLLM
def create_openai_compatible():
    """ "
    client for a vLLM model
    """
    # "mistralai/Mistral-7B-Instruct-v0.2"
    llm = OpenAILike(
        model="CohereForAI/c4ai-command-r-v01",
        api_base="http://141.147.55.249:8888/v1/",
        api_key="token-abc123",
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    return llm


def create_llm():
    """ "
    todo
    """
    # this check is to avoid mistakes in config.py
    # here LLAMA is LLAMA2 on OCI
    model_list = ["OCI", "MISTRAL", "COHERE", "VLLM"]

    check_value_in_list(GEN_MODEL, model_list)

    llm = None

    if GEN_MODEL == "OCI":

        common_oci_params = {
            "compartment_id": COMPARTMENT_OCID,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "service_endpoint": ENDPOINT,
        }

        # these are the name of the models used by OCI GenAI
        # changed 04/06 when new models arrived
        llm = OCIGenAI(auth_type="API_KEY", model=OCI_GEN_MODEL, **common_oci_params)

    if GEN_MODEL == "MISTRAL":
        llm = create_mistral_llm()

    # 16/03 added Cohere command r
    if GEN_MODEL == "COHERE":
        llm = create_cohere_llm()

    if GEN_MODEL == "VLLM":
        llm = create_openai_compatible()

    assert llm is not None

    return llm


def create_reranker(auth=None, verbose=VERBOSE):
    """
    todo
    """
    model_list = ["COHERE", "OCI_BAAI"]

    check_value_in_list(RERANKER_MODEL, model_list)

    reranker = None

    if RERANKER_MODEL == "COHERE":
        reranker = CohereRerank(api_key=COHERE_API_KEY, top_n=TOP_N)

    # reranker model deployed as MD in OCI DS
    if RERANKER_MODEL == "OCI_BAAI":
        baai_reranker = OCIBAAIReranker(
            auth=auth, deployment_id=RERANKER_ID, region="eu-frankfurt-1"
        )

        reranker = OCILLamaReranker(
            oci_reranker=baai_reranker, top_n=TOP_N, verbose=verbose
        )

    return reranker


def create_embedding_model():
    """
    todo
    """
    model_list = ["OCI"]

    check_value_in_list(EMBED_MODEL_TYPE, model_list)

    embed_model = None

    if EMBED_MODEL_TYPE == "OCI":
        embed_model = OCIGenAIEmbeddings(
            auth_type="API_KEY",
            model=EMBED_MODEL,
            service_endpoint=ENDPOINT_EMBED,
            compartment_id=COMPARTMENT_OCID,
            truncate="END",
        )

    return embed_model


#
# the entire chain is built here
#
def create_chat_engine(token_counter=None, verbose=False):
    """
    Create the entiire RAG chain
    """

    logger.info("Calling create_chat_engine()...")

    print_configuration()

    if ADD_PHX_TRACING:
        os.environ["PHOENIX_PORT"] = PHX_PORT
        os.environ["PHOENIX_HOST"] = PHX_HOST
        px.launch_app()

        set_global_handler("arize_phoenix")

    # load security info needed for OCI
    oci_config = load_oci_config()
    api_keys_config = ads.auth.api_keys(oci_config)

    # this is to embed the question
    embed_model = create_embedding_model()

    # this is the custom class to access Oracle DB as Vectore Store
    vector_store = OracleVectorStore(
        verbose=verbose,
        # if LA2_ENABLE_INDEX is true, add the approximate clause to the query
        # needs AI Vector Search GA
        enable_hnsw_indexes=LA2_ENABLE_INDEX,
    )

    # this is to access OCI or MISTRAL or Cohere GenAI service
    llm = create_llm()

    # this part has been added to count the total # of tokens
    cohere_tokenizer = Tokenizer.from_pretrained(TOKENIZER)
    token_counter = TokenCountingHandler(tokenizer=cohere_tokenizer.encode)

    # 16/03/2024: removed service_context -> Settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.callback_manager = CallbackManager([token_counter])

    # here we plug AI Vector Search
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # to handle the conversation history
    memory = ChatMemoryBuffer.from_defaults(
        token_limit=MEMORY_TOKEN_LIMIT, tokenizer_fn=cohere_tokenizer.encode
    )

    # the whole chain (query string -> embed query -> retrieval ->
    # reranker -> context, query-> GenAI -> response)
    # is wrapped in the chat engine

    # here we could plug a reranker improving the quality

    if ADD_RERANKER is True:
        reranker = create_reranker(auth=api_keys_config)

        node_postprocessors = [reranker]
    else:
        node_postprocessors = None

    chat_engine = index.as_chat_engine(
        chat_mode=CHAT_MODE,
        memory=memory,
        verbose=verbose,
        similarity_top_k=TOP_K,
        node_postprocessors=node_postprocessors,
        # to enable streaming the output
        streaming=STREAM_CHAT,
    )

    # to add a blank line in the log
    logger.info("")

    return chat_engine, token_counter
