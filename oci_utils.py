"""
File name: oci_utils.py
Author: Luigi Saetta
Date created: 2023-12-17
Date last modified: 2024-03-16
Python Version: 3.11

Description:
    This module provides some utilities

Usage:
    Import this module into other scripts to use its functions. 
    Example:
    ...

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to buil a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

import logging
import oci

from config import (
    EMBED_MODEL_TYPE,
    EMBED_MODEL,
    GEN_MODEL,
    TOP_K,
    ADD_RERANKER,
    RERANKER_MODEL,
    TOP_N,
    ADD_PHX_TRACING,
)

logger = logging.getLogger("ConsoleLogger")


def load_oci_config():
    # read OCI config to connect to OCI with API key

    # are you using default profile?
    oci_config = oci.config.from_file("~/.oci/config", "DEFAULT")

    return oci_config


def print_configuration():
    logger.info("------------------------")
    logger.info("Configuration used:")
    logger.info(f"{EMBED_MODEL_TYPE} {EMBED_MODEL} for embeddings...")
    logger.info("Using Oracle AI Vector Search...")
    logger.info(f"Using {GEN_MODEL} as LLM...")
    logger.info("Retrieval parameters:")
    logger.info(f"TOP_K: {TOP_K}")

    if ADD_RERANKER:
        logger.info(f"Using {RERANKER_MODEL} as reranker...")
        logger.info(f"TOP_N: {TOP_N}")
    if ADD_PHX_TRACING:
        logger.info(f"Enabled observability with Phoenix tracing...")

    logger.info("------------------------")
    logger.info("")


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
