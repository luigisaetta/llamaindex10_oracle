"""
File name: config.py
Author: Luigi Saetta
Date created: 2023-12-15
Date last modified: 2024-03-16
Python Version: 3.11

Description:
    This module provides some configurations


Usage:
    Import this module into other scripts to use its functions. 
    Example:
    ...

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

# the book we're going to split and embed
INPUT_FILES = [
    "AI Generativa - casi d'uso per industry.pdf",
    "CurrentEssentialsofMedicine.pdf",
    "Il conto corrente in parole semplici.pdf",
    "La storia del Gruppo-iccrea.pdf",
    "La_Centrale_dei_Rischi_in_parole_semplici.pdf",
    "Luigi Saetta-CV-2024.pdf",
    "Oracle Generative AI-byAI.pdf",
    "ai-4-italy.pdf",
    "coop-fi-aprile2024.pdf",
    "coop-fi-marzo2024.pdf",
    "covid19_treatment_guidelines.pdf",
    "database-concepts.pdf",
    "database-security-assessment-tool-user-guide_3.1.pdf",
    "feynman_vol1.pdf",
    "gpt-4.pdf",
    "high-availability-23c.pdf",
    "llama2.pdf",
    "oracle-database-23c-new-features-guide.pdf",
    "python4everybody.pdf",
    "rag_review.pdf",
    "the-side-effects-of-metformin-a-review.pdf",
]

VERBOSE = False

STREAM_CHAT = True

# the ony one for now
EMBED_MODEL_TYPE = "OCI"
# Cohere embeddings model in OCI
# for multilingual (es: italian) use this one
EMBED_MODEL = "cohere.embed-multilingual-v3.0"
# for english use this one
# EMBED_MODEL = "cohere.embed-english-v3.0"

# used for token counting
TOKENIZER = "Cohere/Cohere-embed-multilingual-v3.0"

# to enable splitting pages in chunks
# in token
# modified 05/02/2024
ENABLE_CHUNKING = True
# set to 1000
MAX_CHUNK_SIZE = 1600
CHUNK_OVERLAP = 100

# choose the Gen Model
GEN_MODEL = "OCI"

# GEN_MODEL = "MISTRAL"

# must be LLAMA and not LLAMA2
# GEN_MODEL = "LLAMA"

# for command-r in Cohere API
# GEN_MODEL = "COHERE"

# for Cohere in vLLM
# GEN_MODEL = "VLLM"


# for retrieval
TOP_K = 8
# reranker
TOP_N = 4

# for GenAI models
MAX_TOKENS = 1024
# added 29/02
TEMPERATURE = 0.1

# if we want to add a reranker (Cohere or BAAI for now)
ADD_RERANKER = True
RERANKER_MODEL = "COHERE"
# RERANKER_MODEL = "OCI_BAAI"
RERANKER_ID = "ocid1.datasciencemodeldeployment.oc1.eu-frankfurt-1.amaaaaaangencdyaulxbosgii6yajt2jdsrrvfbequkxt3mepz675uk3ui3q"

# for chat engine
CHAT_MODE = "condense_plus_context"
# cambiato per Cohere command-R
MEMORY_TOKEN_LIMIT = 10000

# bits used to store embeddings
# possible values: 32 or 64
# must be aligned with the create_tables.sql used
EMBEDDINGS_BITS = 64

# ID generation: LLINDEX, HASH, BOOK_PAGE_NUM
# define the method to generate ID
ID_GEN_METHOD = "HASH"

# Tracing
ADD_PHX_TRACING = True
PHX_PORT = "7777"
PHX_HOST = "0.0.0.0"

# To enable approximate query
# disable if you're using AI vector Search LA1 or you
# have not create indexes on vector table
LA2_ENABLE_INDEX = False

# UI
ADD_REFERENCES = True
