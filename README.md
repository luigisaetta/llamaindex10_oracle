# Integrate Oracle AI Vector DB and OCI GenAI with Llama-index (v. 0.10+)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## The UI of the **Knowledge Assistant** you can build using following examples.

![screenshot](./screenshot.png)

This repository contains all the work done on the development of RAG applications using:

* [Oracle AI Vector Search](https://www.oracle.com/news/announcement/ocw-integrated-vector-database-augments-generative-ai-2023-09-19/)
* Oracle OCI [GenAI Service](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/generative-ai/home.htm)
* Oracle OCI[ Embeddings](https://docs.public.oneportal.content.oci.oraclecloud.com/en-us/iaas/Content/generative-ai/embed-models.htm)
* Cohere Reranking
* [Reranker](https://github.com/luigisaetta/llamaindex_oracle/blob/main/deploy_reranker.ipynb) models deployed in OCI Data Science
* OCI [ADS 2.10.0](https://accelerated-data-science.readthedocs.io/en/latest/user_guide/large_language_model/langchain_models.html) (with support for OCI GenAI)
* [llama-index](https://docs.llamaindex.ai/en/stable/)

In the [Video demos](https://github.com/luigisaetta/llamaindex_oracle/wiki/Video-demos) section of the Wiki you'll find some video of the demo.

## What is RAG?

A very good introduction to what **Retrieval Augmented Generation** (RAG) is can be found [here](https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/)

## Features

* basic (12/2023) integration between **Oracle DB Vector Store (23c)** and **llama-index**
* All documents stored in an **Oracle AI Vector Search**
* Oracle AI Vector Search used for Semantic Search
* Reranking to improve retrieval
* How to show references (documents used for the response generation)
* (30/12/2023) Added reranker implemented as OCI Model Deployment
* (20/01/2024) Added implementation of Vector Store for LangChain and demo
* Finding duplicates in the documentation
* (2/03/2024) Added Phoenix Traces for observability
* (25/3/2024) This is the code for **LlamaIndex 0.10+**

## Demos

* [demo1](./custom_vector_store_demo1.ipynb) This NB shows how you get answers to questions on Oracle Database and new features in 23c, using Oracler AI Vector Search
* [Knowledge assistant full demo](./run_oracle_chat_with_memory.sh)

## Setup

See the [wiki](https://github.com/luigisaetta/llamaindex_oracle/wiki/Setup-of-the-Python-conda-environment) pages.

## Loading data

* You can use the [create_save_embeddings](./create_save_embeddings.py) Python program to load all the data in the Oracle DB schema.
* You can launch it using the script [load_books](./load_books.sh).
* The list of files to be loaded is specified in the file config.py

You need to have pdf files in the same directory.

## Limited Availability

* Oracle **AI Vector Search** (Vector DB) is a new feature in Oracle DB 23c, in **Limited Availability**. 

Customers can easily enter in the LA/Beta program.

To test these functionalities you need to enroll in the LA program and install the proper versions of software libraries.

Code and functionalities can change, as a result of feedbacks from customers.

## Releases used for the demo

* OCI 2.124.1
* OCI ADS 2.11.3
* LangChain >= 0.1.12
* LangChain Community >= 0.0.28
* Llama-index >= 0.10
* Oracle Database 23c (23.4) Enterprise Edition with **AI Vector Search**

You can install a complete Python environment using the instructions in the **Setup* section of the Wiki.

## Libraries

* OCI Python SDK
* OCI ADS
* oracledb
* Streamlit
* Llama-index
* LangChain
* Arize-AI/phoenix for Observability and Tracing

## Documentation

* [OCI GenAI in LangChain](https://python.langchain.com/docs/integrations/llms/oci_generative_ai)

## Embeddings

One of the key pieces in a **RAG** solution s the Retrieval module. 
To use the **AI DB Vector Store** you need an **Embeddings Model**: a model that does the magic of transforming text in vectors, capturing the content and the semantic of the text.
The Embeddings Model used in these demos is [Cohere Embeds V3](https://txt.cohere.com/introducing-embed-v3/), provided from **OCI GenAI** service.

With few changes, you can switch to use any Open Source model. But you need to have the computational power (GPU) to run it.

## Observability

(02/03/2024) Added integration with **Arize Phoenix** (Phoenix traces).

To enable tracing you must set ADD_PHX_TRACING = True, in config.py

In case of troubles with installing Phoenix  a quick solution is to disable it.

## Factory methods

In the module **prepare_chain_4_chat** are defined the **factory methods** to create: embeddings, llm, reranker...

The philosophy is to make things simpler. So all the configuration are taken from config.py.

If you want to change the llm, (or anything else) go to config.py. No params in the NBs
