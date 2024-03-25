# Integrate Oracle AI Vector DB and OCI GenAI with Llama-index and LangChain

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
* [LangChain](https://python.langchain.com/docs/get_started/introduction)

In the [Video demos](https://github.com/luigisaetta/llamaindex_oracle/wiki/Video-demos) section of the Wiki you'll find some video of the demo.

## What is RAG?

A very good introduction to what **Retrieval Augmented Generation** (RAG) is can be found [here](https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/)

## Features

* basic (12/2023) integration between **Oracle DB Vector Store (23c)** and **llama-index**
* All documents stored in an Oracle AI Vector Search
* Oracle AI Vector Search used for Semantic Search
* Reranking to improve retrieval
* How to show references (documents used for the response generation)
* (30/12/2023) Added reranker implemented as OCI Model Deployment
* (20/01/2024) Added implementation of Vector Store for LangChain and demo
* Finding duplicates in the documentation
* (2/03/2024) Added Phoenix Traces for observability

## Demos

* [demo1](./custom_vector_store_demo1.ipynb) This NB shows how you get answers to questions on Oracle Database and new features in 23c, using Oracler AI Vector Search
* [demo2](./custom_vector_store_demo2.ipynb) This NB shows a QA demo on Medicine (Covid19), using Public docs from NIH.
* [Bot](./oracle_bot.py) powered by **Oracle Vector DB** and **OCI GenAI**
* [demo3](./custom_vector_store_demo3.ipynb) shows how to add a Reranker to the RAG chain; I have used **Cohere** Reranker
* [demo5](./rag_chain_demo5.ipynb) shows a full RAG chain where the reranker is deployed on OCI DS
* [LangChain](./demo_langchain2.ipynb) demo based on Oracle AI Vector Search and LangChain
* [finding duplicates](./find_duplicates.ipynb): how to identify duplicates in a book, using Embeddings and Similarity Search
* [Knowledge assistant full demo](./run_oracle_chat_with_memory.sh)

## Setup

See the [wiki](https://github.com/luigisaetta/llamaindex_oracle/wiki/Setup-of-the-Python-conda-environment) pages.

## Loading data

* You can use the [create_save_embeddings](./create_save_embeddings.py) Python program to load all the data in the Oracle DB schema.
* You can launch it using the script [load_books](./load_books.sh).
* The list of files to be loaded is specified in the file config.py

You need to have pdf files in the same directory.

## Limited Availability

* **OCI GenAI Service is General Availability** since 23/01/2024, see updated docs for setup

Oracle **AI Vector Search** (Vector DB) is a new feature in Oracle DB 23c, in **Limited Availability**. 

Customers can easily enter in the LA/Beta program.

To test these functionalities you need to enroll in the LA program and install the proper versions of software libraries.

Code and functionalities can change, as a result of feedbacks from customers.

## Releases used for the demo

* OCI 2.119.1
* OCI ADS 2.10.0
* LangChain >= 0.1.4
* LangChain Community >= 0.0.16
* Llama-index >= 0.9.37.post1 < 0.10
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

(02/03/2024) I have added integration with **Arize Phoenix** (Phoenix traces).

To enable tracing you must set ADD_PHX_TRACING = True, in config.py

In case of troubles with installing Phoenix  a quick solution is to disable it.

## Factory methods

In the module prepare_chain are defined the **factory methods** to create: embedder, llm, reranker...

The philosophy is to make things simpler. So all the configuration are taken from config.py.

If you want to change the llm, (or anything else) go to config.py. No params in the NBs
