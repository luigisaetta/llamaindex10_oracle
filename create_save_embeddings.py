"""
File name: create_save_embeddings.py
Author: Luigi Saetta
Date created: 2023-12-14
Date last modified: 2024-03-24
Python Version: 3.11

Description:
    This module provides the code to create and store embeddings and text
    in Oracle DB
    Create embeddings with OCI GenAI, Cohere V3 and loads in Oracle Vector DB

Usage:
    The programs takes all the config from config.py (and secrets from config_private.py)
    Example:
        python create_save_embeddings.py

License:
    This code is released under the MIT License.

Notes:
    This is a part of a set of demo showing how to use Oracle Vector DB,
    OCI GenAI service, Oracle GenAI Embeddings, to build a RAG solution,
    where all he data (text + embeddings) are stored in Oracle DB 23c 

Warnings:
    This module is in development, may change in future versions.
"""

import logging
import re
from typing import List
import time
from glob import glob
import numpy as np

# to generate id from text
import hashlib

from tqdm import tqdm

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

import oracledb
import ads


# This is the wrapper for GenAI Embeddings
from ads.llm import GenerativeAIEmbeddings

from oci_utils import load_oci_config
from oracle_vector_db import save_embeddings_in_db

# this way we don't show & share
from config_private import (
    DB_USER,
    DB_PWD,
    DB_SERVICE,
    DB_HOST_IP,
    COMPARTMENT_OCID,
    ENDPOINT,
)

#
# Configs
#
from config import (
    DIR_BOOKS,
    EMBED_MODEL,
    ID_GEN_METHOD,
    ENABLE_CHUNKING,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP,
)

# to create embeddings in batch
BATCH_SIZE = 40

#
# Functions
#


def generate_id(nodes_list: List):
    """
    get a list of nodes (pages, chunks) and generate the id

    return: list of id
    """
    if ID_GEN_METHOD == "LLINDEX":
        nodes_ids = [doc.id_ for doc in nodes_list]
    # this way generated hashing the page
    if ID_GEN_METHOD == "HASH":
        logging.info("Hashing to compute id...")
        nodes_ids = []
        for doc in tqdm(nodes_list):
            encoded_text = doc.text.encode()
            hash_object = hashlib.sha256(encoded_text)
            hash_hex = hash_object.hexdigest()
            nodes_ids.append(hash_hex)

    return nodes_ids


def read_and_split_in_pages(input_files):
    """
    read the content of a set of pdf files and split in chunks
    """
    pages = SimpleDirectoryReader(input_files=input_files).load_data()

    logging.info(f"Read total {len(pages)} pages...")

    # preprocess text
    for doc in pages:
        doc.text = preprocess_text(doc.text)

    # remove pages with num words < threshold
    pages = remove_short_pages(pages, threshold=10)

    # create a list of text (these are the chuncks to be embedded and saved)
    pages_text = [doc.text for doc in pages]

    # 23/12 register the num of the page
    # must be a string
    pages_num = [doc.metadata["page_label"] for doc in pages]

    # extract list of id
    # this way id have been generated by llama-index
    # 08/01/2024 refactored
    pages_id = generate_id(pages)

    return pages_text, pages_id, pages_num


# in case chunking is enabled
def read_and_split_in_chunks(input_files):
    """
    read a set of pdf files and split in chunks
    """
    node_parser = SentenceSplitter(
        chunk_size=MAX_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    pages = SimpleDirectoryReader(input_files=input_files).load_data()

    logging.info(f"Read total {len(pages)} pages...")

    # preprocess text
    for doc in pages:
        doc.text = preprocess_text(doc.text)

    # remove pages with num words < threshold
    pages = remove_short_pages(pages, threshold=10)

    # splits in chunks
    nodes = node_parser.get_nodes_from_documents(pages, show_progress=True)

    # create a list of text (these are the chuncks to be embedded and saved)
    nodes_text = [doc.text for doc in nodes]

    # 23/12 register the num of the page
    # must be a string
    pages_num = [doc.metadata["page_label"] for doc in nodes]

    nodes_id = generate_id(nodes)

    return nodes_text, nodes_id, pages_num


# some simple text preprocessing
def preprocess_text(text):
    """
    adds some preprocessing, to be customized !
    """
    text = text.replace("\t", " ")
    text = text.replace(" -\n", "")
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")

    # remove repeated blanks
    text = re.sub(r"\s+", " ", text)

    return text


# remove pages with num words < threshold
def remove_short_pages(pages, threshold):
    """
    remove pages with < threshold chars
    """
    n_removed = 0
    for pag in pages:
        if len(pag.text.split(" ")) < threshold:
            pages.remove(pag)
            n_removed += 1

    logging.info(f"Removed {n_removed} short pages...")

    return pages


def check_tokenization_length(tokenizer, batch):
    """
    Check that the number of token dosn't exceed a threshold
    It is an hard check (fails)
    """
    for text in tqdm(batch):
        assert len(tokenizer.encode(text)) <= MAX_CHUNK_SIZE
    logging.info("Tokenization OK...")


# take the list of txts and return a list of embeddings vector
def compute_embeddings(embed_model, nodes_text):
    """
    compute embeddings in batch
    """
    embeddings = []
    for i in tqdm(range(0, len(nodes_text), BATCH_SIZE)):
        batch = nodes_text[i : i + BATCH_SIZE]

        # here we compute embeddings for a batch
        embeddings_batch = embed_model.embed_documents(batch)
        # add to the final list
        embeddings.extend(embeddings_batch)

    return embeddings


# this function is called once for each book
# and saves in DB all the pages of the book + embeddings
def save_chunks_in_db(pages_text, pages_id, pages_num, book_id, connection):
    tot_errors = 0

    with connection.cursor() as cursor:
        logging.info("Saving texts to DB...")
        cursor.setinputsizes(None, oracledb.DB_TYPE_CLOB)

        for id, text, page_num in zip(tqdm(pages_id), pages_text, pages_num):
            try:
                cursor.execute(
                    "insert into CHUNKS (ID, CHUNK, PAGE_NUM, BOOK_ID) values (:1, :2, :3, :4)",
                    [id, text, page_num, book_id],
                )
            except Exception as e:
                logging.error("Error in save chunks...")
                logging.error(e)
                tot_errors += 1

    logging.info(f"Tot. errors in save_chunks: {tot_errors}")


# with this function every book added to DB is registered with a unique id
def register_book(book_name, connection):
    with connection.cursor() as cursor:
        # get the new key
        cursor.execute("SELECT MAX(ID) FROM BOOKS")

        # Fetch the result
        row = cursor.fetchone()

        if row[0] is not None:
            new_key = row[0] + 1
        else:
            new_key = 1

    # insert the record for the book
    with connection.cursor() as cursor:
        query = "INSERT INTO BOOKS (ID, NAME) VALUES (:1, :2)"

        # Execute the query with your values
        cursor.execute(query, [new_key, book_name])

    return new_key


#
# Main
#

# mark start
time_start = time.time()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

print("")
print("Start processing...")
print("")
print("List of books to be loaded and indexed:")

# print list of book to be loaded
# 24/07 modified
input_files = glob(DIR_BOOKS + "/*.pdf")

for book_name in input_files:
    print(book_name)
print("")

oci_config = load_oci_config()

# need to do this way
api_keys_config = ads.auth.api_keys(oci_config)

# the embedding client
embed_model = GenerativeAIEmbeddings(
    compartment_id=COMPARTMENT_OCID,
    model=EMBED_MODEL,
    auth=api_keys_config,
    # LS (05/02/2024) modified to avoid chunking and eerrors if tokens > 512
    # its is a choice to simplify
    truncate="END",
    # Optionally you can specify keyword arguments for the OCI client, e.g. service_endpoint.
    client_kwargs={"service_endpoint": ENDPOINT},
)

# connect to db
logging.info("Connecting to Oracle DB...")

DSN = f"{DB_HOST_IP}/{DB_SERVICE}"

with oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN) as connection:
    logging.info("Successfully connected to Oracle Database...")

    num_pages = []
    for book in input_files:
        logging.info(f"Processing book: {book}...")

        if ENABLE_CHUNKING is False:
            # chunks are pages
            logging.info("Chunks are pages of the book...")
            nodes_text, nodes_id, pages_num = read_and_split_in_pages([book])
            num_pages.append(len(nodes_text))
        else:
            logging.info(f"Enabled chunking, chunck_size: {MAX_CHUNK_SIZE}...")
            nodes_text, nodes_id, pages_num = read_and_split_in_chunks([book])

        # create embeddings
        # process in batch (max 96 for batch, chosen BATCH_SIZE, see above)
        logging.info("Computing embeddings...")

        embeddings = compute_embeddings(embed_model, nodes_text)

        # determine book_id and save in table BOOKS
        logging.info("Registering book...")

        book_id = register_book(book, connection)

        # store embeddings
        # here we save in DB
        save_embeddings_in_db(embeddings, nodes_id, connection)

        logging.info("Save embeddings OK...")

        # store text chunks (pages for now)
        save_chunks_in_db(nodes_text, nodes_id, pages_num, book_id, connection)

        # a txn is a book
        connection.commit()

        logging.info("Save texts OK...")

    # end !!!
    tot_pages = np.sum(np.array(num_pages))

time_elapsed = time.time() - time_start

print("")
print("Processing done !!!")
print(
    f"We have processed {tot_pages} pages and saved text chunks and embeddings in the DB"
)
print(f"Total elapsed time: {round(time_elapsed, 0)} sec.")
print()
