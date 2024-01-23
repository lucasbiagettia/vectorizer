import os
import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

import pinecone
import itertools

class PosgresManager:
    def __init__(self, dbname):
        self.dbname = dbname
        self.user = os.environ.get('DB_USER')
        self.password = os.environ.get('DB_PASSWORD')
        self.host = os.environ.get('DB_HOST')
        self.port = os.environ.get('DB_PORT')
        
        self.conn = None
        self.cur = None

    def connect(self):
        self.conn = psycopg2.connect(dbname=self.dbname, user=self.user, password=self.password, host=self.host, port=self.port, options="-c client_encoding=utf8")
        self.cur = self.conn.cursor()

    def create_extension(self):
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        self.conn.commit()
        register_vector(self.conn)

    def create_table(self, name, dimension):
        table_create_command = """
        CREATE TABLE IF NOT EXISTS {} (
            title text,
            embedding vector({})
        );
        """.format(name, dimension)
        self.cur.execute(table_create_command)
        self.conn.commit()


    def insert_data(self, data, table_name):
        data_list = [(row['title'], np.array(row['embeddings'])) for _, row in data.iterrows()]
        execute_values(self.cur, "INSERT INTO {} (title, embedding) VALUES %s".format(table_name), data_list)
        self.conn.commit()



    def get_similar_docs(self, query_embedding, number, table):
        embedding_array = np.array(query_embedding)
        register_vector(self.conn)
        
        query = f"SELECT title FROM {table} ORDER BY embedding <=> %s LIMIT %s"
        self.cur.execute(query, (embedding_array, number))
        
        top_docs = self.cur.fetchall()
        text_top_docs = [doc[0] for doc in top_docs]
        return text_top_docs


    def close_connection(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()


class PineconeManager:
    def __init__(self, api_key, environment, index_name):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
            

    def upsert_data(self, data, batch_size=100):
        for ids_vectors_chunk in self.chunks(data, batch_size):
            self.index.upsert(vectors=ids_vectors_chunk)

    def query (self, embedding, top_k):
        return self.index.query(embedding, top_k = top_k, include_metadata=True)
