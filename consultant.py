import json
import unicodedata
import numpy as np
import pandas as pd
from data_processor import DataProcessor
from pinecone_indexer import PineconeIndexer
from posgres_writer import DatabaseManager
from vectorizer import EmbeddingModel

embedding_model = EmbeddingModel()


dbname = 'vectorpoc'


db_manager = DatabaseManager(dbname)
db_manager.connect()
table_name = "borges2"
user_inp = ""

def decode_json(result):
    for row in result:
        for value in row:
            json_object = json.loads(value)
            print ("Titulo:")
            print (json_object['title']) 
            print ("texto")
            print (json_object['chunk'])

while user_inp != "fin":
    user_inp = input("Que quieres leer: ")
    embedding = embedding_model.get_embedding(user_inp)
    sim_docs = db_manager.get_similar_docs(embedding, 5, table_name)
    decode_json(sim_docs)


db_manager.close_connection()


