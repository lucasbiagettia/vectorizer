import numpy as np
import pandas as pd
from data_processor import DataProcessor
from pinecone_indexer import PineconeIndexer
from posgres_writer import DatabaseManager
from vectorizer import EmbeddingModel

embedding_model = EmbeddingModel()


csv_file = 'sample_data/borges_metadata_text.csv'
column_names = ['text_metadata', 'text']

data_processor = DataProcessor(csv_file, column_names, embedding_model)
d2v = data_processor.data2vector2(data_processor.order_data())

dbname = 'vectorpoc'


db_manager = DatabaseManager(dbname)
db_manager.connect()
db_manager.create_extension()
dim = len(d2v[0][1])
table_name = "borges2"
db_manager.create_table(table_name, dim)



# Inicializa un DataFrame vacío con columnas 'Texto' y 'Numeros'
df = pd.DataFrame(columns=['title', 'embeddings'])

# Itera sobre el array de pares y agrega cada par como una nueva fila
for text, embed in d2v:
    #df = df.append({'title': text, 'embeddings': embed}, ignore_index=True)
    df = pd.concat([df, pd.DataFrame({'title': [text], 'embeddings': [embed]})], ignore_index=True)



db_manager.insert_data(df, table_name)

db_manager.close_connection()

# import csv

# # Ruta al archivo CSV de salida
# output_csv_file = '/out.csv'



# api_key = '9eed3e4b-a2ce-4368-99fd-1b30a1ed9521'
# environment = 'gcp-starter'
# index_name = 'borges'

# # Crear una instancia de PineconeIndexer
# pinecone_indexer = PineconeIndexer(api_key, environment, index_name)

# # Utilizar la clase para upsert (insertar o actualizar) datos en el índice de Pinecone
# pinecone_indexer.upsert_data(d2v)


# # Realizar una consulta
# query = "la arena"
# xq = embedding_model.get_embedding(query)
# xc = pinecone_indexer.query(xq, top_k = 5)

# print(xc['matches'][0])

