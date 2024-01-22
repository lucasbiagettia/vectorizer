import pandas as pd
import numpy as np
import psycopg2
import pgvector
import math
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


dbname = 'vectorPoC'
user = 'postgres'
password = 'pochovive'
host = 'localhost'
port = '5433'

conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
cur = conn.cursor()

#install pgvector
cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()

# Register the vector type with psycopg2
register_vector(conn)

# Create table to store embeddings and metadata
table_create_command = """
CREATE TABLE IF NOT EXISTS embeddings (
            title text,
            embedding vector(10)
            );
            """

cur.execute(table_create_command)
cur.close()
conn.commit()



register_vector(conn)
cur = conn.cursor()



data = {'title': ['Documento 1', 'Documento 2', 'Documento 3', 'Documento 4', 'Documento 5'],
        'embeddings': [np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.rand(10), np.random.rand(10)]}

df_new = pd.DataFrame(data)

# Crear data_list con 5 elementos
data_list = [(row['title'], np.array(row['embeddings'])) for index, row in df_new.iterrows()]
# Use execute_values to perform batch insertion
execute_values(cur, "INSERT INTO embeddings (title,  embedding) VALUES %s", data_list)
# Commit after we insert all embeddings
conn.commit()

"""Sanity check by running some simple queries against the embeddings table"""

cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
num_records = cur.fetchone()[0]
print("Number of vector records in table: ", num_records,"\n")
# Correct output should be 129

# print the first record in the table, for sanity-checking
cur.execute("SELECT * FROM embeddings LIMIT 1;")
records = cur.fetchall()
print("First record in table: ", records)

"""Create index on embedding column for faster cosine similarity comparison"""

# Create an index on the data for faster retrieval
# this isn't really needed for 129 vectors, but it shows the usage for larger datasets
# Note: always create this type of index after you have data already inserted into the DB

#calculate the index parameters according to best practices
num_lists = num_records / 1000
if num_lists < 10:
    num_lists = 10
if num_records > 1000000:
    num_lists = math.sqrt(num_records)

#use the cosine distance measure, which is what we'll later use for querying
cur.execute(f'CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {num_lists});')
conn.commit()


cur.close()
conn.close()