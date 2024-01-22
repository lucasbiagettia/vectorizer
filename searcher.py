import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


dbname = 'vectorpoc'
user = 'postgres'
password = 'pochovive'
host = 'localhost'
port = '5432'

conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
cur = conn.cursor()
# Helper function: Get top 3 most similar documents from the database
embedding_array = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # Register pgvector extension
register_vector(conn)
    # Get the top 3 most similar documents using the KNN <=> operator
cur.execute("SELECT title text FROM embeddings ORDER BY embedding <=> %s LIMIT 3", (embedding_array,))
top3_docs = cur.fetchall()

for doc in top3_docs:
    print("Document Content:", doc[0])
