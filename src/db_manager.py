import os
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from psycopg2.pool import ThreadedConnectionPool


class PosgresManager:
    def __init__(self, dbname):
        self.dbname = dbname
        self.user = os.environ.get('DB_USER')
        self.password = os.environ.get('DB_PASSWORD')
        self.host = os.environ.get('DB_HOST')
        self.port = os.environ.get('DB_PORT')

        self.connection_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            options="-c client_encoding=utf8"
        )

    def get_connection(self):
        return self.connection_pool.getconn()

    def release_connection(self, connection):
        self.connection_pool.putconn(connection)

    def close_all_connections(self):
        self.connection_pool.closeall()


    def _create_vector_extension(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                register_vector(conn)
        finally:
            self.release_connection(conn)

    def create_table_with_id_name(self, name):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                table_create_command = """
                CREATE TABLE IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    nombre VARCHAR(255)
                );
                """.format(name)
                cur.execute(table_create_command)
                conn.commit()
        finally:
            self.release_connection(conn)

    def create_table(self, name, dimension):
        self._create_vector_extension()  #

        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                table_create_command = """
                CREATE TABLE IF NOT EXISTS {} (
                    title text,
                    embedding vector({})
                );
                """.format(name, dimension)
                cur.execute(table_create_command)
                conn.commit()
        finally:
            self.release_connection(conn)

    def insert_tabular_data(self, name, table_name = 'index'):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                insert_command = """
                INSERT INTO {} (nombre) VALUES (%s);
                """.format(table_name)

                cur.execute(insert_command, (name,))
                conn.commit()
        finally:
            self.release_connection(conn)


    def get_index_by_name(self,  name, table_name ='index'):
      
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                select_command = """
                SELECT id FROM {} WHERE nombre = %s;
                """.format(table_name)

                cur.execute(select_command, (name,))
                result = cur.fetchone()

                if result:
                    return result[0]
                else:
                    return None
        finally:
            self.release_connection(conn)


    def insert_data(self, data, table_name):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                data_list = [(row['title'], np.array(row['embeddings'])) for _, row in data.iterrows()]
                execute_values(cur, "INSERT INTO {} (title, embedding) VALUES %s".format(table_name), data_list)
                conn.commit()
        finally:
            self.release_connection(conn)

    def get_similar_docs(self, query_embedding, number, table):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                embedding_array = np.array(query_embedding)
                register_vector(conn)
            
                query = f"SELECT title FROM {table} ORDER BY embedding <=> %s LIMIT %s"
                cur.execute(query, (embedding_array, number))
            
                top_docs = cur.fetchall()
                text_top_docs = [doc[0] for doc in top_docs]
                return text_top_docs
        finally:
            self.release_connection(conn)
    
    def get_table_names(self):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                query = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public';
                """
                cur.execute(query)
                table_names = cur.fetchall()
                table_names = [name[0] for name in table_names]
                return table_names
        finally:
            self.release_connection(conn)

