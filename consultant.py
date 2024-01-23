import json
from posgres_writer import PosgresManager
from vectorizer import EmbeddingModel

embedding_model = EmbeddingModel()


dbname = 'vectorpoc'


db_manager = PosgresManager(dbname)
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


