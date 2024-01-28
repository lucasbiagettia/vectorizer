import json
from db_manager import PosgresManager
from inference_model.answer_generator import ConversationalAgent
from vectorizer import EmbeddingModel

embedding_model = EmbeddingModel()


dbname = 'vectorpoc'


db_manager = PosgresManager(dbname)
db_manager.connect()
table_name = "marx"
user_inp = ""

def decode_json(result):
    for row in result:
        for value in row:
            json_object = json.loads(value)
            print ("Titulo:")
            print (json_object['title']) 
            print ("texto")
            print (json_object['chunk'])

def create_json(main_text, context_texts):
    result = {"pregunta": main_text, "contexto": context_texts}
    json_result = json.dumps(result, indent=2, ensure_ascii=False)
    return json_result

while True:
    user_inp = input("Que quieres saber: ")
    if user_inp == 'fin':
        break
    embedding = embedding_model.get_embedding(user_inp)
    sim_docs = db_manager.get_similar_docs(embedding, 5, table_name)
    agent = ConversationalAgent()
    ans = agent.answer_question(user_inp, sim_docs)
    print(ans)

db_manager.close_connection()


