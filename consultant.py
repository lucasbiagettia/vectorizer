from db_manager import PosgresManager
from inference_model.answer_generator import ConversationalAgent
from vectorizer import EmbeddingModel

embedding_model = EmbeddingModel()


dbname = 'vectorpoc'


db_manager = PosgresManager(dbname)
db_manager.connect()
table_name = "marx"
user_inp = ""


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


