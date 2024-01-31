from data_processor import TxtProcessor
from db_manager import PosgresManager


class AppManager:
    document_list = []
    def add_document(dbname, document, embedding_model):
        table_name = document.replace(' ', '_').replace('.', '').replace('-', '').replace('.pdf', '').replace('.txt', '')
        data_processor = TxtProcessor(document, embedding_model)
        df = data_processor.get_processed_data()
        dim = embedding_model.get_hidden_size()
        db_manager = PosgresManager(dbname)
        db_manager.connect()
        db_manager.create_extension()
        db_manager.create_table(table_name, dim)
        db_manager.insert_data(df, table_name)
        db_manager.close_connection()

    def make_question(dbname, table_name, question, embedding_model, inference_model):
        db_manager = PosgresManager(dbname)
        db_manager.connect()
        embedding = embedding_model.get_embedding(question)
        sim_docs = db_manager.get_similar_docs(embedding, 5, table_name)
        ans = inference_model.answer_question(question, sim_docs)
        return ans

    def get_all_tables(dbname):
        db_manager = PosgresManager(dbname)
        db_manager.connect()
        tables = db_manager.get_table_names()
        return tables