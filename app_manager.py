import os
from src.data_processor import TxtProcessor
from src.db_manager import PosgresManager



class AppManager:
    document_list = []

    def get_entry_name(self, document,model_name,splitter):
        entry_name = document + model_name + splitter
        return entry_name
    def add_document(self, dbname, document, embedding_model, model_name, splitter):
        print("añado")
        file_name, file_extension = os.path.splitext(document.name)
        entry_name = self.get_entry_name(file_name,model_name,splitter)
        db_manager = PosgresManager(dbname)
        db_manager.create_table_with_id_name()
        db_manager.insert_tabular_data(entry_name)
        print('sadas')
        embed_table_name = "embed" + db_manager.get_index_by_name(entry_name)
        data_processor = TxtProcessor(document, embedding_model)
        df = data_processor.get_processed_data()
        dim = embedding_model.get_embedding_dim()
        db_manager.create_table(embed_table_name, dim)
        db_manager.insert_data(df, embed_table_name)

    def make_question(self, dbname, document, question, embedding_model, inference_model):
        table_name = self.get_entry_name(document)
        db_manager = PosgresManager(dbname)
        db_manager.connect()
        embedding = embedding_model.get_embedding(question)
        sim_docs = db_manager.get_similar_docs(embedding, 5, table_name)
        ans = inference_model.answer_question(question, sim_docs)
        return ans

    def get_all_tables(self, dbname):
        db_manager = PosgresManager(dbname)
        db_manager.connect()
        tables = db_manager.get_table_names()
        return tables