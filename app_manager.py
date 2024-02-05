import os
from src.data_processor import TxtProcessor
from src.db_manager import PosgresManager



class AppManager:
    document_list = []

    def get_entry_name(self, document,model_name,splitter):
        entry_name = "-".join([document, model_name, splitter])
        return entry_name
    def get_embed_table_name(index, additional_argument):
    # Hacer algo con index y additional_argument
        return "embed" + str(index)

    def add_document(self, dbname, document, embedding_model, model_name, splitter):
        file_name, file_extension = os.path.splitext(document.name)
        entry_name = self.get_entry_name(file_name,model_name,splitter)
        db_manager = PosgresManager(dbname)
        db_manager.create_table_with_id_name()
        db_manager.insert_tabular_data(entry_name)
        index = db_manager.get_index_by_name(entry_name)
        embed_table_name = self.get_embed_table_name(index)
        data_processor = TxtProcessor(document, embedding_model)
        df = data_processor.get_processed_data()
        dim = embedding_model.get_embedding_dim()
        db_manager.create_table(embed_table_name, dim)
        db_manager.insert_data(df, embed_table_name)



    def make_question(self, dbname, document, question, embedding_model, inference_model):
        table_name = self.get_embed_table_name(document[0]) 
        db_manager = PosgresManager(dbname)
        embedding = embedding_model.get_embedding(question)

        sim_docs = db_manager.get_similar_docs(embedding, 5, table_name)
        ans = inference_model.answer_question(question, sim_docs)
        return ans

    def get_all_documents(self, dbname):
        db_manager = PosgresManager(dbname)
        entries = db_manager.get_all_entries()

        return entries