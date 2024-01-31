from data_processor import CsvProcessor, TxtProcessor
from db_manager import PosgresManager
from vectorizer import EmbeddingModel
from inference_model.answer_generator import InferenceModel

class DataProcessor:
    def __init__(self, file_path, embedding_model):
        self.file_path = file_path
        self.embedding_model = embedding_model

    def process_data(self):
        if self.file_path.endswith('.csv'):
            return CsvProcessor(self.file_path, self.embedding_model).get_processed_data()
        elif self.file_path.endswith('.txt'):
            return TxtProcessor(self.file_path, self.embedding_model).get_processed_data()
        else:
            raise ValueError("Unsupported file format")

class DataUploader:
    def __init__(self, dbname, table_name, data_processor):
        self.dbname = dbname
        self.table_name = table_name
        self.data_processor = data_processor

    def upload_data(self):
        embedding_model = self.data_processor.embedding_model
        df = self.data_processor.process_data()
        dim = embedding_model.get_hidden_size()

        db_manager = PosgresManager(self.dbname)
        db_manager.connect()
        db_manager.create_extension()
        db_manager.create_table(self.table_name, dim)
        db_manager.insert_data(df, self.table_name)
        db_manager.close_connection()



class AnswerProcessor:
    def __init__(self, dbname='vectorpoc', table_name='marx'):
        self.embedding_model = EmbeddingModel()
        self.db_manager = PosgresManager(dbname)
        self.table_name = table_name

    def connect_to_database(self):
        self.db_manager.connect()

    def get_answer(self):
        user_inp = input("Que quieres saber: ")
        embedding = self.embedding_model.get_embedding(user_inp)
        sim_docs = self.db_manager.get_similar_docs(embedding, 5, self.table_name)
        agent = InferenceModel()
        ans = agent.answer_question(user_inp, sim_docs)
        print(ans)

    def close_connection(self):
        self.db_manager.close_connection()

# Example usage:
answer_processor = AnswerProcessor()
answer_processor.connect_to_database()
answer_processor.get_answer()
answer_processor.close_connection()


if __name__ == "__main__":

    pdf_file = 'sample_data/communist_manifest.pdf'
    data_processor = DataProcessor(pdf_file, EmbeddingModel())

    uploader = DataUploader('vectorpoc', 'marx', data_processor)
    uploader.upload_data()
