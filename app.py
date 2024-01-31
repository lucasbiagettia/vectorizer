from data_processor import CsvProcessor, TxtProcessor
from db_manager import PosgresManager
from vectorizer import EmbeddingModel

embedding_model = EmbeddingModel()


# csv_file = 'sample_data/borges_metadata_text.csv'
# column_names = ['text_metadata', 'text']
# data_processor = CsvProcessor(csv_file, column_names, embedding_model)


pdf_file = 'sample_data/communist_manifest.pdf'
data_processor = TxtProcessor(pdf_file, embedding_model)

df = data_processor.get_processed_data()
dim = embedding_model.get_hidden_size()

dbname = 'embeddings'
db_manager = PosgresManager(dbname)
db_manager.connect()
db_manager.create_extension()

table_name = 'poc'
db_manager.create_table(table_name, dim)

db_manager.insert_data(df, table_name)

db_manager.close_connection()



