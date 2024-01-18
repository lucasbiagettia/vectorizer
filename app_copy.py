from transformers import RobertaTokenizer, RobertaModel

from data_processor import DataProcessor
from pinecone_indexer import PineconeIndexer
from vectorizer import EmbeddingModel

embedding_model = EmbeddingModel()


csv_file = 'sample_data/borges_metadata_textcorto.csv'
column_names = ['text_metadata', 'text']

data_processor = DataProcessor(csv_file, column_names, embedding_model)
d2v = data_processor.data2vector2(data_processor.order_data())


import csv

# Ruta al archivo CSV de salida
output_csv_file = '/out.csv'

# Guardar los resultados en el archivo CSV
with open(output_csv_file, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Escribir el encabezado
    csv_writer.writerow(['json_data_with_embedding', 'original_text_chunk'])
    
    # Escribir los datos
    for json_data, original_text_chunk in d2v:
        csv_writer.writerow([json_data, original_text_chunk])

print(f"Resultados exportados a {output_csv_file}")

'''
api_key = '9eed3e4b-a2ce-4368-99fd-1b30a1ed9521'
environment = 'gcp-starter'
index_name = 'borges'

# Crear una instancia de PineconeIndexer
pinecone_indexer = PineconeIndexer(api_key, environment, index_name)

# Utilizar la clase para upsert (insertar o actualizar) datos en el índice de Pinecone
pinecone_indexer.upsert_data(d2v)


# Realizar una consulta
query = "la arena"
xq = embedding_model.get_embedding(query)
xc = index.query(xq, top_k=5, include_metadata=True)

print(xc['matches'][0])
'''
