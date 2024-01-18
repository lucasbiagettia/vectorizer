from transformers import RobertaTokenizer, RobertaModel
import torch
import csv
import json
import tqdm
import pinecone
import itertools

# Inicializar modelo y tokenizer
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Función para obtener la representación embedding
def get_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :].tolist()[0]

# Función para procesar datos
def data2vector2(data):
    result_list = []
    for metadata, text in tqdm.tqdm(data, desc="Processing data"):
        chunks = [get_embedding(chunk) for chunk in split_into_batches(text)]
        result_list.extend([(add_field_to_json(metadata, chunk), chunk) for chunk in chunks])
    return result_list

# Función para dividir texto en lotes
def split_into_batches(text, words_per_batch=20):
    words = text.split()
    return [" ".join(words[i:i+words_per_batch]) for i in range(0, len(words), words_per_batch)]

# Función para añadir campo a JSON
def add_field_to_json(json_str, value):
    json_data = json.loads(json_str)
    del json_data['link'], json_data['author'], json_data['metadata']
    json_data['chunk'] = value
    return json.dumps(json_data).replace('"', "'")

# Función para leer datos del archivo CSV
def order_data(csv_file, column_names):
    result_list = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        if all(column in header for column in column_names):
            result_list.extend([tuple([row[header.index(column)] for column in column_names]) for row in csv_reader])
        else:
            raise Exception("One or more specified columns are not present in the file.")
    return result_list

# Obtener datos del archivo CSV
csv_file = 'C:/Users/Notebook/Documents/Lucas/Programación/gitProjects/vectorizer/sample_data/borges_metadata_text.csv'
data = order_data(csv_file, ['text_metadata', 'text'])

# Procesar datos y obtener embeddings
d2v = data2vector2(data)

# Escribir resultados en archivo CSV
csv_file_path = 'result_list3.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['metadata', 'embedding'])
    csv_writer.writerows(d2v)

print(f'Results written to {csv_file_path}')

# Inicializar Pinecone
pinecone.init(api_key='9eed3e4b-a2ce-4368-99fd-1b30a1ed9521', environment='gcp-starter')
index = pinecone.Index("borges")

# Subir embeddings a Pinecone
for ids_vectors_chunk in chunks(d2v, batch_size=100):
    index.upsert(vectors=ids_vectors_chunk)

# Realizar una consulta
query = "la arena"
xq = get_embedding(query)
xc = index.query(xq, top_k=5, include_metadata=True)

print(xc['matches'][0])
