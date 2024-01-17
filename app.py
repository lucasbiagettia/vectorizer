from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')

import torch

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ... (previous code remains unchanged)

# Move the model to the GPU
model.to(device)

import torch
def get_embedding(text):
  tokens = tokenizer( text,return_tensors="pt", padding=True, truncation=True).to(device)

  with torch.no_grad():
        output = model(**tokens)

  embedding = output.last_hidden_state[:, 0, :]

  return embedding.tolist()[0]

import csv

csv_file = '/content/borges_metadata_text.csv'



def order_data(csv_file, column_names):
    result_list = []

    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Read the header

        if all(column in header for column in column_names):
            for i, row in enumerate(csv_reader):

                values = [row[header.index(column)] for column in column_names]
                result_list.append(tuple(values))
        else:
            raise Exception("One or more specified columns are not present in the file.")

    return result_list



data = order_data(csv_file, ['text_metadata', 'text'])
def split_into_batches(text, words_per_batch=20):
    embeds =[]
    words = text.split()
    for i in range(0, len(words), words_per_batch):
        batch = " ".join(words[i:i+words_per_batch])
        embeds.append(batch)
    return embeds

import json

def add_field_to_json(json_str, value):
    json_data = json.loads(json_str)
    json_data['chunk'] = value
    del json_data['link']
    del json_data['author']
    del json_data['metadata']
    return json.dumps(json_data).replace('"', "'")


def split_into_batches(text, words_per_batch=50):
    embeds =[]
    words = text.split()
    for i in range(0, len(words), words_per_batch):
        batch = " ".join(words[i:i+words_per_batch])
        embeds.append(batch)
    return embeds


import tqdm

def data2vector2(data):
    result_list = []

    # Use tqdm to create a progress bar
    for metadata, text in tqdm.tqdm(data, desc="Processing data"):
        chunks = split_into_batches(text)
        for chunk in chunks:
            result_list.append((add_field_to_json(metadata, chunk), get_embedding(chunk)))

    return result_list



d2v = data2vector2(data)

csv_file_path = 'result_list3.csv'

# Write result_list to CSV file
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    csv_writer.writerow(['metadata', 'embedding'])

    # Write data
    csv_writer.writerows(d2v)

print(f'Results written to {csv_file_path}')

bandera = True
for i in range (len(d2v)):
  text = d2v[i][0]
  if len(text) > 500:
    print(i)
    bandera = False



import pinecone
pinecone.init(
            api_key='9eed3e4b-a2ce-4368-99fd-1b30a1ed9521',
            environment='gcp-starter'
        )

'''pinecone.create_index(
                "borges",
                dimension=768,
                metric='cosine'
            )'''

index = pinecone.Index("borges")

import itertools
def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

if(bandera):
  for ids_vectors_chunk in chunks(d2v, batch_size=100):
      index.upsert(vectors=ids_vectors_chunk)

query = "la arena"

# create the query vector
xq = get_embedding(query)

# now query
xc = index.query(xq, top_k=5, include_metadata=True)
xc

print(xc['matches'][0])