import csv

csv_file = '/content/borges_metadata_text.csv'





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

with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerow(['metadata', 'embedding'])

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