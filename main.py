# main.py
import csv
from vectorizer import EmbeddingModel
from data_processor import DataProcessor
from pinecone_indexer import PineconeIndexer

def main():
    vectorizer = EmbeddingModel()
    data_processor = DataProcessor()
    pinecone_indexer = PineconeIndexer(
        api_key='9eed3e4b-a2ce-4368-99fd-1b30a1ed9521',
        environment='gcp-starter'
    )

    csv_file = '/content/borges_metadata_text.csv'
    data = data_processor.order_data(csv_file, ['text_metadata', 'text'])
    d2v = data_processor.data2vector2(data, vectorizer)

    csv_file_path = 'result_list3.csv'
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['metadata', 'embedding'])
        csv_writer.writerows(d2v)

    print(f'Results written to {csv_file_path}')

    bandera = True
    for i in range(len(d2v)):
        text = d2v[i][0]
        if len(text) > 500:
            print(i)
            bandera = False

    if bandera:
        for ids_vectors_chunk in chunks(d2v, batch_size=100):
            pinecone_indexer.upsert_vectors(ids_vectors_chunk)

    query = "la arena"
    xq = vectorizer.get_embedding(query)
    xc = pinecone_indexer.query_index(xq, top_k=5, include_metadata=True)
    print(xc['matches'][0])

if __name__ == "__main__":
    main()
