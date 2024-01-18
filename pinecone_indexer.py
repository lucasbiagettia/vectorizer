import pinecone
import itertools

class PineconeIndexer:
    def __init__(self, api_key, environment, index_name):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def chunks(self, iterable, batch_size=100):
        """A helper function to break an iterable into chunks of size batch_size."""
        it = iter(iterable)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            yield chunk
            chunk = tuple(itertools.islice(it, batch_size))
            

    def upsert_data(self, data, batch_size=100):
        for ids_vectors_chunk in self.chunks(data, batch_size):
            self.index.upsert(vectors=ids_vectors_chunk)

    def query (self, embedding, top_k):
        return self.index.query(embedding, top_k = top_k, include_metadata=True)

# Uso de la clase

