import pinecone

class PineconeIndexer:
    def __init__(self, api_key, environment, index_name='borges'):
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)

    def upsert_vectors(self, vectors):
        self.index.upsert(vectors=vectors)

    def query_index(self, query_vector, top_k=5, include_metadata=True):
        return self.index.query(query_vector, top_k=top_k, include_metadata=include_metadata)
