from langchain_community.embeddings import HuggingFaceEmbeddings


    
class EmbeddingModel:
    _instance = None
    _embedding_dim = None

    def __new__(cls, model_name=None):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.model_kwargs = {'device': 'cpu'}
            cls._instance.encode_kwargs = {'normalize_embeddings': False}
            cls._instance.hf = HuggingFaceEmbeddings(
                model_name=cls._instance.model_name,
                model_kwargs=cls._instance.model_kwargs,
                encode_kwargs=cls._instance.encode_kwargs
            )
            cls._instance._embedding_dim = cls._instance.hf.client.get_sentence_embedding_dimension()
        return cls._instance
    
    def get_embedding(self, text):
        embed = self.hf.embed_query(text)
        if self._embedding_dim == None:
            self._embedding_dim = len(embed)
        return embed
    
    def get_multiple_embedding(self, texts):
        return self.hf.embed_documents(texts)
    
    def get_embedding_dim(self):
        return self._embedding_dim

    


