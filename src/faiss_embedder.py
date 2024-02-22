import tempfile
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


class EmbeddingsProvider:
    _instance = None

    def __new__(cls, embedding_model, document):
        if cls._instance is None:
            cls._instance = super(EmbeddingsProvider, cls).__new__(cls)
            cls._instance.initialize(embedding_model, document)
        return cls._instance

    def initialize(self, embedding_model, document):
        self.initialized = False
        self.embedding_model = embedding_model
        self.documents = self.load_documents(document)
        self.embeddings, self.knowledge_base = self.create_embeddings(self.embedding_model, self.documents)
        self.initialized = True

    def load_documents(self, file_path):
        file_content = file_path.read()
        loader = PyPDFLoader(self.get_file_path_in_memory(file_content))
        return loader.load_and_split()
    
    def get_file_path_in_memory(self, file_content, file_extension=".pdf"):
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        return temp_file_path
    

    def create_embeddings(self, model_name, documents):
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

            
        knowledge_base = FAISS.from_documents(documents, embeddings)
        return embeddings, knowledge_base
    def get_embeddings(self):
        return self.embeddings, self.knowledge_base

