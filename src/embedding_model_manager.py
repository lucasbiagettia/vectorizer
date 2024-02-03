import os
from transformers import RobertaTokenizer, RobertaModel
import torch

class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            folder_path = 'model'
            model_path = None

            if os.path.isdir(folder_path):
                model_path = folder_path
            else:
                model_path = 'PlanTL-GOB-ES/roberta-base-bne'
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
            model = RobertaModel.from_pretrained(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            cls._instance = super().__new__(cls)
            cls._instance.tokenizer = tokenizer
            cls._instance.model = model
            cls._instance.device = device

        return cls._instance

    def get_embedding(self, text):
      
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            output = self.model(**tokens)

        embedding = output.last_hidden_state[:, 0, :]

        return embedding.tolist()[0]
    
    def get_hidden_size(self):
        return self.model.config.hidden_size