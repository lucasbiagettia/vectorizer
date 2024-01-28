from transformers import RobertaTokenizer, RobertaModel
import torch

class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
            model = RobertaModel.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
            directorio_guardado = "model"
            model.save_pretrained(directorio_guardado)
            tokenizer.save_pretrained(directorio_guardado)

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