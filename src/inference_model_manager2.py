from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class TextGenerationSingleton:
    _instance = None

    def __new__(cls, model_id):
        if cls._instance is None:
            cls._instance = super(TextGenerationSingleton, cls).__new__(cls)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(model_id)
            cls._instance.model = AutoModelForCausalLM.from_pretrained(model_id)
            cls._instance.pipe = pipeline("text-generation", model=cls._instance.model, tokenizer=cls._instance.tokenizer, max_new_tokens=400, temperature=0.1)
        return cls._instance

    @classmethod
    def get_pipeline(cls, model_id):
        instance = cls(model_id)
        return instance.pipe

