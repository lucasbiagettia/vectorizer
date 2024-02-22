import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub
from src.config.config_provider import ConfigProvider



class TextGenerationSingleton:
    _instance = None

    def __new__(cls, model_id, use_api_key = True):
        if cls._instance is None:
            cls._instance = super(TextGenerationSingleton, cls).__new__(cls)
            config_provider = ConfigProvider()
            inference_model_args = config_provider.get_inference_model_args()
            temperature = inference_model_args['temperature']
            max_new_tokens = inference_model_args['max_new_tokens']
            # num_return_sequences = 1
            # no_repeat_ngram_size = 6
            # top_k = 35
            # top_p = 0.95
            if (use_api_key):
                api_token = os.getenv('HF_TOKEN')
                cls._instance.hf_pipeline = HuggingFaceHub(
                    huggingfacehub_api_token= api_token,
                    repo_id= model_id,
                    
                    model_kwargs={
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        # "num_return_sequences": 1,
                        # "no_repeat_ngram_size": 6,
                        # "top_k": 35,
                        # "top_p": 0.95,
                    }
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(model_id)
                pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens, temperature=temperature)
                cls._instance.hf_pipeline = HuggingFacePipeline(pipeline= pipe)
        return cls._instance

    @classmethod
    def get_pipeline(cls, model_id):
        instance = cls(model_id)
        return instance.hf_pipeline

