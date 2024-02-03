import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class InferenceModel:
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')
    TEMPLATE = """
    Sistema: Eres un chatbot inteligente. Debes responder la siguiente pregunta en base al contexto.
    Pregunta: {query}.
    Contexto: {context}.
    """
    INPUT_VARIABLES = ["query", "context"]

    def __init__(self, model_id="tiiuae/falcon-7b"):
        self.MODEL_ID = model_id
        self.conv_model = HuggingFaceHub(
            huggingfacehub_api_token=self.HUGGINGFACEHUB_API_TOKEN,
            repo_id=self.MODEL_ID,
            model_kwargs={
                "temperature": 0.8,
                "max_new_tokens": 200,
                "num_return_sequences": 1,
                "no_repeat_ngram_size": 6,
                "top_k": 35,
                "top_p": 0.95,
            }
        )
        self.prompt = PromptTemplate(template=self.TEMPLATE, input_variables=self.INPUT_VARIABLES)
        self.conv_chain = LLMChain(llm=self.conv_model, prompt=self.prompt, verbose=True)

    def print_dictionary(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, (dict, list)):
                print(f"{key}:")
                self.print_recursive(value)
            else:
                print(f"{key}: {value}")

    def print_recursive(self, element):
        if isinstance(element, dict):
            self.print_dictionary(element)
        elif isinstance(element, list):
            for sub_element in element:
                self.print_recursive(sub_element)
        else:
            print(element)

    def answer_question(self, question, context):
        input_data = {'query': question, 'context': context}
        res = self.conv_chain.invoke(input_data)
        self.print_dictionary(res)
        return (res)

