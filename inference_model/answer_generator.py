import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class ConversationalAgent:
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HF_TOKEN')
    MODEL_ID = "tiiuae/falcon-7b-instruct"

    TEMPLATE = """
    Sistema: Eres un chatbot inteligente. Debes responder la siguiente pregunta en base al contexto.
    Pregunta: {query}.
    Contexto: {context}.
    Respuesta:"""

    INPUT_VARIABLES = ['query']

    def __init__(self):
        self.conv_model = HuggingFaceHub(huggingfacehub_api_token=self.HUGGINGFACEHUB_API_TOKEN,
                            repo_id=self.MODEL_ID,
                            model_kwargs={
                                "temperature": 0.8,
                                "max_new_tokens": 200,
                                "num_return_sequences": 1,
                                "no_repeat_ngram_size": 6,
                                "top_k": 35,
                                "top_p": 0.95,
                            })


        self.prompt = PromptTemplate(template=self.TEMPLATE, input_variables=self.INPUT_VARIABLES)

        self.conv_chain = LLMChain(llm=self.conv_model,
                                   prompt=self.prompt,
                                   verbose=True)

    def answer_question(self, question, context):
        input_data = {'query': question, 'context': context}
        return self.conv_chain.invoke(input_data)

