from langchain.prompts import PromptTemplate

class PromptGenerator:
    def __init__(self):
        self.qa_template = """
        Sistema: Eres un asistente virtual preparado para responder preguntas sobre un lenguaje de programación llamado FastPrg.
        FastPrg es un lenguaje de programación diseñado para crear aplicaciones de negocios.
        A continuación se te fragmentos de los documentos que explican el funcionamiento de FastPrg y tu tarea es responder la pregunta.
        Si en los fragmentos de documento no se encuentra la respuesta responde que no es posible elaborar una respuesta.
        Contexto: {context}
        Pregunta: {question}
        Respuesta: """

        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=self.qa_template,
        )

    def get_qa_prompt(self):
        return self.QA_CHAIN_PROMPT
     


