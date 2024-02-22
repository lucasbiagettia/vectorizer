from langchain.prompts import PromptTemplate

class PromptGenerator:
    def __init__(self):
        # self.qa_template = """
        # Sistema: Eres un asistente virtual preparado para responder preguntas en base fragmentos de un documento que se proporcionan como contexto.
        # Si en los fragmentos de documento no se encuentra la respuesta responde que no es posible elaborar una respuesta.
        # Contexto: {context}
        # Pregunta: {question}
        # Respuesta: """
        self.qa_template = """<start_of_turn>system
        Eres un asistente virtual preparado para responder preguntas en base fragmentos de un documento que se proporcionan como contexto.
        Si en los fragmentos de documento no se encuentra la respuesta responde que no es posible elaborar una respuesta.
        <end_of_turn>
        <start_of_turn>context
        {context}
        <end_of_turn>
        <start_of_turn>user
        {question}
        <end_of_turn>
        <start_of_turn>model
        """


        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=self.qa_template,
        )

    def get_qa_prompt(self):
        return self.QA_CHAIN_PROMPT
     


