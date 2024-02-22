from langchain.chains import LLMChain
from src.prompt_provider import PromptGenerator
from src.inference_model_manager import TextGenerationSingleton

class QuestionAnsweringChain:
    def __init__(self, model_id):
        self.model_id = model_id
        self.pipeline = TextGenerationSingleton.get_pipeline(model_id)
        self.QA_CHAIN_PROMPT = PromptGenerator().get_qa_prompt()
        self.chain = LLMChain(llm=self.pipeline, prompt=self.QA_CHAIN_PROMPT)
    

    def answer_question(self, question, context):
        ans = self.chain.invoke({"question": question, 'context': context})
        return self.extract_model_response (ans['text'])
    
    def extract_model_response(self, input_string):
        start_index = input_string.find("<start_of_turn>model") + len("<start_of_turn>model")
        model_response = input_string[start_index:].strip()
        return model_response
    
