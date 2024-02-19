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
        return ans['text']