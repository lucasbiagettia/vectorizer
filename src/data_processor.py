import json
import PyPDF2
import pandas as pd



class TxtProcessor:
    def __init__(self, pdf_file, embedding_model):
        self.pdf_file = pdf_file
        self.embedding_model = embedding_model
        
    def read_pdf(self):
        with self.pdf_file as file:
            pdf_reader = PyPDF2.PdfReader(file)
        
            
            pdf_content = ""
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text_page = page.extract_text()
                pdf_content += text_page
            return pdf_content
        
    def split_into_sentences(self, text):
        sentences = text.split(".")
        sentences = [sentence.strip() for sentence in sentences if sentence]
        sentences = [sentence for sentence in sentences if len(sentence.split()) >= 20]
        return sentences
    
    def generate_batches(self, string_array):
        batches = [string_array[0]]
        for string in string_array[1:]: 
            words = len(batches[-1].split()) 
            if words + len(string.split()) <= 80:  
                batches[-1] += string 
            else:
                batches.append(string)  
        return batches

    def split_into_batches(self, text):
        sentences = self.split_into_sentences(text)
        batches = self.generate_batches(sentences)
        return batches

   
    def add_to_json (self, text):
        json_result = {'chunk': text}
        return json.dumps(json_result)
    
    def data_to_dataframe(self, batches):
        df = pd.DataFrame(columns=['title', 'embeddings'])
        for batch in batches:
            df = pd.concat([df, pd.DataFrame({'title': [batch], 'embeddings': [self.embedding_model.get_embedding(batch)]})], ignore_index=True)

        return df
    
    def get_processed_data(self):
        text = self.read_pdf()
        batches = self.split_into_batches(text)
        df = self.data_to_dataframe(batches)
        return df



