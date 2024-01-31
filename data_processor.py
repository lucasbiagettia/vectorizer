import json
import csv
import PyPDF2
import tqdm
import re
import pandas as pd


class CsvProcessor:
    def __init__(self, csv_file, column_names, embedding_model):
        self.csv_file = csv_file
        self.column_names = column_names
        self.embedding_model = embedding_model


    def split_sentences(self, text):
        regex = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = regex.split(text)
        processed_sentences = []
        current_word_count = 0

        for sentence in sentences:
            words = sentence.split()
            num_words_sentence = len(words)

            if num_words_sentence >= 20:
                processed_sentences.append(sentence)
                current_word_count = 0
            else:
                if processed_sentences:
                    processed_sentences[-1] += ' ' + sentence
                    current_word_count += num_words_sentence
                else:
                    processed_sentences.append(sentence)
                    current_word_count = num_words_sentence

        if len(processed_sentences[-1].split()) < 20:
            processed_sentences.pop()
        return processed_sentences

        
    def add_field_to_json(self, json_str, value):
        json_data = json.loads(json_str)
        json_data['chunk'] = value
        del json_data['link']
        del json_data['author']
        del json_data['metadata']
        return json.dumps(json_data, ensure_ascii=False)

    def order_data(self):
        result_list = []

        with open(self.csv_file, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader) 
            if all(column in header for column in self.column_names):
                for i, row in enumerate(csv_reader):
                    values = [row[header.index(column)] for column in self.column_names]
                    result_list.append(tuple(values))
            else:
                raise Exception("One or more specified columns are not present in the file.")

        return result_list

    def get_embedding(self, text):
        return self.embedding_model.get_embedding(text)
   
    def data2vector(self, data):
        result_list = []
        for metadata, text in tqdm.tqdm(data, desc="Processing data"):
            chunks = self.split_sentences(text)
            for chunk in chunks:
                result_list.append((self.add_field_to_json(metadata, chunk), self.get_embedding(chunk)))

        return result_list
    
    def data_to_dataframe(self, d2v):
        df = pd.DataFrame(columns=['title', 'embeddings'])
        for text, embed in d2v:
            df = pd.concat([df, pd.DataFrame({'title': [text], 'embeddings': [embed]})], ignore_index=True)
        return df

    
    def get_processed_data(self):
        ordered_data = self.order_data()
        d2v = self.data2vector(ordered_data)
        df = self.data_to_dataframe(d2v)
        return df
    

class TxtProcessor:
    def __init__(self, pdf_file, embedding_model):
        self.pdf_file = pdf_file
        self.embedding_model = embedding_model
        
    def read_pdf(self):
        # with open(self.pdf_file, 'rb') as file:
        #     pdf_reader = PyPDF2.PdfReader(file)
        
        with self.pdf_file as file:
            pdf_reader = PyPDF2.PdfReader(file)
        
            
            pdf_content = ""
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text_page = page.extract_text()
                pdf_content += text_page
            return pdf_content
        
    def split_into_batches(self, text, words_per_batch=100):
        words = text.split()
        batches = []

        for i in range(0, len(words), words_per_batch):
            batch = words[i:i + words_per_batch]
            batches.append(' '.join(batch))

        return batches
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

    def split_into_batches2(self, text):
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



