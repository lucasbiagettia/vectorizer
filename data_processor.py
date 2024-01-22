import json
import csv
import tqdm
import re


class DataProcessor:
    def __init__(self, csv_file, column_names, embedding_model):
        self.csv_file = csv_file
        self.column_names = column_names
        self.embedding_model = embedding_model


    
    def split_into_batches(self, text, words_per_batch=70):
        embeds =[]
        words = text.split()
        for i in range(0, len(words), words_per_batch):
            batch = " ".join(words[i:i+words_per_batch])
            embeds.append(batch)
        return embeds
   
    def split_into_batches_EOL(self, text):
        embeds = []
        lines = text.split('\n')
        for line in lines:
            embeds.append(line.strip()) 
        print("batches: ", len(embeds))
        return embeds
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
            header = next(csv_reader)  # Read the header

            if all(column in header for column in self.column_names):
                for i, row in enumerate(csv_reader):

                    values = [row[header.index(column)] for column in self.column_names]
                    result_list.append(tuple(values))
            else:
                raise Exception("One or more specified columns are not present in the file.")

        return result_list

    def get_embedding(self, text):
        return self.embedding_model.get_embedding(text)

 
    
    
    def data2vector2(self, data):
        result_list = []
        for metadata, text in tqdm.tqdm(data, desc="Processing data"):
            chunks = self.split_sentences(text)
            for chunk in chunks:
                result_list.append((self.add_field_to_json(metadata, chunk), self.get_embedding(chunk)))

        return result_list



