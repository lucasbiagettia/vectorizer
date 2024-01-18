import json
import csv
import tqdm

class DataProcessor:
    def __init__(self, csv_file, column_names, embedding_model):
        self.csv_file = csv_file
        self.column_names = column_names
        self.embedding_model = embedding_model


    
    def split_into_batches(self, text, words_per_batch=50):
        embeds =[]
        words = text.split()
        for i in range(0, len(words), words_per_batch):
            batch = " ".join(words[i:i+words_per_batch])
            embeds.append(batch)
        return embeds

        
    def add_field_to_json(self, json_str, value):
        json_data = json.loads(json_str)
        json_data['chunk'] = value
        del json_data['link']
        del json_data['author']
        del json_data['metadata']
        return json.dumps(json_data).replace('"', "'")

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
            chunks = self.split_into_batches(text)
            for chunk in chunks:
                result_list.append((self.add_field_to_json(metadata, chunk), self.get_embedding(chunk)))

        return result_list



