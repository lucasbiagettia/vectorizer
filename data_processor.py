import json
import csv
import tqdm

class DataProcessor:
    def __init__(self, csv_file, column_names, embedding_model):
        self.csv_file = csv_file
        self.column_names = column_names
        self.embedding_model = embedding_model

    def split_into_batches(self, text, words_per_batch=10):
        words = text.split()
        return [" ".join(words[i:i+words_per_batch]) for i in range(0, len(words), words_per_batch)]

    def add_field_to_json(self, json_str, value):
        json_data = json.loads(json_str)
        del json_data['link'], json_data['author'], json_data['metadata']
        json_data['chunk'] = value
        return json.dumps(json_data).replace('"', "'")

    def order_data(self):
        result_list = []
        with open(self.csv_file, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            if all(column in header for column in self.column_names):
                result_list.extend(tuple(row[header.index(column)] for column in self.column_names) for row in csv_reader)
            else:
                raise Exception("One or more specified columns are not present in the file.")
        return result_list

    def get_embedding(self, text):
        return self.embedding_model.get_embedding(text)

    def data2vector2(self, data):
        result_list = [
            (self.add_field_to_json(metadata, chunk), chunk)
            for metadata, text in tqdm.tqdm(data, desc="Processing data")
            for chunk in [self.get_embedding(chunk) for chunk in self.split_into_batches(text)]
        ]
        return result_list


# Uso de la clase

