import json

class ConfigProvider:
    def __init__(self):
        config_file_path = 'src/config/config.json'
        self.config = self.load_config(config_file_path)

    def load_config(self, file_path):
        with open(file_path, 'r') as file:
            config_data = json.load(file)
        return config_data

    def get_db_name(self):
        return self.config.get('db_name')

    def get_default_embedding_model(self):
        return self.config.get('default_embedding_model')

    def get_default_inference_model(self):
        return self.config.get('default_inference_model')

    def get_inference_model_args(self):
        return self.config.get('inference_model_args', {})

