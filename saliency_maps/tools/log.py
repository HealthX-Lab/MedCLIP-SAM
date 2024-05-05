import logging
import os
from datetime import datetime

class InfoLogger:
    def __init__(self, log_file_name, log_path, log_level=logging.INFO):
        self.log_file_name = log_file_name
        self.log_level = log_level
        self.log_path = log_path
        os.makedirs(self.log_path, exist_ok=True)
        logging.basicConfig(filename=f'{log_path}/{log_file_name}', level=log_level, format='%(asctime)s %(message)s')

    @staticmethod
    def create_log_file_name(key_words, timestamp=True):
        timenow = str(datetime.now())
        return f'{key_words}_{timenow}.txt' if timestamp else f'{key_words}.txt'

    def add_info(self, message):
        logging.info(message)