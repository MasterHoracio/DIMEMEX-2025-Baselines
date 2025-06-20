from sklearn.model_selection import train_test_split
import json
import random
import re
import pandas as pd

class DataHandler:
    def __init__(self, random_seed, unique=False, validation=False):
        super(DataHandler, self).__init__()
        self.random_seed = random_seed
        self.validation = validation
        self.unique = unique

        random.seed(self.random_seed)
    
    def load_json(self, path_dataset, file):
        with open(path_dataset + file, 'r') as file:
            dataset = json.load(file)
        return dataset

    def load_dataset(self, path_dataset, task, evaluation_type = "test"):
        # Loads the JSON files from the datasets
        if task == 1:
            train = self.load_json(path_dataset, "train_dataset_task1.json")
            if evaluation_type == "test":
                test = self.load_json(path_dataset, "test_dataset_task1.json")
            elif evaluation_type == "validation":
                test = self.load_json(path_dataset, "validation_dataset_task1.json")
        elif task == 2:
            train = self.load_json(path_dataset, "train_dataset_task2.json")
            if evaluation_type == "test":
                test = self.load_json(path_dataset, "test_dataset_task2.json")
            elif evaluation_type == "validation":
                test = self.load_json(path_dataset, "validation_dataset_task2.json")
        return [train, test]

    def clean_sentence(self, sentence):
        # Convert instance to string
        sentence = str(sentence)

        # All text to lowercase
        sentence = sentence.lower()

        # Normalize users and URLs
        sentence = re.sub(r'\@\w+', '@user', sentence)
        sentence = re.sub(r"http\S+|www\S+|https\S+", 'url', sentence, flags=re.MULTILINE)
        
        # Separate special characters
        sentence = re.sub(r":", " : ", sentence)
        sentence = re.sub(r",", " , ", sentence)
        sentence = re.sub(r"\.", " . ", sentence)
        sentence = re.sub(r"!", " ! ", sentence)
        sentence = re.sub(r"¡", " ¡ ", sentence)
        sentence = re.sub(r"“", " “ ", sentence)
        sentence = re.sub(r"”", " ” ", sentence)
        sentence = re.sub(r"\(", " ( ", sentence)
        sentence = re.sub(r"\)", " ) ", sentence)
        sentence = re.sub(r"\?", " ? ", sentence)
        sentence = re.sub(r"¿", " ¿ ", sentence)

        # Substituting multiple spaces with a single space
        sentence = re.sub(r'\s+', ' ', sentence, flags=re.I).strip()

        return sentence

    def clean_dataset(self, dataset, task, type_text):
        x_dataset = []

        for i in range(len(dataset)):
            if type_text == "text":
                sentence = self.clean_sentence(dataset[i]['text'])
            elif type_text == "description":
                sentence = self.clean_sentence(dataset[i]['description'])
            
            x_dataset.append(sentence)
        
        return x_dataset

    def load_labels(self, path):
        df = pd.read_csv(path)
        df = df.iloc[:, 1:]
        list_labels = df.values.tolist()
        indices = [row.index(1) for row in list_labels]
        return list_labels, indices
    
    def preprocess_dataset(self, dataset, task, type_text):
        x_train = self.clean_dataset(dataset[0], task, type_text)
        x_test = self.clean_dataset(dataset[1], task, type_text)
        if task == 1:
            y_train, labels_train = self.load_labels("dataset/labels/train_dataset_task1.csv")
        elif task == 2:
            y_train, labels_train = self.load_labels("dataset/labels/train_dataset_task2.csv")
        return x_train, y_train, x_test, labels_train