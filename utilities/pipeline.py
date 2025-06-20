# Avoid warnings from libraries
import warnings
warnings.filterwarnings('ignore')

from transformers import ViTImageProcessor
from transformers import AutoImageProcessor, ViTModel
from PIL import Image

from transformers import get_linear_schedule_with_warmup, logging
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from utilities import texthandler
from utilities import tokenhandler
from utilities import architectures
from utilities import trainhandler
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import metrics
import torch
import random
import copy
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.set_verbosity_error() # Remove warnings from the pre-trained models

class pipeline():
    def __init__(self, evaluation_type, task, architecture_mode, PRE_TRAINED_MODEL_NAME_TEXT, PRE_TRAINED_MODEL_NAME_VISION, MAX_LENGTH_TEXT, encoding_dimension, save_labels = False):
        super(pipeline, self).__init__()
        self.random_seed                    = self.get_random_seed_through_os()
        self.task                           = task
        self.PRE_TRAINED_MODEL_NAME_TEXT    = PRE_TRAINED_MODEL_NAME_TEXT
        self.PRE_TRAINED_MODEL_NAME_VISION  = PRE_TRAINED_MODEL_NAME_VISION
        self.architecture_mode              = architecture_mode
        self.MAX_LENGTH_TEXT                = MAX_LENGTH_TEXT
        self.encoding_dimension             = encoding_dimension
        self.save_labels                    = save_labels
        self.evaluation_type                = evaluation_type
        
        print("***********************************************************************")
        print("Training on Task: " + str(self.task))
        print("Using the proposed approach: " + str(self.architecture_mode))
        print("Getting the predictions from the " + str(self.evaluation_type) + " dataset")
        print("Random seed: " + str(self.random_seed))
        print("***********************************************************************")

    def get_random_seed_through_os(self):
        RAND_SIZE = 3
        random_data = os.urandom(RAND_SIZE)
        random_seed = int.from_bytes(random_data, byteorder="big")
        return random_seed

    def get_hyperparameters(self):
        path_text = "dataset/"
        path_img = "dataset/img/"
        if self.task == 1:
            if self.architecture_mode == "BETO":
                EPOCHS = 3 # Define the number of epochs
                LEARNING_RATE = 2e-5 # 3 Define the learning rate
            elif self.architecture_mode == "ViT":
                EPOCHS = 3 # Define the number of epochs
                LEARNING_RATE = 3e-5 # 3 Define the learning rate
            elif self.architecture_mode == "EF":
                EPOCHS = 3 # Define the number of epochs
                LEARNING_RATE = 2e-5 # 3 Define the learning rate
            BATCH_SIZE = 16
            N_CLASSES = 3 # Define the number of classes
            labels = [0, 1, 2]
            target_names = ['ninguno', 'contenido_inapropiado', 'discurso_odio']
        elif self.task == 2:
            if self.architecture_mode == "BETO":
                EPOCHS = 3 # Define the number of epochs
                LEARNING_RATE = 2e-5 # 3 Define the learning rate
            elif self.architecture_mode == "ViT":
                EPOCHS = 3 # Define the number of epochs
                LEARNING_RATE = 2e-5 # 3 Define the learning rate
            elif self.architecture_mode == "EF":
                EPOCHS = 3 # Define the number of epochs
                LEARNING_RATE = 2e-5 # 3 Define the learning rate
            BATCH_SIZE = 24
            N_CLASSES = 6
            labels = [0, 1, 2, 3, 4, 5]
            target_names = ['ninguno', 'contenido_inapropiado', 'sexismo', 'racismo', 'clasismo', 'otro']
        return path_text, path_img, labels, target_names, N_CLASSES, EPOCHS, BATCH_SIZE, LEARNING_RATE
    
    def get_full_dataset(self, path):
        if self.task == 1:
            task_id = 1
        elif self.task == 2:
            task_id = 2

        #Load train and text dataset
        datahandler_text = texthandler.DataHandler(self.random_seed)
        dataset = datahandler_text.load_dataset(path, task_id)
        x_train_text, y_train, x_test_text, labels_train = datahandler_text.preprocess_dataset(dataset, task_id, "text")
        
        x_train_img = []
        for i in range(len(dataset[0])):#len(data_train)
            x_train_img.append(dataset[0][i]['MEME-ID'])
        
        x_test_img = []
        for i in range(len(dataset[1])):
            x_test_img.append(dataset[1][i]['MEME-ID'])
        
        return x_train_text, x_train_img, y_train, x_test_text, x_test_img, labels_train

    def get_class_weights(self, labels_train):
        labels_train = np.array(labels_train)

        # Calcular las frecuencias de cada clase
        class_counts = np.bincount(labels_train)  # Cuenta ocurrencias de cada clase
        #class_weights = 1.0 / class_counts   # Peso inversamente proporcional a la frecuencia
        class_weights = class_counts / sum(class_counts)
        class_weights = 1.0/class_weights
        
        return class_weights

    def get_image_tensors(self, path_img, x_train, x_test):
        image_processor = AutoImageProcessor.from_pretrained(self.PRE_TRAINED_MODEL_NAME_VISION, use_fast=True)
        
        print("***********************************************************************")
        print("Loading training images.")
        print("***********************************************************************")
        image_tensors_train = []
        for i in tqdm(range(len(x_train))):
            name = x_train[i]
            image_train = Image.open(path_img + "train/" + name)#load image
            image_train = image_train.convert('RGB')#convert image into RGB
            aux_img = image_processor(image_train, return_tensors="pt")#preprocess image according to ViT
            image_tensors_train.append(aux_img["pixel_values"])

        print("***********************************************************************")
        print("Loading testing images.")
        print("***********************************************************************")
        image_tensors_test = []
        for i in tqdm(range(len(x_test))):
            name = x_test[i]
            image_test = Image.open(path_img + "test/" + name)#load image
            image_test = image_test.convert('RGB')#convert image into RGB
            aux_img = image_processor(image_test, return_tensors="pt")#preprocess image according to ViT
            image_tensors_test.append(aux_img["pixel_values"])
        
        return image_tensors_train, image_tensors_test
    
    def tokenize_full_dataset(self, x_train_text, img_tensors_train, y_train, x_test_text, img_tensors_test, labels_train, BATCH_SIZE):
        train_labels = torch.tensor(y_train, dtype=torch.float)
        if self.architecture_mode == "ViT" or self.architecture_mode == "EF":
            img_tensors_train = torch.stack(img_tensors_train)
            img_tensors_test = torch.stack(img_tensors_test)

        if self.architecture_mode == "BETO" or self.architecture_mode == "EF":
            # Create tokenizer instance for the text on the memes
            tokenizer = tokenhandler.tokenhandler(self.PRE_TRAINED_MODEL_NAME_TEXT, self.MAX_LENGTH_TEXT)
            
            # Tokenize all the dataset (for the text on the memes)
            input_ids_train, input_token_ids_train, attention_masks_train = tokenizer.tokenize_dataset(x_train_text)
            input_ids_test, input_token_ids_test, attention_masks_test = tokenizer.tokenize_dataset(x_test_text)

        if self.architecture_mode == "EF":        
            # Combine the training and validation inputs to a single tensor
            train_dataset = TensorDataset(input_ids_train, input_token_ids_train, attention_masks_train, img_tensors_train, train_labels)
            test_dataset = TensorDataset(input_ids_test, input_token_ids_test, attention_masks_test, img_tensors_test)
        elif self.architecture_mode == "BETO":
            # Combine the training and validation inputs to a single tensor
            train_dataset = TensorDataset(input_ids_train, input_token_ids_train, attention_masks_train, train_labels)
            test_dataset = TensorDataset(input_ids_test, input_token_ids_test, attention_masks_test)
        elif self.architecture_mode == "ViT":
            # Combine the training and validation inputs to a single tensor
            train_dataset = TensorDataset(img_tensors_train, train_labels)
            test_dataset = TensorDataset(img_tensors_test)
            
        class_weights = self.get_class_weights(labels_train)
        weights = class_weights[labels_train]
        train_dataloader = DataLoader(
            train_dataset,  # training instances
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
            batch_size = BATCH_SIZE # train with this batch size.
        )
        
        test_dataloader = DataLoader(
                test_dataset, # validation instances
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = BATCH_SIZE # Evaluate with this batch size.
            )
        
        return train_dataloader, test_dataloader
    
    def load_model(self, N_CLASSES):
        # Assign the training device
        device      = torch.device("cuda:3")  # Select GPU 4
        torch.cuda.set_device(device)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.architecture_mode == "EF":
            model = architectures.EarlyFusion(self.encoding_dimension, N_CLASSES, self.PRE_TRAINED_MODEL_NAME_VISION, self.PRE_TRAINED_MODEL_NAME_TEXT)
        elif self.architecture_mode == "BETO":
            model = architectures.BETO(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME_TEXT, N_CLASSES)
        elif self.architecture_mode == "ViT":
            model = architectures.ViT(self.encoding_dimension, self.PRE_TRAINED_MODEL_NAME_VISION, N_CLASSES)

        n_params = sum(p.numel() for p in model.parameters())

        return model, n_params, device

    def train_model(self, model, device, train_dataloader, epochs, LEARNING_RATE):
        # Load the model into the GPU
        model.to(device)

        # Define the optimization algorithm
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

        # Define the number of training steps (epochs * number of batches)
        total_steps = len(train_dataloader) * epochs

        # Create a schedule for the LR update
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create trainhabnlder instance
        t_handler = trainhandler.trainhandler(device, scheduler)

        # Training cicle
        best_f1_score = 0
        for epoch in range(epochs):
            if self.architecture_mode == "EF":
                t_handler.train_ef(epoch + 1, epochs, model, train_dataloader, optimizer, criterion)
            elif self.architecture_mode == "BETO":
                t_handler.train_beto(epoch + 1, epochs, model, train_dataloader, optimizer, criterion)
            elif self.architecture_mode == "ViT":
                t_handler.train_vit(epoch + 1, epochs, model, train_dataloader, optimizer, criterion)
        
        return t_handler

    def evaluate(self, t_handler, model, test_dataloader, N_CLASSES):
        if self.architecture_mode == "EF":
            predictions = t_handler.evaluate_ef(model, test_dataloader)
        elif self.architecture_mode == "BETO":
            predictions = t_handler.evaluate_beto(model, test_dataloader)
        elif self.architecture_mode == "ViT":
            predictions = t_handler.evaluate_vit(model, test_dataloader)
        
        model_prediction_test, predicted_labels = t_handler.pred_to_label(predictions, N_CLASSES)
        
        return model_prediction_test