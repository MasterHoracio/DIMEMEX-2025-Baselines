import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch

class tokenhandler():
    def __init__(self, PRE_TRAINED_MODEL, MAX_LENGTH):
        super(tokenhandler, self).__init__()
        self.pre_trained_model = PRE_TRAINED_MODEL
        self.max_length = MAX_LENGTH

        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL, do_lower_case=True)  # BertTokenizer, DistilBertTokenizer

    def tokenize_dataset(self, sentences):
        input_ids = []
        token_type_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,  # Instancia a codificar
                add_special_tokens=True,  # Añadir '[CLS]' y '[SEP]'
                max_length=self.max_length,  # Truncar todas las instancias
                truncation=True,
                pad_to_max_length=True,
                padding='max_length',  # Añadir padding
                return_attention_mask=True,  # Construir las máscaras de atención.
                return_tensors='pt')
            
            # Add the encoded instance to the list
            input_ids.append(encoded_dict['input_ids'])
            token_type_ids.append(encoded_dict["token_type_ids"])
            # Add the attention mask
            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        
        return input_ids, token_type_ids, attention_masks

    def tensor_the_dataset_bert(self, input_ids, input_token_ids, attention_masks, labels):
        new_labels = torch.tensor(labels, dtype=torch.float)
        tensor_dataset = TensorDataset(input_ids, input_token_ids, attention_masks, new_labels)
        return tensor_dataset
    
    
    def create_dataloader_random(self, dataset, BATCH_SIZE):
        ds_dataloader = DataLoader(
            dataset,  # training instances
            sampler = RandomSampler(dataset), # Pull out batches randomly
            batch_size = BATCH_SIZE # train with this batch size.
        )
        return ds_dataloader
    
    def create_dataloader_sequential(self, dataset, BATCH_SIZE):
        ds_dataloader = DataLoader(
            dataset,  # training instances
            sampler = SequentialSampler(dataset), # Pull out batches randomly
            batch_size = BATCH_SIZE # train with this batch size.
        )
        return ds_dataloader