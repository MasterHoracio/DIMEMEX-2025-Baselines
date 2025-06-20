from transformers import AutoModel
from transformers import ViTModel
import torch.nn.functional as F
from torch import nn
import torch
import math
import copy

class BETO(nn.Module):
    def __init__(self, encoding_dimension, PRE_TRAINED_MODEL_NAME, n_classes):
        super(BETO, self).__init__()
        self.bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.encoding_dimension = encoding_dimension
        self.n_classes = n_classes

        self.drop_out = nn.Dropout(p = 0.05)
        self.fc1 = nn.Linear(40, 1)
        
        
        self.classifier = nn.Sequential(\
                            nn.Linear(self.encoding_dimension, self.n_classes))

    def forward(self, ids, mask, token_type_ids):
        outputs =                           self.bert_model(ids,
                                                           attention_mask = mask,
                                                           token_type_ids = token_type_ids,
                                                           return_dict = True,
                                                           output_hidden_states = True,
                                                           output_attentions = False)
        last_hidden_states  = outputs.hidden_states[-1]
        
        cls_tensor = self.fc1(last_hidden_states.permute(0, 2, 1)).squeeze(-1)
        cls_drop = self.drop_out(cls_tensor)
        out = self.classifier(cls_drop)

        return out
    
class ViT(nn.Module):
    def __init__(self, encoding_dimension, PRE_TRAINED_MODEL_NAME, n_classes):
        super(ViT, self).__init__()
        self.vit_model = ViTModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        self.encoding_dimension = encoding_dimension
        self.n_classes = n_classes

        self.drop_out = nn.Dropout(p = 0.05)
        self.fc1 = nn.Linear(197, 1)

        self.classifier = nn.Sequential(\
                            nn.Linear(self.encoding_dimension, self.n_classes))
    
    def forward(self, _input):
        _output = self.vit_model(_input)
        last_hidden_states = _output.last_hidden_state
        
        cls_tensor = self.fc1(last_hidden_states.permute(0, 2, 1)).squeeze(-1)
        cls_drop = self.drop_out(cls_tensor)
        
        out = self.classifier(cls_drop)
        
        return out
    
class EarlyFusion(nn.Module):
    def __init__(self, encoding_dimension, n_classes, PRE_TRAINED_MODEL_NAME_VISION, PRE_TRAINED_MODEL_NAME_TEXT):
        super(EarlyFusion, self).__init__()
        self.vit_model = ViTModel.from_pretrained(PRE_TRAINED_MODEL_NAME_VISION)
        self.bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME_TEXT)
        
        self.encoding_dimension = encoding_dimension
        self.n_classes = n_classes
        
        self.drop_out = nn.Dropout(p = 0.05)
        
        self.fc1 = nn.Linear(197, 1)
        self.fc2 = nn.Linear(40, 1)
        
        self.layer_norm = nn.LayerNorm(self.encoding_dimension*2)
        
        self.classifier = nn.Sequential(\
                            nn.Linear(self.encoding_dimension * 2, self.n_classes))
    
    def forward(self, ids, mask, token_type_ids, _input):
        _output = self.vit_model(_input)
        last_hidden_states = _output.last_hidden_state
        cls_tensor = self.fc1(last_hidden_states.permute(0, 2, 1)).squeeze(-1)
        hidden_state_vit = self.drop_out(cls_tensor)
        
        last_hidden_states, pooler_output = self.bert_model(ids, 
                                                        attention_mask = mask, 
                                                        token_type_ids = token_type_ids, 
                                                        return_dict = False, 
                                                        output_hidden_states = False,
                                                        output_attentions = False)
        
        cls_tensor = self.fc2(last_hidden_states.permute(0, 2, 1)).squeeze(-1)
        hidden_state_beto = self.drop_out(cls_tensor)

        concat_output = torch.cat((hidden_state_vit, hidden_state_beto), 1)
        
        norm_output = self.layer_norm(concat_output)
        
        out = self.classifier(norm_output)
        
        return out