import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch

class trainhandler():
    def __init__(self, device, scheduler):
        self.device = device
        self.scheduler = scheduler
        super(trainhandler, self).__init__()
    
    #test baseline txt function
    def evaluate_beto(self, model, dataloader):
        model_prediction = []
    
        model.eval()
    
        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_token_ids = batch[1].to(self.device)
                b_input_mask = batch[2].to(self.device)
    
                output = model(b_input_ids, b_input_mask, b_input_token_ids)
    
                # Move labels and predictions to the CPU
                model_prediction.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())
    
        return model_prediction

    #test baseline img function
    def evaluate_vit(self, model, dataloader):
        model_prediction = []
    
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                _input = batch[0].to(self.device)
                _input = _input[:,0,:,:,:]
                output = model(_input)
                
                # Move labels and predictions to the CPU
                model_prediction.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())
        
        return model_prediction

    def evaluate_ef(self, model, dataloader):
        model_prediction = []
    
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_token_ids = batch[1].to(self.device)
                b_input_mask = batch[2].to(self.device)
                _input = batch[3].to(self.device)
                _input = _input[:,0,:,:,:]
    
                output = model(b_input_ids, b_input_mask, b_input_token_ids, _input)
                
                # Move labels and predictions to the CPU
                model_prediction.extend(torch.sigmoid(output).cpu().detach().numpy().tolist())
        
        return model_prediction

    #this function converts all predictions into labels
    def pred_to_label(self, predictions, n_clases):
        mod_predictions = []
        labels = []
        for prediction in predictions:
            index = 0
            max_val = prediction[0]
            for i in range(n_clases):
                if prediction[i] > max_val:
                    index = i
            row = []
            labels.append(index)
            for i in range(n_clases):
                row.append(0)
            row[index] = 1
            mod_predictions.append(row)
        return mod_predictions, labels
    
    #train baseline txt function
    def train_beto(self, epoch, total_epoch, model, iterator, optimizer, criterion, clip=1):
        model.train()

        epoch_loss = 0
        epoch_acc = 0
        correct = 0
        total = 0
    
        loop = tqdm(enumerate(iterator), total=len(iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        model.train()
        for batch_index, (input_ids, input_token_ids, input_mask, labels) in loop:
            b_input_ids = input_ids.to(self.device)
            b_input_token_ids = input_token_ids.to(self.device)
            b_input_mask = input_mask.to(self.device)
            b_labels = labels.to(self.device, dtype = torch.float)
    
            # Reset the values of the gradients
            model.zero_grad()
    
            output  = model(b_input_ids, b_input_mask, b_input_token_ids)
    
    
            loss = criterion(output, b_labels)
    
            # Calculate the gradients
            loss.backward()
            # Prevent the gradient explotion
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # Update the weigths of the model
            optimizer.step()
    
            # Compute batch accuracy
            preds = output.argmax(dim=1)  # Get class with highest probability
            class_indices = torch.argmax(b_labels, dim=1)
            correct += (preds == class_indices).sum().item()
            total += class_indices.size(0)
            batch_accuracy = correct / total

            epoch_loss += loss.item()
            epoch_acc += batch_accuracy
    
            # Update progress bar
            loop.set_description(f"Epoch[{epoch}/{total_epoch}]")
            loop.set_postfix(loss = loss.item(), acc=batch_accuracy)
    
        mean_loss = epoch_loss / len(iterator)
        mean_acc = epoch_acc / len(iterator)
        
        # Update the LR
        self.scheduler.step(mean_loss)
    
        print(f'Mean loss: {mean_loss:.6f}, Mean accuracy: {mean_acc:.6f}')

    #train baseline img function
    def train_vit(self, epoch, total_epoch, model, iterator, optimizer, criterion, clip = 1):
        model.train()
        
        epoch_loss = 0
        epoch_acc = 0
        correct = 0
        total = 0
        
        loop = tqdm(enumerate(iterator), total=len(iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        model.train()
        for batch_index, (_input, labels) in loop:
            
            _input = _input.to(self.device)
            b_labels = labels.to(self.device)
    
            # Reset the values of the gradients
            model.zero_grad()
            _input = _input[:,0,:,:,:]
            output  = model(_input)
            
            loss = criterion(output, b_labels)
    
            # Calculate the gradients
            loss.backward()
            # Prevent the gradient explotion
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # Update the weigths of the model
            optimizer.step()
            
            # Compute batch accuracy
            preds = output.argmax(dim=1)  # Get class with highest probability
            class_indices = torch.argmax(b_labels, dim=1)
            correct += (preds == class_indices).sum().item()
            total += class_indices.size(0)
            batch_accuracy = correct / total

            epoch_loss += loss.item()
            epoch_acc += batch_accuracy
            
            # Update progress bar
            loop.set_description(f"Epoch[{epoch}/{total_epoch}]")
            loop.set_postfix(loss = loss.item(), acc=batch_accuracy)
        
        mean_loss = epoch_loss / len(iterator)
        mean_acc = epoch_acc / len(iterator)
        
        # Update the LR
        self.scheduler.step(mean_loss)
    
        print(f'Mean loss: {mean_loss:.6f}, Mean accuracy: {mean_acc:.6f}')

    # Define training function
    def train_ef(self, epoch, total_epoch, model, iterator, optimizer, criterion, clip=1):
        model.train()

        epoch_loss = 0
        epoch_acc = 0
        correct = 0
        total = 0
        
        loop = tqdm(enumerate(iterator), total=len(iterator), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        model.train()
        for batch_index, (input_ids, input_token_ids, input_mask, _input, labels) in loop:
            b_input_ids = input_ids.to(self.device)
            b_input_token_ids = input_token_ids.to(self.device)
            b_input_mask = input_mask.to(self.device)
            _input = _input.to(self.device)
            b_labels = labels.to(self.device)
    
            # Reset the values of the gradients
            model.zero_grad()
            _input = _input[:,0,:,:,:]
            output  = model(b_input_ids, b_input_mask, b_input_token_ids, _input)
            
            loss = criterion(output, b_labels)
    
            #l1_penalty = sum(p.abs().sum() for p in model.parameters())
            #loss += l1_penalty * 1e-8
    
            # Calculate the gradients
            loss.backward()
            # Prevent the gradient explotion
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # Update the weigths of the model
            optimizer.step()
            
            # Compute batch accuracy
            preds = output.argmax(dim=1)  # Get class with highest probability
            class_indices = torch.argmax(b_labels, dim=1)
            correct += (preds == class_indices).sum().item()
            total += class_indices.size(0)
            batch_accuracy = correct / total

            epoch_loss += loss.item()
            epoch_acc += batch_accuracy
            
            # Update progress bar
            loop.set_description(f"Epoch[{epoch}/{total_epoch}]")
            loop.set_postfix(loss = loss.item(), acc=batch_accuracy)
        
        mean_loss = epoch_loss / len(iterator)
        mean_acc = epoch_acc / len(iterator)
        
        # Update the LR
        self.scheduler.step(mean_loss)
    
        print(f'Mean loss: {mean_loss:.6f}, Mean accuracy: {mean_acc:.6f}')