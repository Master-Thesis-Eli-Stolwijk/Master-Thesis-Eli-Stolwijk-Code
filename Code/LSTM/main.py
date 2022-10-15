import numpy as np 
import os
import torch
import pickle
import datetime
from torch.utils.data import DataLoader
import random
import sys 

sys.path.insert(0, '/Fridge/users/eli/Code/LSTM')
from data_loaders import LSTM_loader
import models

sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util


def train_model(train_loader, val_loader):

    model = models.ST_AutoEncoder_L1(1)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    train_loss = 0
    train_losses = []
    val_losses = []
    best_epoch = 0   
    lowest_val_loss = 100000
    
    for epoch in range(0, n_epochs):
        
        ################################TRAINING################################
        
        model = model.train()
        train_iter = iter(train_loader)
        train_epoch_loss = 0
        val_epoch_loss = 0
                       
        for u in range(0, len(train_iter)):
            
            train_batch, train_labels = next(train_iter)
            
            train_batch = torch.stack(train_batch)
            train_batch = train_batch.unsqueeze(1)
            
            reconstructed,_ = model(train_batch.to(device).to(dtype=torch.float))
            
            train_batch_loss = loss_function(reconstructed, train_batch.to(device).to(dtype=torch.float))
            train_epoch_loss += train_batch_loss.item()
            
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()
    
        
        ###############################Validating#################################
        
        params_before = Util.get_model_params(model.parameters())
    
        with torch.no_grad():
            
            model = model.eval()
            
            val_iter = iter(val_loader)
            
            for v in range(0, len(val_iter)):
                
                val_batch, val_labels = next(val_iter)
                val_batch = torch.stack(val_batch)
                val_batch = val_batch.unsqueeze(1)
                
                
                reconstructed,_ = model(val_batch.to(device).to(dtype=torch.float))
                
                val_batch_loss = loss_function(reconstructed, val_batch.to(device).to(dtype=torch.float))
                val_epoch_loss += val_batch_loss.item()
                    
            params = Util.get_model_params(model.parameters())
            
            if sum(filter(None, params)) == 0:
                
                raise Exception("All weights have converged to 0's. Training was aborted")
               
        current_loss = val_epoch_loss / len(val_loader)
        
        val_losses.append(current_loss)
        
        params_after = Util.get_model_params(model.parameters())
        
        if params_before != params_after:
            raise Exception("Model parameters changed during testing")
        
        train_losses.append(train_epoch_loss / len(train_loader))
        
        print("Finished epoch " + str(epoch + 1), " out of " + str(n_epochs) + ". Train loss = " + 
              str(train_epoch_loss / len(train_loader)) + "; Validation loss = " + str(current_loss))
    
    
        if current_loss < lowest_val_loss:
            
            lowest_val_loss = current_loss
            
            with open('/Fridge/users/eli/Code/LSTM/files/temp/best_model', "wb") as fp:   #Pickling
                pickle.dump(model, fp)
                
            best_epoch = epoch + 1
            
        
            
    with open('/Fridge/users/eli/Code/LSTM/files/temp/best_model', "rb") as fp:   # Unpickling
        best_model = pickle.load(fp)
    
    return model, train_losses, val_losses, optimizer, best_epoch

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)      



def test_model(model, test_loader, loss_function):
    
    test_representations = []
    
    with torch.no_grad():
            
        model = model.eval()
        
        test_iter = iter(test_loader)
        test_loss_total = 0
        test_loss_total = 0
        for v in range(0, len(test_iter)):
            
            test_batch, test_labels = next(test_iter)
            test_batch = torch.stack(test_batch)
            test_batch = test_batch.unsqueeze(1)
            
            reconstructed,test_representation = model(test_batch.to(device).to(dtype=torch.float))
            
            test_representations.append(test_representation)
            test_batch_loss = loss_function(reconstructed, test_batch.to(device).to(dtype=torch.float))
            test_loss_total += test_batch_loss.item()
                
               
        test_loss = test_loss_total / len(test_iter)
        
        print("Test loss was: " + str(test_loss))
        
        return test_loss

def train_test_log():
        
    model, train_losses, val_losses, optimizer, best_epoch = train_model(train_loader, val_loader)
                
    test_loss = test_model(model, test_loader, loss_function)
    
    save_and_log_model(model, train_losses, val_losses, test_loss, optimizer, best_epoch)
        
        
def save_and_log_model(model, train_losses, val_losses, test_loss, optimizer, best_epoch):
          
    now = datetime.datetime.now()
    
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    
    #####################Showing reconstruction of individual example#################
    
    ind_iter = iter(ind_loader)
            
    for u in range(0, len(ind_iter)):
    
        ind_batch, ind_label = next(ind_iter)
        ind_batch = torch.stack(ind_batch)
        ind_batch = ind_batch.unsqueeze(1)
    
    with torch.no_grad():
            
        model = model.eval()
        
        reconstructed = model(ind_batch.to(device).to(dtype=torch.float))[0]
    
    ####################################################################################
    
    with open("/Fridge/users/eli/Code/LSTM/files/pickled_files/models/" + model.name + "_lr=" + str(learning_rate) + ",e=" + str(best_epoch) + ",b=" + str(batch_size) + "_"+ str(participant) + "_" + str(dt_string), "wb") as fp:   #Pickling
        pickle.dump(model, fp)
        print("Saved model")
        
    print("Saving logs...")
    Util.log_training(ind_batch.squeeze().cpu().detach().numpy(), reconstructed.squeeze().cpu().detach().numpy(), train_losses, val_losses, test_loss, model.name, batch_size, participant, learning_rate, weight_decay, best_epoch, optimizer, loss_function, loader.name, dt_string)    


set_seed(0)

participant = 'F1'

batch_size = 1
shape = [47, 47]
n_epochs = 50
learning_rate = 0.001
weight_decay = 1e-8
loss_function = torch.nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parent_dir = os.path.join("/Fridge/users/eli/Data_OG", participant)
directory_video = participant + "_video"
video_path = os.path.join(parent_dir, directory_video)

loader = LSTM_loader(participant, batch_size, 0.9, shape)

train_loader, test_loader, val_loader, ind_loader, eval_loader = loader.get_loaders(video_path)
train_test_log()

