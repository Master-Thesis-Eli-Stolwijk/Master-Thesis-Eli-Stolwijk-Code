import numpy as np 
import os
import torch
import pickle
import datetime
import pandas as pd
import random
import sys 

sys.path.insert(0, '/Fridge/users/eli/Code/LSTM')
from data_loaders import LSTM_loader
import models

sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util
from loss_functions import Vanilla_Loss
from loss_functions import PLD_Loss
from loss_functions import MPD_Loss
from loss_functions import per_frame_loss
from loss_functions import PLD_only_Loss

def train_model(model, train_loader, val_loader):
    
    """
    Trains the model on the train set, validates on the validation set and picks the best performing model 
    on the validation set during any epoch.
    """ 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    train_loss = 0
    train_losses = []
    train_pld_losses = []
    train_mse_losses = []
    val_losses = []
    best_epoch = 0   
    lowest_val_loss = 100000
    
    for epoch in range(0, n_epochs):
        
        ################################TRAINING################################
        
        model = model.train()
        train_epoch_loss = 0
        train_epoch_mse_loss = 0
        train_epoch_pld_loss = 0
        val_epoch_loss = 0
                       
        for u in range(0, train_loader.length):
            
            train_batch, train_labels = train_loader.get_batch()
            
            train_batch = torch.stack(train_batch)
            train_batch = train_batch.unsqueeze(1)
            
            reconstructed, train_representations = model(train_batch.to(device).to(dtype=torch.float))
            
            train_batch_loss, mse_loss, pld_loss = loss_function(reconstructed, train_batch.to(device).to(dtype=torch.float), train_labels, train_representations)
                        
            train_epoch_loss += train_batch_loss.item()
            train_epoch_mse_loss += mse_loss.item()
            train_epoch_pld_loss += pld_loss.item()
            
            
            optimizer.zero_grad()
            train_batch_loss.backward(retain_graph=True)
            optimizer.step()
    
        
        ###############################Validating#################################
        
        params_before = Util.get_model_params(model.parameters())
    
        with torch.no_grad():
            
            model = model.eval()
            
            
            for v in range(0, val_loader.length):
                
                val_batch, val_labels = val_loader.get_batch()
                val_batch = torch.stack(val_batch)
                val_batch = val_batch.unsqueeze(1)
                
                
                reconstructed,val_representations = model(val_batch.to(device).to(dtype=torch.float))
                
                val_batch_loss, _, _ = loss_function(reconstructed, val_batch.to(device).to(dtype=torch.float), val_labels, val_representations)
            
                val_epoch_loss += val_batch_loss.item()
                    
            params = Util.get_model_params(model.parameters())
            
            if sum(filter(None, params)) == 0:
                
                raise Exception("All weights have converged to 0's. Training was aborted")
               
        current_loss = val_epoch_loss / val_loader.length
        
        val_losses.append(current_loss)
        
        params_after = Util.get_model_params(model.parameters())
        
        if params_before != params_after:
            raise Exception("Model parameters changed during testing")
        
        train_losses.append(train_epoch_loss / train_loader.length)
        train_mse_losses.append(train_epoch_mse_loss / train_loader.length)
        train_pld_losses.append(train_epoch_pld_loss / train_loader.length)
        
        print("Finished epoch " + str(epoch + 1), " out of " + str(n_epochs) + ". Train loss = " + 
              str(round(train_epoch_loss / train_loader.length, 2)) + ": (mse=" + str(round(train_epoch_mse_loss / train_loader.length, 2)) +
              ",pld=" + str(round(train_epoch_pld_loss / train_loader.length, 2)) + "); Validation loss = " + str(round(current_loss, 2)))
    
    
        train_loader.reset()
        val_loader.reset()
    
        if current_loss < lowest_val_loss:
            
            lowest_val_loss = current_loss
            
            with open('/Fridge/users/eli/Code/LSTM/files/temp/best_model', "wb") as fp:   #Pickling
                pickle.dump(model, fp)
                
            best_epoch = epoch + 1
            
        
            
    with open('/Fridge/users/eli/Code/LSTM/files/temp/best_model', "rb") as fp:   # Unpickling
        best_model = pickle.load(fp)
    
    return model, train_losses, val_losses, optimizer, best_epoch

def set_seed(seed):
    
    """
    Sets all random seeds to given value
    """
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)      



def test_model(model, test_loader, loss_function):
    
    """
    Tests the model on unseen data with the given loss function
    """
    
    test_representations = []
    
    with torch.no_grad():
            
        model = model.eval()
        
        test_loss_total = 0
        mse_loss_total = 0
        pld_loss_total = 0
        
        for v in range(0, test_loader.length):
            
            test_batch, test_labels = test_loader.get_batch()
            test_batch = torch.stack(test_batch)
            test_batch = test_batch.unsqueeze(1)
            
            reconstructed, test_representations = model(test_batch.to(device).to(dtype=torch.float))
            
            test_batch_loss, test_mse_loss, test_pld_loss = loss_function(reconstructed, test_batch.to(device).to(dtype=torch.float), test_labels, test_representations)
            
            test_loss_total += test_batch_loss.item()
            mse_loss_total += test_mse_loss.item()
            pld_loss_total += test_pld_loss.item()
               
        test_loss = test_loss_total / test_loader.length
        mse_loss = mse_loss_total / test_loader.length
        pld_loss = pld_loss_total / test_loader.length
        
        print("Test loss was: " + str(round(test_loss, 2)) + ": (mse=" + str(round(mse_loss, 2)) + ", pld=" + str(round(pld_loss, 2)) + ")")
        
        return test_loss, mse_loss, pld_loss

def train_test_log(model):
    
    """
    method that calls all needed functions to train, test, log and save the model
    """
        
    model, train_losses, val_losses, optimizer, best_epoch = train_model(model, train_loader, val_loader)
                
    test_loss, mse_loss, pld_loss = test_model(model, test_loader, loss_function)
    
    save_and_log_model(model, train_losses, val_losses, test_loss, mse_loss, pld_loss, optimizer, best_epoch)
        
        
def save_and_log_model(model, train_losses, val_losses, test_loss, mse_loss, pld_loss, optimizer, best_epoch):
    
    """
    Method that generates a visual example of one reconstruction and then writes it with all other usefull information
    to an excell file. Also saves the model (pickled)
    """
          
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
    
    extra_info = pd.DataFrame([['description', model.description], ['loss function', loss_function.name], 
                               ['bottleneck size', model.bottle_neck_size], ['seed', seed], ['MSE loss', mse_loss],
                               ['PLD loss', pld_loss]])
    
    with open("/Fridge/users/eli/Code/LSTM/files/pickled_files/models/" + model.name + "(" + str(model.bottle_neck_size) + ")_lr=" + str(learning_rate) + ",e=" + str(best_epoch) + ",b=" + str(batch_size) + "_"+ str(participant) + "_" + str(dt_string), "wb") as fp:   #Pickling
        pickle.dump(model, fp)
        print("Saved model")
        
    print("Saving logs...")
    Util.log_training(model.bottle_neck_size, ind_batch.squeeze().cpu().detach().numpy(), reconstructed.squeeze().cpu().detach().numpy(), train_losses, val_losses, test_loss, model.name, batch_size, participant, learning_rate, weight_decay, best_epoch, optimizer, loss_function, loader.name, dt_string)    

seed = 0
set_seed(seed)

participant = 'F1'

batch_size = 10
shape = [47, 47]
n_epochs = 300
learning_rate = 0.001
weight_decay = 1e-8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parent_dir = os.path.join("/Fridge/users/eli/Data_OG", participant)
directory_video = participant + "_video"
video_path = os.path.join(parent_dir, directory_video)

#custom_weights = [0, 0.1, 0.5, 1, 1.5, 3, 5, 10]

custom_weights = [10]


for u in range (0, len(custom_weights)):
    
    loss_function = PLD_Loss(custom_weights[u], device)
    
    loader = LSTM_loader(participant, batch_size, 0.9, shape)
    
    model = models.ST_AutoEncoder_G7(1, 16*11*11, 0)
    model.to(device)
    
    train_loader, test_loader, val_loader, ind_loader, eval_loader = loader.get_loaders(video_path)
    train_test_log(model)

