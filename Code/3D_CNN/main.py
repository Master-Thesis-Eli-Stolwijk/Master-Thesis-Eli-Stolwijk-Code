import numpy as np 
import os
import torch_models
import torch
import pickle
import datetime
from torch.utils.data import DataLoader
import pandas as pd
import random
import sys

sys.path.insert(0, '/Fridge/users/eli/Code/3D_CNN')
from data_loaders import julia_loader

sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util

from loss_functions import Vanilla_Loss
from loss_functions import PLD_Loss
from loss_functions import MPD_Loss
from loss_functions import per_frame_loss


def train_model(model, train_loader, val_loader, bottle_neck_size, drop_out_rate):

    """
    Trains the model on the train set, validates on the validation set and picks the best performing model 
    on the validation set during any epoch.
    """    

    model.to(device)
    
    train_loss = []
    train_vanilla_loss = []
    train_custom_loss = []
    val_losses = []
    val_vanilla_losses = []
    val_custom_losses = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    
    lowest_val_loss = 10000000
    best_epoch = 0   
    
    for epoch in range(0, n_epochs):
        
        ################################TRAINING################################
        
        model = model.train()
        train_iter = iter(train_loader)
        torch_loss_epoch_total = 0
        vanilla_loss_epoch_total = 0
        custom_loss_epoch_total = 0
        val_torch_loss_epoch_total = 0
        val_vanilla_loss_epoch_total = 0
        val_custom_loss_epoch_total = 0
        
        counter = 0
                       
        for u in range(0, len(train_iter)):
            
            counter += 1
            
            og_train_batch, indices, labels, char_ohe_batch, phoneme_ohe_batch = next(train_iter)
            train_batch = torch.stack(og_train_batch)
            train_batch = train_batch.unsqueeze(1)
            phoneme_ohe_batch = torch.stack(phoneme_ohe_batch)
            
            
            reconstructed, representations = model(train_batch.to(device).to(dtype=torch.float), phoneme_ohe_batch.to(device))
            
            torch_loss, vanilla, custom = loss_function(reconstructed, train_batch.to(device).to(dtype=torch.float), labels, representations)
            
            torch_loss_epoch_total += torch_loss.item()
            vanilla_loss_epoch_total += vanilla.item()
            custom_loss_epoch_total += custom.item()
            
            loss = torch_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        
        ###############################Validating#################################
        
        params_before = Util.get_model_params(model.parameters())
    
        with torch.no_grad():
            
            model = model.eval()
            
            val_iter = iter(val_loader)
                                
            for v in range(0, len(val_iter)):
                
                og_val_batch, val_indices, val_labels, val_char_ohe_batch, val_phoneme_ohe_batch = next(val_iter)
                val_batch = torch.stack(og_val_batch)
                val_batch = val_batch.unsqueeze(1)
                val_phoneme_ohe_batch = torch.stack(val_phoneme_ohe_batch)
                
                output,val_representations = model(val_batch.to(device).to(dtype=torch.float), val_phoneme_ohe_batch.to(device))
                
                val_torch_loss, val_vanilla, val_custom = loss_function(output, val_batch.to(device).to(dtype=torch.float), val_labels, val_representations)
                
                val_torch_loss_epoch_total += val_torch_loss.item()
                val_vanilla_loss_epoch_total += val_vanilla.item()
                val_custom_loss_epoch_total += val_custom.item()
                    
            params = Util.get_model_params(model.parameters())
            
            if sum(filter(None, params)) == 0:
                
                raise Exception("All weights have converged to 0's. Training was aborted")
               
        current_loss = val_torch_loss_epoch_total / len(val_loader)
        
        val_losses.append(current_loss)
        val_vanilla_losses.append(val_vanilla_loss_epoch_total/len(val_loader))
        val_custom_losses.append(val_custom_loss_epoch_total/len(val_loader))
        params_after = Util.get_model_params(model.parameters())
        
        if params_before != params_after:
            raise Exception("Model parameters changed during testing")
        
        train_loss.append(torch_loss_epoch_total / len(train_loader))
        train_vanilla_loss.append(vanilla_loss_epoch_total/ len(train_loader))
        train_custom_loss.append(custom_loss_epoch_total/ len(train_loader))
        
        print("Finished epoch " + str(epoch + 1), " out of " + str(n_epochs) + ". Train loss = " + 
              str(torch_loss_epoch_total / len(train_loader)) + "; Validation loss = " + str(current_loss))
        
        
        
        if current_loss < lowest_val_loss:
            
            lowest_val_loss = current_loss
            
            with open('/Fridge/users/eli/Code/3D_CNN/files/temp/best_model', "wb") as fp:   #Pickling
                pickle.dump(model, fp)
                
            best_epoch = epoch
            
        
            
    with open('/Fridge/users/eli/Code/3D_CNN/files/temp/best_model', "rb") as fp:   # Unpickling
        best_model = pickle.load(fp)
    
    return best_model, train_loss, val_losses, optimizer, best_epoch

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
    
    
    FrameLoss = per_frame_loss()
    
    with torch.no_grad():
            
        model = model.eval()
        
        test_iter = iter(test_loader)
        test_loss_total = 0
        
        for v in range(0, len(test_iter)):
            
            og_test_batch, test_indices, test_labels, test_char_ohe_batch, test_phoneme_ohe_batch = next(test_iter)
            test_batch = torch.stack(og_test_batch)
            test_batch = test_batch.unsqueeze(1)
            test_char_ohe_batch = torch.stack(test_char_ohe_batch)
            test_phoneme_ohe_batch = torch.stack(test_phoneme_ohe_batch)
            
            output, test_representations = model(test_batch.to(device).to(dtype=torch.float), test_phoneme_ohe_batch.to(device))
            
            FrameLoss.update(output.cpu(), og_test_batch)
            test_torch_loss,_,_ = loss_function(output, test_batch.to(device).to(dtype=torch.float), test_labels, test_representations)
            test_loss_total += test_torch_loss
                
               
        test_loss = test_loss_total / len(test_iter)
        
        print("Test loss was: " + str(test_loss))
        print("Average loss per frame:")
        print(FrameLoss.get_average_loss_per_frame())
        
        return test_loss

def train_test_log(model):
        
    """
    method that calls all needed functions to train, test, log and save the model
    """

    model, train_loss, val_loss, optimizer, epoch = train_model(train_loader, val_loader)
                
    test_loss  = test_model(model, test_loader, loss_function)            
    
    save_and_log_model(model, lv_weight, epoch, test_loss, train_loss, val_loss, optimizer)
    
    
def save_and_log_model(model, lv_weight, epoch, test_loss, train_loss, val_loss, optimizer):
    
    """
    Method that generates a visual example of one reconstruction and then writes it with all other usefull information
    to an excell file. Also saves the model (pickled)
    """
    
    
    now = datetime.datetime.now()
    
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    
    #####################Showing reconstruction of individual example#################
    
    ind_iter = iter(ind_loader)
            
    for u in range(0, len(ind_iter)):
    
        ind_batch, ind_indices, _, ind_char_ohe_batch, ind_phoneme_ohe_batch = next(ind_iter)
        ind_batch = torch.stack(ind_batch)
        ind_batch = ind_batch.unsqueeze(1)
        ind_phoneme_ohe_batch = torch.stack(ind_phoneme_ohe_batch)
    
    with torch.no_grad():
            
        model = model.eval()
        
        output = model(ind_batch.to(device).to(dtype=torch.float), ind_phoneme_ohe_batch.to(device))[0]
    
    ####################################################################################
    
    with open("/Fridge/users/eli/Code/3D_CNN/files/pickled_files/models/" + model.name + "(lvw=" + str(lv_weight) + ")_geq_" + str(min_frame) + "_leq_" + str(max_frame) + "_lr=" + str(learning_rate) + ",e=" + str(epoch + 1) + ",b=" + str(batch_size) + "_"+ str(participant) + "_" + str(dt_string), "wb") as fp:   #Pickling
        pickle.dump(model, fp)
        print("Saved model")
        
    print("Saving logs...")
    
    extra = pd.DataFrame([['Custom loss weight', lv_weight], ['Best epoch', epoch], ['Bottleneck size', bottle_neck_size]])
    
    Util.log_training(model.bottleneck_size, ind_batch.squeeze().squeeze().cpu().detach().numpy(), output.squeeze().squeeze().cpu().detach().numpy(), train_loss, val_loss, test_loss, model.name, batch_size, participant, learning_rate, weight_decay, (epoch + 1), optimizer, loss_function, loader.name, dt_string, extra_info=extra)    


set_seed(5)

participants = ['F1']
    
lev_weights = [3]

drop_out_rate = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for u in range(0, len(participants)):
    for v in range(0, len(lev_weights)):
        participant = participants[u]
        bottle_neck_size = 100
        lv_weight = lev_weights[v]

        save_model = True
        min_frame = 5
        max_frame = 20
        seq_len = 20
        batch_size = 10
        shape = [48, 44]
        n_epochs = 100
        learning_rate = 0.001
        weight_decay = 1e-8
        loss_function = PLD_Loss(lv_weight, device)
        
        parent_dir = os.path.join("/Fridge/users/eli/Data_OG", participant)
        directory_video = participant + "_video"
        video_path = os.path.join(parent_dir, directory_video)
        
        loader = julia_loader(participant, batch_size, min_frame, max_frame, 0.9)
        
        _, _, _, _, all_data = loader.load_data(video_path)
        
        most_characters, most_phonemes = Util.get_most_char_and_phonemes(all_data)
        
        train_loader, test_loader, val_loader, ind_loader, _ = loader.get_loaders(video_path, 20, most_characters, most_phonemes)
        
        model = torch_models.CNN_AE_P01(bottle_neck_size, drop_out_rate)
        
        train_test_log(model)

