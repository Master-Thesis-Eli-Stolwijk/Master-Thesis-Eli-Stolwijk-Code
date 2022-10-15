import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import torch
import tkinter as tk
from tkinter import filedialog
import pickle

def load_model_from_file():
    root = tk.Tk()
    root.withdraw()
                
    model_path = filedialog.askopenfilename()
                
    with open(model_path, "rb") as fp:   # Unpickling
        
         model = pickle.load(fp)
         
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    return model

def get_longest_video(list_videos):
    
    max_length = 0
    
    for video in list_videos:
        if len(video) > max_length:
            max_length = len(video)
    return max_length

def cut_frame(full_frame, width, heigth, upper_left_anker):
    
    anker_x = upper_left_anker[0]
    anker_y = upper_left_anker[1]
    
    new_frame = [ [0]* width for i in range(heigth)]
    
    do = 'nothing'
    
    for u in range(0, len(full_frame)):
        for v in range(0, len(full_frame)):
            if u >= anker_y and u < anker_y + heigth:
                if v >= anker_x and v < anker_x + width:
                    
                    new_frame[u - anker_y][v - anker_x] = full_frame[u][v]
                
    return np.asarray(new_frame)

def tensor_to_numpy(tensor):
    out = tensor.cpu().detach().numpy()
    out = out[0]
    out = out[0]
    return out

def show_video_frames(np_array, title):
    
    if len(np_array.shape) == 3:
        counter = 0
        for frame in np_array:
            counter += 1
            plt.imshow(frame)
            plt.title(title + ": " + str(counter))
            plt.show()
    else:
        raise ValueError("Dimensions of the input has to be 3, got dim: " + str(len(np_array.shape)) + " instead")

def get_total_var(array_original, array_reconstructed):
    
    total_err = 0
    
    for u in range(0, len(array_original)):
        
        batch_original = array_original[u].squeeze()
        batch_reconstructed = array_reconstructed[u].squeeze()
        
        for v in range(0, len(batch_original)):
            video_original = batch_original[v]
            video_reconstructed = batch_reconstructed[v]
            
            for w in range(0, len(video_original)):
                frame_original = video_original[w]
                frame_reconstructed = video_reconstructed[w]
    
                err = mse(frame_original, frame_reconstructed)
                
                total_err = total_err + err
    return total_err

def mse(imageA, imageB):
    
    imageA = np.asarray(imageA)
    imageB = np.asarray(imageB)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])  
    
    return err

def get_model_params(params): 
    
    values = []
    
    for param in params:
       
            values.append(0)
        
            value_grad = param.grad
            if value_grad == None:
                values.append(value_grad)
            else:
                values.append(value_grad.data.sum().item())
    return values        

def strip_string(string, keep_number=False):
    
    out = ""
    
    for char in string:
        if char == '.':
            return out
        if keep_number == False:
            if char in [')', '(', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                do = 'nothing'
            else:
                out += char
        else:
            out += char
        
    return out

def format_string(string):
    
    # From WORD(1) to WORD1
    
    out = ""

    for char in string:
        if char in ['(', ')']:
            do = 'nothing'
        else:
            out = out + char
    return out

def get_MPD_group(phoneme):
    
    phoneme = strip_string(phoneme.upper())
    
    if phoneme in ['AA', 'AH', 'EH', 'IH', 'AO', 'UW']:
        return 'group_O', 0
    if phoneme in ['D', 'DH', 'TH', 'L', 'N', 'T']:
        return 'group_D', 1
    if phoneme in ['EY', 'OY', 'G', 'Y', 'ZH', 'IY', 'JH']:
        return 'group_J', 2
    if phoneme in ['K', 'NG', 'R']:
        return 'group_K', 3
    if phoneme in ['AE', 'AW', 'AY', 'HH']:
        return 'group_A', 4
    if phoneme in ['S', 'Z', 'CH', 'SH']:
        return 'group_S', 5
    if phoneme in ['OW', 'UH', 'ER']:
        return 'group_E', 6
    if phoneme in ['B', 'P', 'M', 'W']:
        return 'group_B', 7
    if phoneme in ['V', 'F']:
        return 'group_V', 8

    raise Exception("Phoneme " + str(phoneme) + " was not found in any groups")
    
def get_MPD(phonemes_a, phonemes_b):
    
    cost_matrix = np.array([
        [0, 2, 1, 1, 1, 2, 1, 1, 1],
        [2, 0, 2, 2, 1, 1, 2, 1, 2],
        [1, 2, 0, 1, 1, 2, 1, 1, 2],
        [1, 2, 1, 0, 1, 2, 1, 2, 2],
        [1, 1, 1, 1, 0, 2, 2, 2, 2],
        [2, 1, 2, 2, 2, 0, 2, 1, 1],
        [1, 2, 1, 1, 2, 2, 0, 1, 1],
        [1, 1, 1, 2, 2, 1, 1, 0, 1],
        [1, 2, 2, 2, 2, 1, 1, 1, 0]
        ])
   
    matrix = np.zeros((len(phonemes_a)+1,len(phonemes_b)+1), dtype=int)
    
    for i in range(len(phonemes_a)+1): 
        for j in range(len(phonemes_b)+1): 
            
            if i == 0:  
                matrix[i][j] = j  
            elif j == 0: 
                matrix[i][j] = i
            else: 
                
                phoneme_a = phonemes_a[i - 1]
                phoneme_b = phonemes_b[j - 1]
                _, index_a = get_MPD_group(phoneme_a)
                _, index_b = get_MPD_group(phoneme_b)
                
                
                substitution_cost = cost_matrix[index_a, index_b]
                
                matrix[i][j] = min(matrix[i][j-1] + 1,  
                                   matrix[i-1][j] + 1,        
                                   matrix[i-1][j-1] + substitution_cost if phonemes_a[i-1] != phonemes_b[j-1] else matrix[i-1][j-1] + 0)     
                                   # Adjusted the cost accordinly, insertion = 1, deletion=1 and substitution=2
    
    return matrix[len(phonemes_a)][len(phonemes_b)]
        
def get_pld(phonemes_a, phonemes_b):
    #gets phonemic levenshtein distance
    
    len_a = len(phonemes_a)
    len_b = len(phonemes_b)
    d = [[i] for i in range(1, len_a + 1)]   
    d.insert(0, list(range(0, len_b + 1)))   
    for j in range(1, len_b + 1):
        for i in range(1, len_a + 1):
            if phonemes_a[i - 1] == phonemes_b[j - 1]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i].insert(j, min(d[i - 1][j] + 1,
                               d[i][j - 1] + 1,
                               d[i - 1][j - 1] + substitutionCost))
    return d[-1][-1]
    
    

def log_training(original, reconstructed, train_loss, validation_loss, test_loss, model_name, bs, pt, lr, wd, ne, opt, lf, dl, name, extra_info=False):
    #Big ugly function that creates the excell sheet for all the logging
    
    indices = ['C',  'K', 'S', 'AC', 'AK', 'AS', 'BC', 'BK', 'BS', 'CC', 'CK', 'CS', 'DC', 'DK', 'DS', 'EC', 'EK', 'ES', 'FC', 'FK', 'FS']
    
    info = pd.DataFrame([['Model', model_name], ['Batch size', bs], ['Participant', pt], 
                         ['Learning rate', lr], ['Weight decay', wd],  
                         ['Optimizer', opt], ['Loss function', lf], ['Data loader', dl], 
                         ['Test loss', test_loss],
                         ['Number of epochs', ne]])
    
    if isinstance(extra_info, bool):
        do = 'nothing'
    else:
        to_concat = [info, extra_info]
        
        info = pd.concat(to_concat)
    
    dt_string = name
    
    if len(original) == len(reconstructed):
        
        writer = pd.ExcelWriter('/Fridge/users/eli/Code/LSTM/logs/' + model_name + "_" + str(pt) + ',e=' + str(ne) + ",lr=" + str(lr) + "bs=" + str(bs) + "_" + str(dt_string) + '.xlsx', engine = 'xlsxwriter')
        info.to_excel(writer, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']

        video_original = original
        video_reconstructed = reconstructed
        
        if len(video_original) == len(video_reconstructed):
            
            for w in range(0, len(video_original)):
                
                plt.imshow(video_original[w])
                plt.title("Original video, frame " + str(w))
                plt.savefig('/Fridge/users/eli/Code/LSTM/files/temp/a_plot_' + str(w) + '.png')
                
                worksheet.insert_image(indices[w] + str(35),'/Fridge/users/eli/Code/LSTM/files/temp/a_plot_' + str(w) + '.png')
                
                plt.clf()
                
                dif = mse(video_original[w], video_reconstructed[w])
                
                plt.imshow(video_reconstructed[w].squeeze().squeeze())
                plt.title("Reconstructed video, frame " + str(w) + "; difference: " + str(dif))
                plt.savefig('/Fridge/users/eli/Code/LSTM/files/temp/a_plot2_' + str(w) + '.png')
                
                worksheet.insert_image(indices[w] + str(55),'/Fridge/users/eli/Code/LSTM/files/temp/a_plot2_' + str(w) + '.png')
                
                plt.clf()
        
        else:
            raise ValueError("Length of lists are not the same")
        
        plt.clf()
        plt.plot(train_loss[1:])
        plt.title("Train loss per epoch")
        plt.savefig('/Fridge/users/eli/Code/LSTM/files/temp/train_loss_plot.png')
        worksheet.insert_image('F4','/Fridge/users/eli/Code/LSTM/files/temp/train_loss_plot.png')
        
        plt.clf()
        half_index = int(len(train_loss) / 2)
        plt.plot(train_loss[-half_index:])
        plt.title("Train loss per epoch for the last 50 percent of training")
        plt.savefig('/Fridge/users/eli/Code/LSTM/files/temp/train_loss_last_half_plot.png')
        worksheet.insert_image('AP4','/Fridge/users/eli/Code/LSTM/files/temp/train_loss_last_half_plot.png')
        
        plt.clf()
        half_index = int(len(validation_loss) / 2)
        plt.plot(validation_loss[-half_index:])
        plt.title("Validation loss per epoch for the last 50 percent of training")
        plt.savefig('/Fridge/users/eli/Code/LSTM/files/temp/test_loss_last_half_plot.png')
        worksheet.insert_image('X4','/Fridge/users/eli/Code/LSTM/files/temp/test_loss_last_half_plot.png')
        
        plt.clf()
        plt.plot(validation_loss[1:])
        plt.title("Validation loss per epoch")
        plt.savefig('/Fridge/users/eli/Code/LSTM/files/temp/test_loss_plot.png')
        worksheet.insert_image('O4','/Fridge/users/eli/Code/LSTM/files/temp/test_loss_plot.png')
        
        
        writer.save()
        
    else:
        raise ValueError("Length of lists are not the same")    
        