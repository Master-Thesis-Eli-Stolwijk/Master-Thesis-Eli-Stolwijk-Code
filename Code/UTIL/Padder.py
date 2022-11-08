import numpy as np
import torch


def phoneme_padder(batch, phoneme_len):
    
    data = []
    labels = []
    phoneme_ohes = []
    for item in batch:
        
        labels.append(item.label.strip())
        
        data.append(torch.tensor(np.array(item.data)))
        
        phoneme_ohe_data = torch.tensor(item.phoneme_ohe)
        
        if phoneme_ohe_data.shape[0] < phoneme_len:
            a = phoneme_len - phoneme_ohe_data.shape[0]
            pd = (0,0,int(a/2), a- int(a/2))
        
            phoneme_ohes.append(torch.nn.functional.pad(phoneme_ohe_data, pd, value=0))
        else:
            phoneme_ohes.append(phoneme_ohe_data)
            
    return data, labels, phoneme_ohes

def batch_padder(batch, seq_len, char_len, phoneme_len): 

    data = []
    indices = []
    labels = []
    char_ohes = []
    phoneme_ohes = []
    for item in batch:
        
        labels.append(item.label.strip())
        
        
        ######################## Padding video data ######################
        pt_data = torch.from_numpy(item.data)
        
        if pt_data.shape[0] < seq_len:
            a = seq_len - pt_data.shape[0]
            pd = (0,0,0,0,int(a/2), a- int(a/2))
        
            data.append(torch.nn.functional.pad(pt_data, pd, value=np.min(pt_data.numpy())))
        else:
            data.append(pt_data)
        indices.append(pt_data.shape[0])
        
        
        ##################### Padding character one hot encodings ######################
        char_ohe_data = torch.tensor(item.char_ohe)
        
        if char_ohe_data.shape[0] < char_len:
            a = char_len - char_ohe_data.shape[0]
            pd = (0,0,int(a/2), a- int(a/2))
        
            char_ohes.append(torch.flatten(torch.nn.functional.pad(char_ohe_data, pd, value=0)))
        else:
            char_ohes.append(torch.flatten(char_ohe_data))
            
        ##################### Padding character one hot encodings ######################
        phoneme_ohe_data = torch.tensor(item.phoneme_ohe)
        
       # print(item.label)
       # print(phoneme_ohe_data)
        
        if phoneme_ohe_data.shape[0] < phoneme_len:
            a = phoneme_len - phoneme_ohe_data.shape[0]
            pd = (0,0,int(a/2), a- int(a/2))
        
            phoneme_ohes.append(torch.nn.functional.pad(phoneme_ohe_data, pd, value=0))
        else:
            phoneme_ohes.append(phoneme_ohe_data)
        
    return data, indices, labels, char_ohes, phoneme_ohes
    
def ind_padder(item, seq_len, char_len, phoneme_len): 
    
    data = []
    indices = []
    labels = []
    char_ohes = []
    phoneme_ohes = []
    item = item[0]
    
    labels.append(item.label.strip())
    
    ######################## Padding video data ######################
    pt_data = torch.from_numpy(item.data)
    
    if pt_data.shape[0] < seq_len:
        a = seq_len - pt_data.shape[0]
        pd = (0,0,0,0,int(a/2), a- int(a/2))
    
        data.append(torch.nn.functional.pad(pt_data, pd, value=np.min(pt_data.numpy())))
    else:
        data.append(pt_data)
    indices.append(pt_data.shape[0])
    
    
    ##################### Padding characters one hot encodings ######################
    char_ohe_data = torch.tensor(item.char_ohe)
    
    if char_ohe_data.shape[0] < char_len:
        a = char_len - char_ohe_data.shape[0]
        pd = (0,0,int(a/2), a- int(a/2))
    
        char_ohes.append(torch.nn.functional.pad(char_ohe_data, pd, value=0))
    else:
        char_ohes.append(char_ohe_data)
        
    ##################### Padding phonemes one hot encodings ######################
    phoneme_ohe_data = torch.tensor(item.phoneme_ohe)
    
    if phoneme_ohe_data.shape[0] < phoneme_len:
        a = phoneme_len - phoneme_ohe_data.shape[0]
        pd = (0,0,int(a/2), a- int(a/2))
    
        phoneme_ohes.append(torch.nn.functional.pad(phoneme_ohe_data, pd, value=0))
    else:
        phoneme_ohes.append(phoneme_ohe_data)
    
    return data, indices, labels, char_ohes, phoneme_ohes
    

def word_padder(batch, seq_len, char_len, phoneme_len): 
    
    data = []
    item_labels = []
    indices = []
    char_ohes = []
    phoneme_ohes = []
    for item in batch:
        item_labels.append(item.label.strip())
        item_data = item.data
        if item_data.shape[0] < seq_len:
            a = seq_len - item_data.shape[0]
            pd = (0,0,0,0,int(a/2), a- int(a/2))
        
            data.append(torch.nn.functional.pad(torch.from_numpy(item_data), pd, value=np.min(item_data)))
        else:
            data.append(torch.from_numpy(item_data))
        indices.append(item_data.shape[0])
        
        
    ##################### Padding char one hot encodings ######################
        char_ohe_data = torch.tensor(item.char_ohe)
        
        if char_ohe_data.shape[0] < char_len:
            a = char_len - char_ohe_data.shape[0]
            pd = (0,0,int(a/2), a- int(a/2))
        
            char_ohes.append(torch.nn.functional.pad(char_ohe_data, pd, value=0))
        else:
            char_ohes.append(char_ohe_data)
            
        ##################### Padding char one hot encodings ######################
        phoneme_ohe_data = torch.tensor(item.phoneme_ohe)
        
        if phoneme_ohe_data.shape[0] < phoneme_len:
            a = phoneme_len - phoneme_ohe_data.shape[0]
            pd = (0,0,int(a/2), a- int(a/2))
        
            phoneme_ohes.append(torch.nn.functional.pad(phoneme_ohe_data, pd, value=0))
        else:
            phoneme_ohes.append(phoneme_ohe_data)
        
    return data, item_labels, indices, char_ohes, phoneme_ohes