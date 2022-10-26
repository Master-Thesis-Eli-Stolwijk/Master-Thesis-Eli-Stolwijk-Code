import numpy as np
import torch
import os
import pickle
import datetime
import cv2
import math
import random
import string
import nltk
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util

class LSTM_loader():
    
    def __init__(self, participant, batch_size, tt_ratio, shape):
        
        self.batch_size = batch_size
        self.shape = shape
        self.participant = participant
        self.name = "LSTM_loader"
        self.ratio = tt_ratio
    
    def get_batch(debug, batch): 
        
        """
        Returns data and labels of a batch seperately.
        """
        
        data = []
        labels = []
        for item in batch:
            
            labels.append(item.label.strip())
            
            data.append(torch.tensor(np.array(item.data)))
            
        return data, labels
    
    def train_test_split(self, data, ratio): # Divides the data into train and test split and provides one individual datapoint for the additional reconstruction graphics
        
        """
        Method that divides the data into a train, validation, test set and keeps one individual datapoint
        for the visual reconstrucion example
        """
    
        random.shuffle(data)        
        
        train = []
        test = []
        val = []
        individual = None
        
        
        test_val_size = int(len(data) * (1 - ratio))
        train_size = int(len(data) - (test_val_size * 2))
        
        for u in range(0,len(data)):
            if u == 0:
                individual = data[u]
            else:
                if u < train_size:
                    train.append(data[u])
                else:
                    if u < train_size + test_val_size:
                        val.append(data[u])
                    else:
                        test.append(data[u])
        return train, test, val, individual

    def load_data(self, root_dir): 
        
        """
        Loads the required data files from specified file path. When the data is loaded, it is pickled and stored.
        The next time this pickled file will be detected and loaded.
        """
    
        if self.participant.upper() == 'F1':
            anker = [8, 9]
        if self.participant.upper() == 'F2':
            anker = [8, 9]
        if self.participant.upper() == 'F3':
            anker = [5, 10]
        if self.participant.upper() == 'F4':
            anker = [4, 5]
        if self.participant.upper() == 'F5':
            anker = [5, 12]
        if self.participant.upper() == 'M2':
            anker = [9, 14]
        if self.participant.upper() == 'M3':
            anker = [5, 5]
        if self.participant.upper() == 'M4':
            anker = [5, 9]
        if self.participant.upper() == 'M5':
            anker = [5, 13]
    
    
        words = []
        arpabet = nltk.corpus.cmudict.dict()
        
        if os.path.exists("/Fridge/users/eli/Code/LSTM/files/pickled_files/" + "LSTM_np_words_cut_" + str(self.shape) + "_" + str(self.participant)):
            
            print("Pickled file detected, loading...")
            with open("/Fridge/users/eli/Code/LSTM/files/pickled_files/" + "LSTM_np_words_cut_" + str(self.shape) + "_" + str(self.participant), "rb") as fp:   # Unpickling
                list_words = pickle.load(fp)
            print("Loaded pickled file")
        else:
            print("Started loading files...")
            start_time = datetime.datetime.now()
            progress_count = 0
            list_words = []
            for filename in os.listdir(root_dir):
                progress_count += 1
                vidcap = cv2.VideoCapture(os.path.join(root_dir, filename))
                success,image = vidcap.read()
                
                video = []
                
                count = 0
                while success:    
                    success,image = vidcap.read()
                    if success:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
                        cut_image = Util.cut_frame(image, self.shape[0], self.shape[1], anker)
                        
                        video.append(cut_image)
                        
                    count += 1
                    
                new_word = Word(filename, video, arpabet)
                list_words.append(new_word)
                
                if progress_count % 100 == 0:
                    end_time = datetime.datetime.now()
                    time_diff = (end_time - start_time)
                    execution_time = time_diff.total_seconds()
                    progress = ((len(os.listdir(root_dir)) - progress_count) / 100) * execution_time
                    print("Loaded " + str(progress_count) + " out of " + str(len(os.listdir(root_dir))) + " videos in " + str(execution_time) + "seconds, ETA: " + str(int(progress)) + " seconds")
                    start_time = datetime.datetime.now()
                    
                
            
            with open("/Fridge/users/eli/Code/LSTM/files/pickled_files/" + "LSTM_np_words_cut_" + str(self.shape) + "_" + str(self.participant), "wb") as fp:   #Pickling
                pickle.dump(list_words, fp)
        
        
        clean_words = []
    
        for word in list_words:
            if isinstance(word.phoneme_ohe, (bool)) or len(word.data) < 5:
                do = 'nothing'
            else:
                clean_words.append(word)
        
        train_data, test_data, val_data, ind = self.train_test_split(clean_words, self.ratio)
        
        
        
        return train_data, test_data, val_data, ind, clean_words
        
    def get_loaders(self, path):
        
        """
        Returns the dataloaders
        """
        
        train, test, val, ind, all_data = self.load_data(path)
        
        train_loader = DataLoader(train,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=lambda batch: self.get_batch(batch))

        test_loader = DataLoader(test,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=lambda batch: self.get_batch(batch))
        
        val_loader = DataLoader(val,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=lambda batch: self.get_batch(batch))
        ind_loader = DataLoader([ind],
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=lambda batch: self.get_batch(batch))
        eval_loader = DataLoader(all_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=lambda batch: self.get_batch(batch))
        
        return train_loader, test_loader, val_loader, ind_loader, eval_loader
       
    
class Word: # this object stores the label, data and additional info such as the one hot encodings
    
    

    def __init__(self, label, data, arpa):
        self.label = label.upper()
        self.data = data
        self.arpa = arpa
        self.char_ohe = self.get_label_ohe(label)
        self.phoneme_ohe = self.get_phoneme_ohe(label)
        
    
    
    def get_label_ohe(self, label):
        
        """
        Returns a one hot encoding of the labels characters
        """
        
        stripped_label = Util.strip_string(label)
        
        vector = np.array([[0]*26]*len(stripped_label))
        
        for u in range(0, len(stripped_label)):
            index = ord(stripped_label[u]) - 97
            
            if index > 0 and index < 27:
                
                vector[u][index] = 1
    
        return vector
    
    def get_phoneme_ohe(self, label):
        
        """
        Returns a one hot encoding of the labels phonemes
        """
        
        word = Util.strip_string(label).lower().strip()
        arpabet = self.arpa
        nr_phonemes = 39
        try:
            phonemes = arpabet[word][0]
        except:
            print(word + " was not found in the nltk dictionary")
            return False
        
        vector = np.array([[0]* nr_phonemes] * len(phonemes))
        
        for u in range(0, len(phonemes)):
            phoneme = phonemes[u]

            index = self.get_index(phoneme)
            
            vector[u][index] = 1
        
        return vector
        
    def get_index(self, phoneme):
        
        """
        Returns indice of each phoneme for the one hot encodings
        """
        
        indicies = {
          "AA": 0,
          "AH": 1,
          "AW": 2,
          "B": 3,
          "D": 4,
          "EH": 5,
          "EY": 6,
          "G": 7,
          "IH": 8,
          "JH": 9,
          "L": 10,
          "N": 11,
          "OW": 12,
          "P": 13,
          "S": 14,
          "T": 15,
          "UH": 16,
          "V": 17,
          "Y": 18,
          "ZH": 19,
          "AE": 20,
          "AO": 21,
          "AY": 22,
          "CH": 23,
          "DH": 24,
          "ER": 25,
          "F": 26,
          "HH": 27,
          "IY": 28,
          "K": 29,
          "M": 30,
          "NG": 31,
          "OY": 32,
          "R": 33,
          "SH": 34,
          "TH": 35,
          "UW": 36,
          "W": 37,
          "Z": 38
        }
        
        
        
        return indicies[Util.strip_string(phoneme)]
        