import tkinter as tk
from tkinter import filedialog
import pickle
import torch
from Condensed_Word import Condensed_word


class Condenser:
    
    def condense(self, loader, model_mode, model=None):
        
        #Condenses all words with the model given, if no model is given a filedialog is opened to select one
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        if model == None:
            root = tk.Tk()
            root.withdraw()
            
            model_path = filedialog.askopenfilename()
            
            with open(model_path, "rb") as fp:   # Unpickling
                model = pickle.load(fp)
        
        
        list_of_representations = []
        
        print("Started condensing...")
        
        eval_iter = iter(loader)
        
        with torch.no_grad():
                    
            model = model.eval()
        
            for u in range(0, len(eval_iter)):
                
                if model_mode == '3DCNN':
                
                    eval_item_data, eval_item_label, eval_indices, eval_char_ohe, eval_phoneme_ohe = next(eval_iter)
                    
                    eval_item_data = torch.stack(eval_item_data)
                        
                    eval_item_data = eval_item_data.unsqueeze(1)
                    
                    eval_phoneme_ohe_batch = torch.stack(eval_phoneme_ohe)
                    
                    representation,_,_,_ = model.encoder(eval_item_data.to(device).to(dtype=torch.float), eval_phoneme_ohe_batch.to(device))
                
                if model_mode == 'LSTM':
                    
                    eval_item_data, eval_item_label = next(eval_iter)
                    
                    eval_item_data = torch.stack(eval_item_data)
                        
                    eval_item_data = eval_item_data.unsqueeze(1)
                    
                    _, representation = model(eval_item_data.to(device).to(dtype=torch.float))
                    
                    representation = representation[0][0].squeeze().flatten()
                
                if model_mode == 'GRU':
                    
                    eval_item_data, eval_item_label, eval_item_phonemes = next(eval_iter)
                    
                    eval_item_data = torch.stack(eval_item_data)
                        
                    eval_item_data = eval_item_data.unsqueeze(1)
                    
                    eval_item_phonemes = torch.stack(eval_item_phonemes)
                    
                    _, representation = model(eval_item_data.to(device).to(dtype=torch.float), eval_item_phonemes.to(device))
                    
                    if isinstance(representation, list):
                        
                        representation = representation[0]
                    
                    representation = representation.squeeze().flatten()
                    
                condensed = Condensed_word(eval_item_label[0], representation.cpu().detach().numpy())
            
                list_of_representations.append(condensed)
            
        print("Finished condensing")
        
        self.representations = list_of_representations
        
        return list_of_representations
    
        
    def condense_and_evaluate(self, loader, model=None):
        
        representations = self.condense(loader, model)
        
        return self.evaluate(representations)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        