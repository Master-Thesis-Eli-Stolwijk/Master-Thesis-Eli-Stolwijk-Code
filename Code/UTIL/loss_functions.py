import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.insert(0, '/Fridge/users/eli/Code/ANALYZER')
import analysis_tools as Analyzer



class per_frame_loss():
    
    def __init__(self):
        self.frames = [0 for x in range(20)]
        self.loss_functions = torch.nn.MSELoss()
        self.frames_counts = [0 for x in range(20)]
        self.name = "object to track loss per frame position"
    
    def update(self, reconstruction_batch, ground_truth_batch):
        
        reconstruction_batch = reconstruction_batch.squeeze()
        
        for u in range(0, len(reconstruction_batch)):
            
            reconstructed_video = reconstruction_batch[u]
            truth_video = ground_truth_batch[u]
            
            for v in range(0, 20):
                
                reconstructed_frame = reconstructed_video[v]
                truth_frame = truth_video[v]
                
                loss = self.loss_functions(reconstructed_frame, truth_frame).item()
                
                if loss > 10:
                    
                    self.frames[v] += loss
                    self.frames_counts[v] += 1
        
    def get_average_loss_per_frame(self):
         
         output = []
         
         for u in range(0, len(self.frames)):
             avg_loss = self.frames[u] / self.frames_counts[u]
             output.append(avg_loss)
             
         return output

class PLD_Loss(nn.Module):
    
   def __init__(self, lev_weight, device):
       super(PLD_Loss, self).__init__()
       self.torch_loss_function = torch.nn.MSELoss()
       self.lv = lev_weight
       self.device = device
       self.name = "Phonemic Levenshtein Distance loss function"
 
   def forward(self, sample1 , sample2, labels, representations): #, lev_diff):
      
       lev_y = torch.from_numpy(np.array(Analyzer.get_phonemic_levenshtein_matrix_from_tensor(labels))).to(self.device).to(dtype=torch.float)
                        
       lev_x = torch.cdist(representations, representations, p=2).to(self.device).to(dtype=torch.float)
            
       
       if sample1.size() != sample2.size():
           debug = 0
           raise Exception("Two video samples have to be same size, got size1=" + str(sample1.size()) + ", size2=" + str(sample2.size()))
    
       if lev_x.size() != lev_y.size():
           debug = 0
           raise Exception("Two lev samples have to be same size, got size1=" + str(lev_x.size()) + ", size2=" + str(lev_y.size())) 
       
       video_distance = self.torch_loss_function(sample1, sample2)
       
       custom_part = self.torch_loss_function(lev_x, lev_y)
       
       custom_part = torch.mul(custom_part, self.lv)
       
       losses = video_distance + custom_part
       
       
       return losses, video_distance, custom_part
   
class MPD_Loss(nn.Module):
    
   def __init__(self, lev_weight, device):
       super(MPD_Loss, self).__init__()
       self.torch_loss_function = torch.nn.MSELoss()
       self.lv = lev_weight
       self.device = device
       self.name = "Mouth Position Distance loss function"
 
   def forward(self, sample1 , sample2, labels, representations): #, lev_diff):
      
       lev_y = torch.from_numpy(np.array(Analyzer.get_mpd_matrix_from_tensor(labels))).to(self.device).to(dtype=torch.float)
                        
       lev_x = torch.cdist(representations, representations, p=2).to(self.device).to(dtype=torch.float)
            
       
       if sample1.size() != sample2.size():
           debug = 0
           raise Exception("Two video samples have to be same size, got size1=" + str(sample1.size()) + ", size2=" + str(sample2.size()))
    
       if lev_x.size() != lev_y.size():
           debug = 0
           raise Exception("Two lev samples have to be same size, got size1=" + str(lev_x.size()) + ", size2=" + str(lev_y.size()))
       
       video_distance = self.torch_loss_function(sample1, sample2)
       
       custom_part = self.torch_loss_function(lev_x, lev_y)
       
       custom_part = torch.mul(custom_part, self.lv)
       
       losses = video_distance + custom_part
       
       
       return losses, video_distance, custom_part
   
class Vanilla_Loss(nn.Module):
    
   def __init__(self):
       super(Vanilla_Loss, self).__init__()
       self.torch_loss_function = torch.nn.MSELoss()
       self.name = "standard MSE loss function"
 
   def forward(self, sample1 , sample2, lev_x, lev_y): #, lev_diff):
       
       if sample1.size() != sample2.size():
           raise Exception("Two video samples have to be same size, got size1=" + str(sample1.size()) + ", size2=" + str(sample2.size()))

       video_distance = self.torch_loss_function(sample1, sample2)
       
       dummy = self.torch_loss_function(torch.tensor(0.0), torch.tensor(0.0))
       
       return video_distance, dummy, dummy


class PLD_only_Loss(nn.Module):
    
   def __init__(self, vector_size, weight, device):
       super(PLD_only_Loss, self).__init__()
       self.torch_loss_function = torch.nn.MSELoss()
       self.w = weight
       self.device = device
       self.vs = vector_size
       self.name = "Phonemic Levenshtein Distance loss function"
    
   def get_dummy(self, representations):
       
       lev_x = torch.cdist(representations, representations, p=2).to(self.device).to(dtype=torch.float)
       
       loss = self.torch_loss_function(lev_x, lev_x)
       
       return loss
       
   def forward(self, labels, representations): #, lev_diff):
      
       lev_y = torch.from_numpy(np.array(Analyzer.get_phonemic_levenshtein_matrix_from_tensor(labels))).to(self.device).to(dtype=torch.float)
                        
       lev_x = torch.cdist(representations, representations, p=2).to(self.device).to(dtype=torch.float)
       
       loss = self.torch_loss_function(lev_x, lev_y)
       
       custom_part = torch.mul(loss, self.w)
       
       return loss