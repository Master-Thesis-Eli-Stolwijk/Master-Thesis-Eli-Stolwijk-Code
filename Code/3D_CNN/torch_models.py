import torch
from torch import Tensor

    
class CNN_AE_V1(torch.nn.Module):

    def __init__(self, bn_size, conv_do_rate):
        
        super().__init__()
        
        self.name = "CNN_AE_V1(" + str(bn_size) + ")"
        
        self.bottleneck_size = bn_size
        self.conv_drop_out_rate = conv_do_rate
        
        self.encoder_part1 = torch.nn.Sequential(
            
            torch.nn.Conv3d(1, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.encoder_part2 = torch.nn.Sequential(
            
            torch.nn.Conv3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.Conv3d(60, 10, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.flat1 = torch.nn.Flatten()
        
        self.pad = torch.nn.ConstantPad1d(padding=0, value=0)
        
        self.fc1 = torch.nn.Linear(in_features=11520,out_features=self.bottleneck_size)
        
        self.fc2 =  torch.nn.Linear(in_features=self.bottleneck_size, out_features=11520)
        
        self.unflat1 = torch.nn.Unflatten(1, unflattened_size=(0,0,0,0))
        
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part1 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(10, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            )
        
        self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part2 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 1, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU()
            
        )
    
    def encoder(self, features: Tensor) -> Tensor:
        
        encoded, indices1 = self.encoder_part1(features)
        
        encoded, indices2 = self.encoder_part2(encoded)
        
        unflat_shape = encoded.shape
        
        encoded = self.flat1(encoded)
        
        encoded = self.fc1(encoded)
        
        return encoded, indices1, indices2, unflat_shape
    
    def decoder(self, encoded: Tensor, indices1, indices2, unflat_shape) -> Tensor:
        
        decoded = self.fc2(encoded)
        
        self.unflat1.unflattened_size = (unflat_shape[1], unflat_shape[2], unflat_shape[3], unflat_shape[4])
        
        decoded = self.unflat1(decoded)
        
        unpooled1 = self.unpool1(decoded, indices2)
        
        decoded = self.decoder_part1(unpooled1)
        
        unpooled2 = self.unpool2(decoded, indices1)
        
        decoded = self.decoder_part2(unpooled2)
        
        return decoded
        
    def forward(self, x):
        
        encoded, indices1, indices2, unflat_shape = self.encoder(x)
        
        decoded = self.decoder(encoded, indices1, indices2, unflat_shape)
        
        return decoded, encoded
    

    
 
class CNN_AE_D1(torch.nn.Module):

    def __init__(self, bn_size, conv_do_rate):
        
        super().__init__()
        
        self.name = "CNN_AE_D1(" + str(bn_size) + ")"
        
        self.bottleneck_size = bn_size
        self.conv_drop_out_rate = conv_do_rate
        
        self.encoder_part1 = torch.nn.Sequential(
            
            torch.nn.Conv3d(1, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
     #       torch.nn.BatchNorm3d(60),
            torch.nn.Conv3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
     #       torch.nn.BatchNorm3d(120),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.encoder_part2 = torch.nn.Sequential(
            
            torch.nn.Conv3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
      #      torch.nn.BatchNorm3d(60),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.Conv3d(60, 10, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
      #      torch.nn.BatchNorm3d(10),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.flat1 = torch.nn.Flatten()
        
        self.flat2 = torch.nn.Flatten()
        
        self.pad = torch.nn.ConstantPad1d(padding=0, value=0)
        
        self.fc1 = torch.nn.Linear(in_features=11910,out_features=self.bottleneck_size)
        
        self.fc2 =  torch.nn.Linear(in_features=self.bottleneck_size, out_features=11520)
        
        self.unflat1 = torch.nn.Unflatten(1, unflattened_size=(0,0,0,0))
        
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part1 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(10, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
   #         torch.nn.BatchNorm3d(60),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
     #       torch.nn.BatchNorm3d(120),
            torch.nn.Dropout(self.conv_drop_out_rate)
            )
        
        self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part2 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
     #       torch.nn.BatchNorm3d(60),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 1, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU()
     #       torch.nn.BatchNorm3d(1),
        )
    
    def encoder(self, features: Tensor, y: Tensor) -> Tensor:
        
        encoded, indices1 = self.encoder_part1(features)
        
        encoded, indices2 = self.encoder_part2(encoded)
        
        unflat_shape = encoded.shape
        
        encoded = self.flat1(encoded)
        
        #######################Incorporating ohe####################
        
        y = self.flat2(y)
        
        encoded = torch.cat((encoded, y), 1)
        
        encoded = self.fc1(encoded)
        
        return encoded, indices1, indices2, unflat_shape
    
    
    def decoder(self, encoded: Tensor, indices1, indices2, unflat_shape) -> Tensor:
        
        decoded = self.fc2(encoded)
        
        self.unflat1.unflattened_size = (unflat_shape[1], unflat_shape[2], unflat_shape[3], unflat_shape[4])
        
        decoded = self.unflat1(decoded)
        
        unpooled1 = self.unpool1(decoded, indices2)
        
        decoded = self.decoder_part1(unpooled1)
        
        unpooled2 = self.unpool2(decoded, indices1)
        
        decoded = self.decoder_part2(unpooled2)
        
        return decoded
    
        
    def forward(self, x, y):
        
        encoded, indices1, indices2, unflat_shape = self.encoder(x, y)
        
        decoded = self.decoder(encoded, indices1, indices2, unflat_shape)
        
        return decoded, encoded

class CNN_AE_P1(torch.nn.Module):

    def __init__(self, bn_size, conv_do_rate):
        
        super().__init__()
        
        self.name = "CNN_AE_P1(" + str(bn_size) + ")"
        
        self.bottleneck_size = bn_size
        self.conv_drop_out_rate = conv_do_rate
        
        self.encoder_part1 = torch.nn.Sequential(
            
            torch.nn.Conv3d(1, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.encoder_part2 = torch.nn.Sequential(
            
            torch.nn.Conv3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.Conv3d(60, 10, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.flat1 = torch.nn.Flatten()
        
        self.flat2 = torch.nn.Flatten()
        
        self.pad = torch.nn.ConstantPad1d(padding=0, value=0)
        
        self.fc1 = torch.nn.Linear(in_features=12027,out_features=self.bottleneck_size)
        
        self.fc2 =  torch.nn.Linear(in_features=self.bottleneck_size, out_features=11520)
        
        self.unflat1 = torch.nn.Unflatten(1, unflattened_size=(0,0,0,0))
        
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part1 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(10, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate)
            )
        
        self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part2 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 1, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU()
        )
    
    def encoder(self, features: Tensor, y: Tensor) -> Tensor:
        
        encoded, indices1 = self.encoder_part1(features)
        
        encoded, indices2 = self.encoder_part2(encoded)
        
        unflat_shape = encoded.shape
        
        encoded = self.flat1(encoded)
        
        #######################Incorporating ohe####################
        
        y = self.flat2(y)
        
        encoded = torch.cat((encoded, y), 1)
        
        encoded = self.fc1(encoded)
        
        return encoded, indices1, indices2, unflat_shape
    
    
    def decoder(self, encoded: Tensor, indices1, indices2, unflat_shape) -> Tensor:
        
        decoded = self.fc2(encoded)
        
        self.unflat1.unflattened_size = (unflat_shape[1], unflat_shape[2], unflat_shape[3], unflat_shape[4])
        
        decoded = self.unflat1(decoded)
        
        unpooled1 = self.unpool1(decoded, indices2)
        
        decoded = self.decoder_part1(unpooled1)
        
        unpooled2 = self.unpool2(decoded, indices1)
        
        decoded = self.decoder_part2(unpooled2)
        
        return decoded
    
        
    def forward(self, x, y):
        
        encoded, indices1, indices2, unflat_shape = self.encoder(x, y)
        
        decoded = self.decoder(encoded, indices1, indices2, unflat_shape)
        
        return decoded, encoded
    

class CNN_AE_PD1(torch.nn.Module):

    def __init__(self, bn_size, conv_do_rate):
        
        super().__init__()
        
        self.name = "CNN_AE_PD1(" + str(bn_size) + ")"
        
        self.bottleneck_size = bn_size
        self.conv_drop_out_rate = conv_do_rate
        
        self.encoder_part1 = torch.nn.Sequential(
            
            torch.nn.Conv3d(1, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.encoder_part2 = torch.nn.Sequential(
            
            torch.nn.Conv3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.Conv3d(60, 10, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.flat1 = torch.nn.Flatten()
        
        self.flat2 = torch.nn.Flatten()
        
        self.flat3 = torch.nn.Flatten()
        
        self.pad = torch.nn.ConstantPad1d(padding=0, value=0)
        
        self.fc1 = torch.nn.Linear(in_features=12417,out_features=self.bottleneck_size)
        
        self.fc2 =  torch.nn.Linear(in_features=self.bottleneck_size, out_features=11520)
        
        self.unflat1 = torch.nn.Unflatten(1, unflattened_size=(0,0,0,0))
        
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part1 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(10, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate)
            )
        
        self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part2 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 1, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU()
        )
    
    def encoder(self, features: Tensor, y1: Tensor, y2: Tensor) -> Tensor:
        
        encoded, indices1 = self.encoder_part1(features)
        
        encoded, indices2 = self.encoder_part2(encoded)
        
        unflat_shape = encoded.shape
        
        encoded = self.flat1(encoded)
        
        #######################Incorporating ohe####################
        
        y1 = self.flat2(y1)
        
        y2 = self.flat3(y2)
        
        encoded = torch.cat((encoded, y1, y2), 1)
        
        encoded = self.fc1(encoded)
        
        return encoded, indices1, indices2, unflat_shape
    
    
    def decoder(self, encoded: Tensor, indices1, indices2, unflat_shape) -> Tensor:
        
        decoded = self.fc2(encoded)
        
        self.unflat1.unflattened_size = (unflat_shape[1], unflat_shape[2], unflat_shape[3], unflat_shape[4])
        
        decoded = self.unflat1(decoded)
        
        unpooled1 = self.unpool1(decoded, indices2)
        
        decoded = self.decoder_part1(unpooled1)
        
        unpooled2 = self.unpool2(decoded, indices1)
        
        decoded = self.decoder_part2(unpooled2)
        
        return decoded
    
        
    def forward(self, x, y1, y2):
        
        encoded, indices1, indices2, unflat_shape = self.encoder(x, y1, y2)
        
        decoded = self.decoder(encoded, indices1, indices2, unflat_shape)
        
        return decoded, encoded
    
    


    
class CNN_AE_P01(torch.nn.Module):

    def __init__(self, bn_size, conv_do_rate):
        
        super().__init__()
        
        self.name = "CNN_AE_P01(" + str(bn_size) + ")"
        
        self.bottleneck_size = bn_size
        self.conv_drop_out_rate = conv_do_rate
        
        self.encoder_part1 = torch.nn.Sequential(
            
            torch.nn.Conv3d(1, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.encoder_part2 = torch.nn.Sequential(
            
            torch.nn.Conv3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.Conv3d(60, 1, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.flat1 = torch.nn.Flatten()
        
        self.flat2 = torch.nn.Flatten()
        
        self.pad = torch.nn.ConstantPad1d(padding=0, value=0)
        
        self.fc1 = torch.nn.Linear(in_features=1659,out_features=self.bottleneck_size)
        
        self.fc2 =  torch.nn.Linear(in_features=self.bottleneck_size, out_features=1152)
        
        self.unflat1 = torch.nn.Unflatten(1, unflattened_size=(0,0,0,0))
        
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part1 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(1, 60, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 120, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate)
            )
        
        self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part2 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(120, 60, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.conv_drop_out_rate),
            torch.nn.ConvTranspose3d(60, 1, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU()
        )
    
    def encoder(self, features: Tensor, y: Tensor) -> Tensor:
        
        encoded, indices1 = self.encoder_part1(features)
        
        encoded, indices2 = self.encoder_part2(encoded)
        
        unflat_shape = encoded.shape
        
        encoded = self.flat1(encoded)
        
        #######################Incorporating ohe####################
        
        y = self.flat2(y)
        
        encoded = torch.cat((encoded, y), 1)
        
        encoded = self.fc1(encoded)
        
        return encoded, indices1, indices2, unflat_shape
    
    
    def decoder(self, encoded: Tensor, indices1, indices2, unflat_shape) -> Tensor:
        
        decoded = self.fc2(encoded)
        
        self.unflat1.unflattened_size = (unflat_shape[1], unflat_shape[2], unflat_shape[3], unflat_shape[4])
        
        decoded = self.unflat1(decoded)
        
        unpooled1 = self.unpool1(decoded, indices2)
        
        decoded = self.decoder_part1(unpooled1)
        
        unpooled2 = self.unpool2(decoded, indices1)
        
        decoded = self.decoder_part2(unpooled2)
        
        return decoded
    
        
    def forward(self, x, y):
        
        encoded, indices1, indices2, unflat_shape = self.encoder(x, y)
        
        decoded = self.decoder(encoded, indices1, indices2, unflat_shape)
        
        return decoded, encoded