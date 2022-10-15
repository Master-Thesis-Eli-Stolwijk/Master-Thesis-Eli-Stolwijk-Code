import torch
import torch.nn as nn
from ConvLSTM import ConvLSTM

class ST_AutoEncoder(nn.Module):
    """
    Sequential Model for the Spatio Temporal Autoencoder (ST_AutoEncoder)
    """
    
    def __init__(self, in_channel):
        super(ST_AutoEncoder, self).__init__()
        
        self.in_channel = in_channel
        self.name = "L0(Works great but too high dimension)"
        
        # Spatial Encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channel, out_channels=128, kernel_size=(1,3,3), stride=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1,3,3), stride=(1,1,1)),
            nn.ReLU()            
        )
                
        # Temporal Encoder & Decoder
        self.temporal_encoder_decoder = Temporal_EncDec()
        
        # Spatial Decoder
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=128, kernel_size=(1,3,3), stride=(1,1,1)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=1, kernel_size=(1,3,3), stride=(1,1,1)),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x : (N, C, T, H, W) / T : sequence length(or D)        
        h = self.spatial_encoder(x)
        
        h = h.permute(0,2,1,3,4)  # (N, C, T, H, W) -> (N, T, C, H, W) 
        h_hat = self.temporal_encoder_decoder(h)
        
        h_hat = h_hat.permute(0,2,1,3,4)  # (N, T, C, H, W) -> (N, C, T, H, W) 
        output = self.spatial_decoder(h_hat)
        return output
    
    
class Temporal_EncDec(nn.Module):
    def __init__(self):
        super(Temporal_EncDec, self).__init__()
        
        self.convlstm_1 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_2 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_3 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3,3), num_layers=1, bias=True)
        
    def forward(self, x):
        layer_output_list, _ = self.convlstm_1(x)
        layer_output_list, _ = self.convlstm_2(layer_output_list[0])
        layer_output_list, _ = self.convlstm_3(layer_output_list[0])
        
        return layer_output_list[0]
    
class ST_AutoEncoder_V1(nn.Module):
    """
    Sequential Model for the Spatio Temporal Autoencoder (ST_AutoEncoder)
    """
    
    def __init__(self, in_channel):
        super(ST_AutoEncoder_V1, self).__init__()
        
        self.in_channel = in_channel
        self.name = "V1 (LSTM_adaptation)"
        
        self.encoder_part1 = torch.nn.Sequential(
            
            torch.nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        self.encoder_part2 = torch.nn.Sequential(
            
            torch.nn.Conv3d(128, 64, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64, 16, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
        
        
        self.temporal_encoder_decoder = Temporal_EncDec_V1()
        
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part1 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(16, 64, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(64, 128, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            )
        
        self.unpool2 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        self.decoder_part2 = torch.nn.Sequential(
            
            torch.nn.ConvTranspose3d(128, 64, kernel_size=(1, 3, 3), padding=0),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(64, 1, kernel_size=(3, 3, 3), padding=0),
            torch.nn.ReLU()
            
        )
    
    def encoder(self, x):
        
        encoded, indices1 = self.encoder_part1(x)
        
        encoded, indices2 = self.encoder_part2(encoded)
        
        return encoded, indices1, indices2
    
    def decoder(self, encoded, indices1, indices2):
        
        unpooled1 = self.unpool1(encoded, indices2)
        
        decoded = self.decoder_part1(unpooled1)
        
        unpooled2 = self.unpool2(decoded, indices1)
        
        decoded = self.decoder_part2(unpooled2)
        
        return decoded
        
    def forward(self, x):
        
        h, indices1, indices2 = self.encoder(x)
        
        h = h.permute(0,2,1,3,4)
        
        h_hat, bottleneck = self.temporal_encoder_decoder(h)
        
        h_hat = h_hat.permute(0,2,1,3,4)
        
        decoded = self.decoder(h_hat, indices1, indices2)
        
        return decoded, bottleneck
    
    
class Temporal_EncDec_V1(nn.Module):
    def __init__(self):
        super(Temporal_EncDec_V1, self).__init__()
        
        self.convlstm_1 = ConvLSTM(input_dim=16, hidden_dim=16, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_2 = ConvLSTM(input_dim=16, hidden_dim=8, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_3 = ConvLSTM(input_dim=8, hidden_dim=16, kernel_size=(3,3), num_layers=1, bias=True)
        
        
    def forward(self, x):
        layer_output_list, _ = self.convlstm_1(x)
        
        layer_output_list, bottleneck = self.convlstm_2(layer_output_list[0])
        
        layer_output_list, _ = self.convlstm_3(layer_output_list[0])
        
        return layer_output_list[0], bottleneck
    
class ST_AutoEncoder_exp(nn.Module):
    """
    Sequential Model for the Spatio Temporal Autoencoder (ST_AutoEncoder)
    """
    
    def __init__(self, in_channel):
        super(ST_AutoEncoder_exp, self).__init__()
        
        self.in_channel = in_channel
        self.name = "ConvLSTM_AE_exp"
        
        # Spatial Encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channel, out_channels=16, kernel_size=(1,5,5), stride=(1,2,2)),
            nn.ReLU()
            
        )
                
        # Temporal Encoder & Decoder
        self.temporal_encoder_decoder = Temporal_EncDec_exp()
        
        # Spatial Decoder
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=16, out_channels=self.in_channel, kernel_size=(1,11,11), stride=(1,1,1)),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x : (N, C, T, H, W) / T : sequence length(or D) 
        
        h = self.spatial_encoder(x)
        
        h = h.permute(0,2,1,3,4)  # (N, C, T, H, W) -> (N, T, C, H, W) 
        h_hat = self.temporal_encoder_decoder(h)
        
        h_hat = h_hat.permute(0,2,1,3,4)  # (N, T, C, H, W) -> (N, C, T, H, W) 
        output = self.spatial_decoder(h_hat)
        return output
    
    
class Temporal_EncDec_exp(nn.Module):
    def __init__(self):
        super(Temporal_EncDec_exp, self).__init__()
        
        self.convlstm_1 = ConvLSTM(input_dim=16, hidden_dim=16, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_2 = ConvLSTM(input_dim=16, hidden_dim=8, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)

        self.convlstm_3 = ConvLSTM(input_dim=8, hidden_dim=16, kernel_size=(3,3), num_layers=1, bias=True)
        
        
    def forward(self, x):
        layer_output_list, _ = self.convlstm_1(x)
        
        layer_output_list, last_state = self.convlstm_2(layer_output_list[0])
        
        layer_output_list, _ = self.convlstm_3(layer_output_list[0])
        
        return layer_output_list[0]
    
class ST_AutoEncoder_pool(nn.Module):
    """
    Sequential Model for the Spatio Temporal Autoencoder (ST_AutoEncoder)
    """
    
    def __init__(self, in_channel):
        super(ST_AutoEncoder_pool, self).__init__()
        
        self.in_channel = in_channel
        self.name = "ConvLSTM_AE_pool"
        
        # Spatial Encoder
        self.spatial_encoder_part1 = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channel, out_channels=128, kernel_size=(1,11,11), stride=(1,1,1)),
            nn.ReLU()
        )
        
        self.spatial_encoder_part2= nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1,5,5), stride=(1,1,1)),
            nn.ReLU(),
            torch.nn.MaxPool3d((1, 2, 2), return_indices = True)
        )
                
        # Temporal Encoder & Decoder
        self.temporal_encoder_decoder = Temporal_EncDec_pool()
        
        self.unpool1 = torch.nn.MaxUnpool3d(kernel_size=(1, 2, 2))
        
        # Spatial Decoder
        self.spatial_decoder_part1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=128, kernel_size=(1,5,5), stride=(1,1,1)),
            nn.ReLU()
        )
        
        self.spatial_decoder_part2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=128, out_channels=self.in_channel, kernel_size=(1,11,11), stride=(1,1,1)),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x : (N, C, T, H, W) / T : sequence length(or D) 
        
        h = self.spatial_encoder_part1(x)
        
        h, indices_2 = self.spatial_encoder_part2(h)
        
        h = h.permute(0,2,1,3,4)  # (N, C, T, H, W) -> (N, T, C, H, W) 
        h_hat = self.temporal_encoder_decoder(h)
        
        h_hat = h_hat.permute(0,2,1,3,4)  # (N, T, C, H, W) -> (N, C, T, H, W) 
        
        h_hat = self.unpool1(h_hat, indices_2)
        
        h_hat = self.spatial_decoder_part1(h_hat)
        
        output = self.spatial_decoder_part2(h_hat)
        
        
        return output
    
    
class Temporal_EncDec_pool(nn.Module):
    def __init__(self):
        super(Temporal_EncDec_pool, self).__init__()
        
        self.convlstm_1 = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_2 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_3 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3,3), num_layers=1, bias=True)
        
    def forward(self, x):
        layer_output_list, _ = self.convlstm_1(x)
        
        layer_output_list, bottle_neck = self.convlstm_2(layer_output_list[0])
        
        layer_output_list, _ = self.convlstm_3(layer_output_list[0])
        
        return layer_output_list[0]
    
class ST_AutoEncoder_L1(nn.Module):
    """
    Sequential Model for the Spatio Temporal Autoencoder (ST_AutoEncoder)
    """
    
    def __init__(self, in_channel):
        super(ST_AutoEncoder_L1, self).__init__()
        
        self.in_channel = in_channel
        self.name = "L1"
        
        # Spatial Encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv3d(in_channels=self.in_channel, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2)),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=32, kernel_size=(1,3,3), stride=(1,2,2)),
            nn.ReLU()            
        )
                
        # Temporal Encoder & Decoder
        self.temporal_encoder_decoder = Temporal_EncDec_L1()
        
        # Spatial Decoder
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose3d(in_channels=32, out_channels=128, kernel_size=(1,3,3), stride=(1,2,2)),
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=1, kernel_size=(1,3,3), stride=(1,2,2)),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x : (N, C, T, H, W) / T : sequence length(or D)        
        h = self.spatial_encoder(x)
        
        h = h.permute(0,2,1,3,4)  # (N, C, T, H, W) -> (N, T, C, H, W) 
        h_hat, bottleneck = self.temporal_encoder_decoder(h)
        
        h_hat = h_hat.permute(0,2,1,3,4)  # (N, T, C, H, W) -> (N, C, T, H, W) 
        output = self.spatial_decoder(h_hat)
        return output, bottleneck
    
    
class Temporal_EncDec_L1(nn.Module):
    def __init__(self):
        super(Temporal_EncDec_L1, self).__init__()
        
        self.convlstm_1 = ConvLSTM(input_dim=32, hidden_dim=32, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_2 = ConvLSTM(input_dim=32, hidden_dim=8, kernel_size=(3,3), num_layers=1, bias=True, return_all_layers=True)
        self.convlstm_3 = ConvLSTM(input_dim=8, hidden_dim=32, kernel_size=(3,3), num_layers=1, bias=True)
        
    def forward(self, x):
        layer_output_list, _ = self.convlstm_1(x)
        layer_output_list, bottleneck = self.convlstm_2(layer_output_list[0])
        layer_output_list, _ = self.convlstm_3(layer_output_list[0])
        
        return layer_output_list[0], bottleneck