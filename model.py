import torch
from torch import nn
from torch.nn import functional as F
import math
from self_attention import SelfAttention
from vae_attention import VAE_AttentionBlock
from residual_block import VAE_ResidualBlock
from encoder import VAE_Encoder
from decoder import VAE_Decoder

class VAE_NET(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def reparameterize(self, mean, std_dev):
        # Sample from the normal distribution with mean and std deviation
        eps = torch.randn_like(std_dev)
        return mean + eps * std_dev    

    def forward(self, x):
        # Forward pass through the encoder to get mean and std deviation
        mean, log_variance = self.encoder(x)
        
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # Sample from the distribution defined by mean and std deviation
        latent_space = self.reparameterize(mean, stdev)
        
        # Forward pass through the decoder
        reconstructed_x = self.decoder(latent_space)
        
        return reconstructed_x,mean, log_variance
