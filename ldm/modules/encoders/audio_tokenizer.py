import torch
import torch.nn as nn
import torch.nn.functional as F
# from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock

from torch import nn, einsum
from einops import rearrange, repeat


class AudioTokenizer(nn.Module):
    def __init__(self,  device="cuda", audio_dim = 1024, inner_dim = 20, target_dim = 768):
        super(AudioTokenizer, self).__init__()

        audio_att_inner_dim = inner_dim
        

        self.head = nn.Sequential(
            nn.Conv1d(1, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            nn.GELU(),
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            nn.GELU(),
            nn.LayerNorm([audio_att_inner_dim, audio_dim]),
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            nn.GELU(), 
            nn.LayerNorm([audio_att_inner_dim, audio_dim]),
            nn.ConvTranspose1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride=3, padding=7),
            nn.GELU(), 
            nn.LayerNorm([audio_att_inner_dim, 3*audio_dim]),
            nn.GELU(), 
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride=4, padding=7)
        )

        self.body = nn.ModuleList([nn.Sequential(
            nn.LayerNorm([audio_att_inner_dim, target_dim]),
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            nn.GELU(),
            nn.LayerNorm([audio_att_inner_dim, target_dim]),
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            nn.GELU(),
            nn.LayerNorm([audio_att_inner_dim, target_dim]),
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            ) for _ in range(4)])
        

        self.last_layer = nn.Sequential(
            nn.LayerNorm([audio_att_inner_dim, target_dim]),
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            )

    def forward(self, audio_context):
        x = self.head(audio_context)

        for block in self.body:
            x = block(x) + x
        
        return self.last_layer(x)


            
   
    
            