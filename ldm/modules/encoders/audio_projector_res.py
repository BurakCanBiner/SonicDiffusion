import torch
import torch.nn as nn
import torch.nn.functional as F
# from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock

from torch import nn, einsum
from einops import rearrange, repeat

#k,q will be from audio

class MyCrossAttention(nn.Module):
    def __init__(self,  device="cuda", audio_dim = 1024, context_dim = 768, dropout=0.0, h = 8, dim_head=40):
        super().__init__()
        self.h = h
        inner_dim = dim_head * h
        self.scale = dim_head ** -0.5

        self.to_q_adapter = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_adapter = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_adapter = nn.Linear(context_dim, inner_dim, bias=False)

    def forward(self, audio):
        q_adapter = self.to_q_adapter(audio) #from text
        k_adapter = self.to_k_adapter(audio)
        v_adapter = self.to_v_adapter(audio)

        q_adapter, k_adapter, v_adapter = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.h), (q_adapter, k_adapter, v_adapter))

        sim_adapter = einsum('b i d, b j d -> b i j', q_adapter, k_adapter) * self.scale
    
        attn_adapter = sim_adapter.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn_adapter, v_adapter)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.h)
        # print(f'ca out shape is: {out.shape}')

        return out


class Adapter(nn.Module):
    def __init__(self,  device="cuda", audio_dim = 1024, context_dim = 768, dropout=0.0, h = 8, dim_head=40, audio_token_count = 10, initial_channel_dim=1, transformer_layer_count=4):
        super(Adapter, self).__init__()
        self.h = h
        inner_dim = dim_head * h

        audio_att_inner_dim = audio_token_count

        self.audio_emb_projection = nn.Sequential(
            nn.Conv1d(initial_channel_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
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
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride=4, padding=7),
            nn.Dropout(dropout)
        )

        #create a stack of MyCrossAttention layers
        self.cross_attention = nn.ModuleList([MyCrossAttention(device, audio_dim, context_dim, dropout, h, dim_head) for _ in range(transformer_layer_count)])
        
        #create a stack of linear, gelu, linear dropout layers to be used after the cross attention
        self.between_attention = nn.ModuleList([nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, context_dim),
            nn.Dropout(dropout)
            ) for _ in range(transformer_layer_count)])

        self.to_out_adapter = nn.Sequential(
        nn.Linear(context_dim, context_dim),
        nn.Dropout(dropout)
        )
  

    def forward(self, audio_context):
        audio_proj = self.audio_emb_projection(audio_context) #[bs, 64, 1024]
        for cross_attention, between_attention in zip(self.cross_attention, self.between_attention):
            out = cross_attention(audio_proj)
            out = between_attention(out) + audio_proj
            # print(f'out shape is: {out.shape}')

        out = self.to_out_adapter(out) #[bs, 77, 768]
        # print(f'context dim is: {out.shape}')
   
        return out
    
            