import torch
import torch.nn as nn
import torch.nn.functional as F
# from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock

from torch import nn, einsum
from einops import rearrange, repeat


class Adapter(nn.Module):
    def __init__(self,  device="cuda", audio_dim = 1024, context_dim = 768, dropout=0.0, h = 8, dim_head=40):
        super(Adapter, self).__init__()
        self.h = h
        inner_dim = dim_head * h

        audio_att_inner_dim = 64
        self.audio_emb_projection = nn.Conv1d(1, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8)

        self.audio_emb_projection = nn.Sequential(
            nn.Conv1d(1, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            nn.GELU(),
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            nn.LayerNorm([audio_att_inner_dim, audio_dim]),
            nn.Conv1d(audio_att_inner_dim, audio_att_inner_dim, kernel_size = 17, stride = 1, padding = 8),
            # torch.nn.Linear(inner_dim_adapter, inner_dim, bias=True)
        )

        self.to_q_adapter = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_adapter = nn.Linear(audio_dim, inner_dim, bias=False)
        self.to_v_adapter = nn.Linear(audio_dim, inner_dim, bias=False)

        self.to_out_adapter = nn.Sequential(
        nn.Linear(inner_dim, context_dim),
        nn.Dropout(dropout)
        )

        self.scale = dim_head ** -0.5


    def forward(self, context, audio_context):

        audio_proj = self.audio_emb_projection(audio_context)

        q_adapter = self.to_q_adapter(context)
        k_adapter = self.to_k_adapter(audio_proj)
        v_adapter = self.to_v_adapter(audio_proj)

        q_adapter, k_adapter, v_adapter = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.h), (q_adapter, k_adapter, v_adapter))

        sim_adapter = einsum('b i d, b j d -> b i j', q_adapter, k_adapter) * self.scale
    
        attn_adapter = sim_adapter.softmax(dim=-1)

        out_adapter = einsum('b i j, b j d -> b i d', attn_adapter, v_adapter)
        out_adapter = rearrange(out_adapter, '(b h) n d -> b n (h d)', h=self.h)
        context = self.to_out_adapter(out_adapter)
            
   
        return context
    
            