from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint
import wandb


def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., audio_dim=1024, 
                 orig_cond_token_count=77, use_adapter=False, adapter_token_count = 32, visualize=False):
        super().__init__()

        inner_dim = dim_head * heads
        self.visualize = visualize

        self.use_adapter = use_adapter
        self.orig_cond_token_count = orig_cond_token_count
        self.adapter_token_count = adapter_token_count

        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        self.context_dim = context_dim 
        self.inner_dim = inner_dim


        # HERE WE DEFINE ADAPTATION PROMPTS AND GATING VARIABLE 
        # GATE IS INITIALIZED AS ZERO
        # ADAPTER PROMPTS SHAPE IS K, CONTEXT_DIM 
        # WHICH CORRESPONDS TO K, 768 HERE


        # if self.use_adapter :
            # self.gate = torch.nn.Parameter(torch.zeros(1))
            # self.adapter_tokens = torch.nn.Parameter(torch.randn(adapter_token_count, context_dim ))

        # # K_V_RESIDUAL APPROACH
        # if self.use_adapter :
        #     inner_dim_adapter = 128
        #     audio_target_dim = 77
        #     # OUTPUTS ARE B, 77, CH -> WE CAN DIRECTLY ADD TO K AND V PROJECTIONS
        #     self.adapter_k_res = nn.Sequential(
        #         nn.Conv1d(1, audio_target_dim, 32, stride=8, bias=True, padding=12),
        #         nn.GELU(),
        #         torch.nn.Linear(inner_dim_adapter, inner_dim, bias=True)
        #     )
        #     self.adapter_v_res = nn.Sequential(
        #         nn.Conv1d(1, audio_target_dim, 32, stride=8, bias=True, padding=12),
        #         nn.GELU(),
        #         torch.nn.Linear(inner_dim_adapter, inner_dim, bias=True)
        #     )




#for visualization:
        self.attn = None
        self.q = None
        self.k = None
        self.v = None

        self.norm_dict = None

    def forward(self, x, context=None, mask=None, audio_context=None,  q_injected=None,
                k_injected=None):

        h = self.heads
        
        if q_injected is None:
            q = self.to_q(x)
        else:
            q = q_injected
            # print("injection to q done")

        context = default(context, x)
        
        if k_injected is None:
            k = self.to_k(context)
        else:
            # print("injection to k done")
            k = k_injected
        
        
        if self.visualize:
            self.q = q
            self.k = k
#             self.v = v
            
#             self.attn = sim
        
#         q = self.to_q(x)
#         context = default(context, x)

        ####
        # THIS IMPLEMENTATION OF THE LLAMA ADAPTER FOLLOWS THE INSTRUCTIONS IN THE PAPER
        # ONE OTHER WAY OF IMPLEMENTING THIS IS PASSING ADAPTATION PROMPTS FROM PROJECTION MATRICES
        # WITHOUT ANY SEPARATION. LLAMA SOURCE CODE IS IMPLEMENTED THAT WAY
        # THEY MUST OUTPUT SAME RESULTS, WE CAN DOUBLE CHECK LATER TO MAKE SURE
        ####

        
#         k = self.to_k(context)
        v = self.to_v(context)

        # # # K_V_RESIDUAL APPROACH
        # # # print(f"q shape is {q.shape} and k {k.shape} and v {v.shape} and x {x.shape}", flush=True)
        # if self.use_adapter and audio_context is not None:
        #     residual_k = self.adapter_k_res(audio_context)
        #     k += residual_k
        #     v += self.adapter_v_res(audio_context)



        # HERE WE INJECT NEW TOKENS INTO ATTENTION
#         if self.use_adapter and audio_context is not None:
# #             batch_size = x.shape[0]

# #             # adapter_prompts SHAPE HERE IS  K, CONTEXT_DIM -> BS, K, CONTEXT_DIM
# #             adapter_prompts = self.adapter_tokens.unsqueeze(0).repeat(batch_size, 1, 1)

# #             # AUDUO CONTEXT SHAPE HERE IS, BS, CONTEXT_DIM -> BS, K, CONTEXT_DIM
# #             audio_context = audio_context.unsqueeze(1).repeat(1, self.adapter_token_count, 1)


# #             adapter_prompts = adapter_prompts + audio_context


# # #             print("inside audio context cat ")
            
# # #             # CONTEXT SHAPE HERE IS BS, (77 + K), CONTEXT_DIM
# #             context = torch.cat((context, adapter_prompts), 1)
# #             context = torch.cat((context, audio_context), 1)

#             k_adapter = self.to_k(audio_context)
#             v_adapter= self.to_v(audio_context)
#             k_adapter, v_adapter = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_adapter, v_adapter))
#             sim_adapter = einsum('b i d, b j d -> b i j', q, k_adapter) * self.scale



#         if exists(mask):
#             mask = rearrange(mask, 'b ... -> b (...)')
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, 'b j -> (b h) () j', h=h)
#             sim.masked_fill_(~mask, max_neg_value)

#         if self.use_adapter and audio_context is not None :
# #             adapter_prompt_sim = sim[:,:,self.orig_cond_token_count:]  # LAST K TOKENS TO USE AS ADAPTATION PROMPTS
# #             sim = sim[:,:,:self.orig_cond_token_count]  # FIRST 77 TOKENS TO USE AS TEXT TOKENS 
# # #             adapter_attn = self.gate * adapter_prompt_sim.softmax(dim=-1) # WE APPLY GATING MECHANISM HERE 
# #             adapter_attn = adapter_prompt_sim.softmax(dim=-1) # WE APPLY GATING MECHANISM HERE 
#             attn_adapter = sim_adapter.softmax(dim=-1) 

        # attention, what we cannot get enough of
        
#         if self.use_adapter and audio_context is not None :
# #             attn = torch.cat((attn, adapter_attn), 2) # WE CONCAT TEXT TOKENS AND ADAPTER TOKENS BACK 
# #             v_text = v[:,:self.orig_cond_token_count,:]
# #             v_adapter = v[:,self.orig_cond_token_count:,:]
            
#             out_text = einsum('b i j, b j d -> b i d', attn, v)
#             out_audio = einsum('b i j, b j d -> b i d', attn_adapter, v_adapter)

#             out_adapter = out_text + out_audio
#             scale_out_norm = torch.norm(out_text)/torch.norm(out_adapter)
#             print(f" text norm {torch.norm(out_text)} out norm is {torch.norm(out_adapter)} torch audio norm {torch.norm(out_audio)} ")
            
#             out = out_adapter * scale_out_norm
#         else:
#             out = einsum('b i j, b j d -> b i d', attn, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, use_adapter=False,
                 adapter_token_count=32, injection_mode=False, visualize=False, adapter_dropout=0.0, unet_layer_name=None):
        super().__init__()

        self.injection_mode = injection_mode

        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, visualize=visualize)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, use_adapter=use_adapter, 
                                    adapter_token_count=adapter_token_count, visualize=visualize)  # is self-attn if context is none
        
        
        ## HERE WE DEFINE ANOTHER CROSS-ATTENTION LAYER TO CALCULATE ATTENTION OF AUDIO AND IMAGE -> FLAMINGO APPROACH
        self.use_adapter = use_adapter
        self.unet_layer_name = unet_layer_name
        #### dropout for this part: 0.1 for trainig and 0.0 for inference
        if self.use_adapter:
            self.attn_adapter = CrossAttention(query_dim=dim, context_dim=context_dim,
                                        heads=n_heads, dim_head=d_head, dropout=adapter_dropout, use_adapter=use_adapter, 
                                    adapter_token_count=adapter_token_count)  # is self-attn if context is none
            ### ANOTHER LAYER NORM FOR OUR ADAPTER -> FLAMINGO APPROACH
            self.norm_adapter = nn.LayerNorm(dim)
            self.gate_adapter = nn.Parameter(torch.tensor([0.]))
        
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        self.checkpoint = checkpoint
        self.norm_adapter_residual = None


    def forward(self, x, context=None, audio_context=None, flamingo_multiplier=1.0, self_attn_q_injected=None,
                 self_attn_k_injected=None, timestep=None):

        #  self_attn_q_injected,  self_attn_k_injected
        
        if self.injection_mode:
            if self_attn_q_injected is not None:
                return checkpoint(self._forward, (x, context, audio_context, flamingo_multiplier, timestep, self_attn_q_injected,  self_attn_k_injected), self.parameters(), self.checkpoint)
            elif audio_context is not None:
                return checkpoint(self._forward, (x, context, audio_context, flamingo_multiplier, timestep), self.parameters(), self.checkpoint)
            else:
                return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        else:
            if audio_context is not None:
                return checkpoint(self._forward, (x, context, audio_context), self.parameters(), self.checkpoint)
            else:
                return checkpoint(self._forward, (x, context ), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, audio_context=None, flamingo_multiplier=1.0, timestep=None, self_attn_q_injected=None,
                 self_attn_k_injected=None):

        x = self.attn1(self.norm1(x), q_injected=self_attn_q_injected,
                       k_injected=self_attn_k_injected) + x
        
        # K_V_RESIDUAL APPROACH -> audio context is given to the cross attention
        # FLAMINGO APPROACH  -> audio context is given as NONE to the cross attention
        # x = self.attn2(self.norm2(x), context=context, audio_context=audio_context) + x
        
        ## -> FLAMINGO APPROACH  
        x = self.attn2(self.norm2(x), context=context, audio_context=None) + x
    

        # if timestep is not None:
        #     print(f"timestep is {timestep}")
        #     norm_x = torch.norm(x).item()
        #     with open('logs/norm_x_only_text.txt', 'a') as file:
        #         # Write content tto the file
        #         file.write(f"{self.unet_layer_name} -t: {timestep} --> {norm_x}\n")
        # else:
        #     print("timestep is None")

        if self.use_adapter and audio_context is not None:
            adapter_residual = self.attn_adapter(self.norm_adapter(x), context=audio_context, audio_context=None)*self.gate_adapter.tanh()
            # self.norm_adapter_residual = torch.norm(adapter_residual).item()
            # norm_x = torch.norm(x).item()
            # log norm of adapter residual
            adapter_residual = adapter_residual*flamingo_multiplier
            
            # if timestep is not None:
            #     with open('logs/norm_new_ckpt2.txt', 'a') as file:
            #         # Write content tto the file
            #         file.write(f"{self.unet_layer_name} -t: {timestep} --> {self.norm_adapter_residual}\n")
            #     with open('logs/norm_x2.txt', 'a') as file:
            #         # Write content tto the file
            #         file.write(f"{self.unet_layer_name} -t: {timestep} --> {norm_x}\n")

            #flamingo_multiplier
            # adapter_norm  = torch.norm(adapter_residual)
            # features_norm = torch.norm(x)
            x =  adapter_residual + x

        # *self.gate_adapter.tanh()
        
        x = self.ff(self.norm3(x)) + x
        
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_adapter=False, adapter_token_count=32,
                injection_mode=False, visualize=False, adapter_dropout=0.0, unet_layer_name=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        
        self.unet_layer_name = unet_layer_name

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, use_adapter=use_adapter 
                                   , adapter_token_count=adapter_token_count, injection_mode=injection_mode, visualize=visualize,
                                   adapter_dropout=adapter_dropout, unet_layer_name=unet_layer_name)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, audio_context=None, self_attn_q_injected=None,
                self_attn_k_injected=None, flamingo_multiplier=1.0, timestep=None):
        
        # note: if no context is given, cross-attention defaults to self-attention

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context, audio_context=audio_context,
                      flamingo_multiplier=flamingo_multiplier, 
                      self_attn_q_injected=self_attn_q_injected,
                self_attn_k_injected=self_attn_k_injected, timestep=timestep)
            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in