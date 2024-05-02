from torch.utils.data import DataLoader
from ldm.data.audio_landscape import AudioTrain


from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os

from tqdm import tqdm

from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


import transformers
transformers.logging.set_verbosity_error()

import librosa
import cv2
import random
from collections import OrderedDict

# from ldm.modules.encoders.lavish_adapter import Adapter

from ldm.modules.encoders.lavish_adapter_mult import Adapter

from ldm.modules.encoders.audio_tokenizer import AudioTokenizer

#############################
# some helper functions 

# standard model load for ddpm 
def load_model_from_config(config, ckpt, device="cpu", verbose=False, start_attn_from_pretrained=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    model.to(device)
    # model.eval()
    model.cond_stage_model.device = device
    return model

# sampling without any backprop 
@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta=0.0, start_code=None, n_samples=1):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])

    shape = [4, h // 8, w // 8]
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                    )
    return samples_ddim

# image loading with path 
def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")
    
    
    tform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ])
    image = tform(image)
    return 2.*image - 1.

# latent to image
def decode_to_im(samples, n_samples=1, nrow=1):
    """Decode a latent and return PIL image"""
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(ims.cpu().numpy(), '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n_samples//nrow, n2=nrow)
    return Image.fromarray(x_sample.astype(np.uint8))

# sample image
# quick_sample = lambda x, s, code: decode_to_im(sample_model(model, sampler, x, h, w, ddim_steps, s, ddim_eta, start_code=code))

# class AudioMLP (torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(torch.nn.Linear(1024, 77*768), 
#         torch.nn.Unflatten(1,(77,768)) ,
#         torch.nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)).to("cuda")
#     def forward(self, x):
#         return self.model(x)
    

class AudioMLP (torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = torch.nn.Sequential(torch.nn.Linear(1024, 768), 
        torch.nn.GELU(),
        torch.nn.Linear(768, 768), 
        ).to("cuda")

    def forward(self, x):
        return self.projector(x)



#############################

# Paths
device = "cuda:0"
config="configs/stable-diffusion/v1-inference.yaml"
ckpt = "ckpts/sd-v1-4.ckpt" # this might cgange 

# audio encoder
audio_encoder_name = "clap"

# Generation parameters
scale=3
h=512
w=512
ddim_steps=45
ddim_eta=0.0

# feature dimension of audio encoder
feat_dim = 1024

# batch size 
bs = 6
clap_weights = '/kuacc/users/bbiner21/hpc_run/Github/CLAP/clap_weight/CLAP_weights_2022.pth'

mapper_lr = 0.001
adapter_lr = 1e-5
clap_lr = 1e-5


projector_lr = 1e-4
inside_model_lr = 1e-4

sr = 44100
reg_weight = 0.5
clip_loss_weight = 0.1

ep_save_freq = 10

token_count = "32"
token_cnt_int = 32
audio_duration = 10
window_len = 5
init_dim = 1

# ckpt_continue_path = "/kuacc/users/bbiner21/share_folder/audio_checkpoints/adapter_lavish_1e-3_clap_1e-4_encoder_decoder_inner_dim/ep0/"
# ckpt_ep_str = "ep0"
# ckpt_ep = 0
#############################
model = load_model_from_config(config, ckpt, device, start_attn_from_pretrained=True)

model.train()

# for name, module in model.named_modules():
#     if "adapter" in name:
#         module.load_state_dict(torch.load(ckpt_continue_path + name + ckpt_ep_str   + '.pth'))

# context_adapter = Adapter().cuda()
# audio_projector = AudioTokenizer().cuda()

# audio_projector = Adapter(audio_token_count=token_cnt_int, initial_channel_dim=init_dim).cuda()

audio_projector = AudioMLP()
audio_projector.train()

import sys
sys.path.append('/kuacc/users/bbiner21/hpc_run/Github/CLAP/src')
from CLAPWrapper import CLAPWrapper
audio_encoder = CLAPWrapper(clap_weights, use_cuda = True) #use_cuda=False

# audio_encoder.clap.load_state_dict(torch.load(ckpt_continue_path + "CLAP_" + ckpt_ep_str + ".pth" ))


# torch.save(audio_encoder.clap.state_dict(), ckpt_main_folder + folder + "/CLAP_ep" + str(i)  + '.pth')


for name, param in audio_encoder.clap.named_parameters():
    param.requires_grad = False

# for name, param in audio_encoder.clap.audio_encoder.named_parameters():
#     param.requires_grad = True

    


dt_audio = AudioTrain()
dl_audio = DataLoader(dt_audio, batch_size=bs, shuffle=True, num_workers=5, drop_last=True)

# audio_mapper = AudioMLP()

#############################
# audio_mapper = AudioMLP()
# # audio_mapper.load_state_dict(torch.load('audio_checkpoints/audio_mapper_small_data_ep95.pth'))

# optimizer_audio = torch.optim.Adam(audio_mapper.parameters(), lr=mapper_lr)





trainable = []
for name, param in model.named_parameters():
    if "gate" in name or "adapter_tokens" in name :
        trainable.append(param)
        

optimizer_adapter = torch.optim.Adam(trainable, lr=inside_model_lr)

optimizer_projector = torch.optim.Adam(audio_projector.parameters(), lr=projector_lr)



# optimizer_clap = torch.optim.Adam(audio_encoder.clap.audio_encoder.parameters(), lr=clap_lr)




# for name, param in audio_mapper.named_parameters():
#     param.requires_grad = False
#     if "adapter" in name:
#         trainable.append(param) 

epoch = 81

# opt = torch.optim.Adam([emb], lr=lr)
criteria = torch.nn.MSELoss()




# history = []

# first = next(iter(dl_audio))
# audio_emb = audio_encoder.get_audio_embeddings(first["audio"], resample = sr)
# audio_emb = audio_emb.cuda()

pbar = tqdm(range(epoch))
# label_cond = model.get_learned_conditioning("underwater bubbling")
# pooled_label = torch.mean(label_cond, dim=1)

for i in pbar:
    for ind, batch in enumerate(tqdm(dl_audio)):
        model.zero_grad()
        optimizer_adapter.zero_grad()
        optimizer_projector.zero_grad()

        # optimizer_clap.zero_grad()
        # optimizer_audio.zero_grad()

        init_img = torch.permute(batch["image"],(0,3,1,2)).cuda()
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_img))

        rand_int = np.random.randint(1,3)
        if  rand_int == 1:
            cond = model.get_learned_conditioning(batch["input_text"])
        elif rand_int == 2:
            cond = model.get_learned_conditioning(bs*[""])


        # else:
        #     cond = model.get_learned_conditioning(bs*[""])

#         new_cond = model.get_learned_conditioning(bs * [""])

        # cond = model.get_learned_conditioning(batch["input_text"])


        # audio_emb = audio_encoder.get_audio_embeddings(batch["audio"], resample = sr)

        audio_emb = audio_encoder.get_audio_embeddings(batch["audio"], audio_duration=audio_duration, window_len=window_len, 
        window_count=init_dim, resample = sr)

        audio_emb = audio_emb.reshape(bs,-1)
        audio_proj = audio_projector(audio_emb)

        # cond = torch.cat((cond, audio_proj), 1)
        # print(cond.shape)

        # noise = torch.randn_like(init_latent)
        # t_enc = torch.randint(1000, (bs,), device=device)
        # z = model.q_sample(init_latent, t_enc, noise=noise)

        l_pixel, loss_dict = model(init_latent, c=cond, audio_context = audio_proj)

        # cond = torch.cat((cond, audio_proj), 1)
        # pred_noise = model.apply_model(z, t_enc, cond, audio_context=audio_proj, 
        #                                features_adapter=None, att_map_residual_adapter = None )

        ## original loss objective
        # loss = criteria(pred_noise, noise)

        # pooled_cond = torch.mean(new_cond, dim=1)
        # print(f'loss_dict: {loss_dict}')
        # print(f'l_pixel: {l_pixel}')

        # loss_noise = criteria(pred_noise, noise)
        # loss_cond_reg = criteria(new_cond, cond)
        # similarity = torch.nn.functional.cosine_similarity(pooled_cond, pooled_label)
        # loss_clip = 1-similarity.mean()

        # loss = loss_noise + reg_weight*loss_cond_reg + clip_loss_weight*loss_clip

        # print(f"loss total {loss} :: loss noise {loss_noise} :: loss cond reg {loss_cond_reg} :: loss clip {loss_clip}")

        loss = l_pixel # if we define any other losses adding here 

        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
    #     history.append(loss.item())

        optimizer_projector.step()
        optimizer_adapter.step()

        # optimizer_clap.step()
        # optimizer_audio.step()

    if i % ep_save_freq == 0:
        ckpt_main_folder = f"/kuacc/users/bbiner21/share_folder/audio_checkpoints/adapter_overfit_underwater_LLama_learnable_tokens_K_32_window_5/"
        folder = "ep" + str(i)
        os.makedirs(ckpt_main_folder + folder, exist_ok = True)
        print(f"saving the model for epoch {i}")
        
        # torch.save(audio_encoder.clap.state_dict(), ckpt_main_folder + folder + "/CLAP_ep" + str(i)  + '.pth')
        
        # for name, param in model.named_parameters():
        #     if "gate" in name:
        #         torch.save(module.state_dict(), ckpt_main_folder + folder + "/" + name + "ep" + str(i)  + '.pth')

        param_dict = {}
        for name, param in model.named_parameters():
            if "gate" in name or "adapter_tokens" in name :
                param_dict[name] = param.data
                
        torch.save(param_dict, ckpt_main_folder + folder + "/" + "gate_adapter_dict_ep" + str(i)  + '.pth')


        # print("gate values are :::::")
        # for name, param in model.named_parameters():
        #     if "gate" in name :
        #         print(name)
        #         print(param.data)

        torch.save(audio_projector.state_dict(), ckpt_main_folder + folder + "/audio_projector_ep" + str(i)  + '.pth')
        
        # torch.save(audio_mapper.state_dict(), ckpt_main_folder + folder  + '/audio_mapper_joint_small_data_ep' + str(i) + '.pth')


# adapter = Adapter(sk=True).cuda(