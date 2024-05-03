# Adapted from https://github.com/MichalGeyer/pnp-diffusers/blob/main/pnp.py

import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline

from pnp_utils import *

from unet2d_custom import UNet2DConditionModel
from pipeline_stable_diffusion_custom import StableDiffusionPipeline 
from ldm.modules.encoders.audio_projector_res import Adapter

# suppress partial model loading warning
logging.set_verbosity_error()

from diffusers import logging
logging.set_verbosity_error()

class PNP(nn.Module):
    def __init__(self, sd_version="1.4", n_timesteps=50, audio_projector_ckpt_path="ckpts/audio_projector_landscape.pth", 
                adapter_ckpt_path="ckpts/landscape.pt", device="cuda",
                clap_path="CLAP/msclap",
                clap_weights = "ckpts/CLAP_weights_2022.pth",
                 
                ):
        super().__init__()
        
        self.device = device
        
        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version == '1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
            print(f"model key is {model_key}")
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")

        model_id = "CompVis/stable-diffusion-v1-4"
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            use_adapter_list=[False, True, True],
            low_cpu_mem_usage=False,
            device_map=None
        ).to("cuda")
        

        gate_dict = torch.load(adapter_ckpt_path)
        
        for name, param in unet.named_parameters():
           if "adapter" in name:
            param.data = gate_dict[name]


        unet.to(self.device);
        
        pipe.unet = unet        
        
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(n_timesteps, device=self.device)

        self.latents_path = "latents_forward"
        self.output_path = "PNP-results/home"
        
    
        import os 
        import sys
        sys.path.append(clap_path)
        from CLAPWrapper import CLAPWrapper
        self.audio_encoder = CLAPWrapper(clap_weights, use_cuda=True)     

        
        self.audio_projector = Adapter(audio_token_count=77, transformer_layer_count=4).cuda()

        self.audio_projector.load_state_dict(torch.load(audio_projector_ckpt_path))
        self.audio_projector.eval()    
        self.sr = 44100
        
    def set_text_embeds(self, prompt, negative_prompt=""):
        self.text_embeds = self.get_text_embeds(prompt, negative_prompt)            
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]

    def set_audio_context(self, audio_path):
        audio_emb, _ = self.audio_encoder.get_audio_embeddings([audio_path], resample = self.sr)
        audio_proj = self.audio_projector(audio_emb.unsqueeze(1))

        audio_emb = torch.zeros(1, 1024).cuda()
        audio_uc = self.audio_projector(audio_emb.unsqueeze(1))    

        self.audio_context = torch.cat([audio_uc, audio_uc, audio_proj]).cuda()
        
        
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self, image_path):
        self.image_path = image_path
        # load image
        image = Image.open(image_path).convert('RGB') 
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        
        # get noise
        latents_path = os.path.join(self.latents_path, f'noisy_latents_{self.scheduler.timesteps[0]}.pt')
        noisy_latent = torch.load(latents_path).to(self.device)
        return image, noisy_latent

    @torch.no_grad()
    def denoise_step(self, x, t, guidance_scale):
        # register the time step and features in pnp injection modules
        source_latents = load_source_latents_t(t, os.path.join(self.latents_path))
        latent_model_input = torch.cat([source_latents] + ([x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, 
                               encoder_hidden_states=text_embed_input,
                              audio_context=self.audio_context)['sample']
        
        # perform guidance
        _, noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self, n_timesteps=50, pnp_f_t=0.5, pnp_attn_t=0.5,
                prompt="", negative_prompt="", 
                audio_path="", image_path="", 
                cfg_scale=5):
        
        self.set_text_embeds(prompt)     
        self.set_audio_context(audio_path=audio_path)
        self.image, self.eps = self.get_data(image_path=image_path)        
        
        
        pnp_f_t = int(n_timesteps * pnp_f_t)
        pnp_attn_t = int(n_timesteps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        
        edited_img = self.sample_loop(self.eps, cfg_scale=cfg_scale)

        return T.ToPILImage()(edited_img[0])      
        
        
    def sample_loop(self, x, cfg_scale):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t, cfg_scale)

            decoded_latent = self.decode_latent(x)
            T.ToPILImage()(decoded_latent[0]).save(f'{self.output_path}/output.png')
                
        return decoded_latent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config_pnp.yaml')
    
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(config["output_path"], exist_ok=True)
    
    with open(os.path.join(config["output_path"], "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    seed_everything(config["seed"])
    print(config)
    pnp = PNP(config)
    temp = pnp.run_pnp()