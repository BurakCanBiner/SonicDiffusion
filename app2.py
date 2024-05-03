# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/app.py

import os
import json
import torch
import random

import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from PIL import Image

from unet2d_custom import UNet2DConditionModel
import torch 
from pipeline_stable_diffusion_custom import StableDiffusionPipeline 

from diffusers import DDIMScheduler
from pnp_utils import *
import torchvision.transforms as T
from preprocess import get_timesteps
from preprocess import Preprocess
        

from pnp import PNP

sample_idx     = 0

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 1.5em;
    min-width: 1.5em !important;
    height: 1.5em;
}
"""

class AnimateController:
    def __init__(self):
        self.sr = 44100
        self.save_steps = 50
        self.device = 'cuda'
        self.seed = 42
        self.extract_reverse = False
        self.save_dir = 'latents'
        self.steps = 50 
        self.inversion_prompt = ''
    
    
        self.seed = 42
        seed_everything(self.seed)
        
        self.pnp = PNP(sd_version="1.4")
        
        self.pnp.unet.to(self.device)
        self.pnp.audio_projector.to(self.device)        
        
    def preprocess(self, image=None):

        model_key = "CompVis/stable-diffusion-v1-4"
        toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        toy_scheduler.set_timesteps(self.save_steps)
        timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=self.save_steps,
                                                               strength=1.0,
                                                               device=self.device)
        
        save_path = os.path.join(self.save_dir + "_forward")
        os.makedirs(save_path, exist_ok=True)
        model = Preprocess(self.device, sd_version='1.4', hf_key=None)
        recon_image = model.extract_latents(data_path=image,
                                             num_steps=self.steps,
                                             save_path=save_path,
                                             timesteps_to_save=timesteps_to_save,
                                             inversion_prompt=self.inversion_prompt,
                                             extract_reverse=False)

        T.ToPILImage()(recon_image[0]).save(os.path.join(save_path, f'recon.jpg'))
    
    def generate(self,  file=None, audio=None, prompt=None, 
                 cfg_scale=5, image_path=None,
                 pnp_f_t=0.8, pnp_attn_t=0.8,):
            
        image = self.pnp.run_pnp(
                            n_timesteps=50, 
                            pnp_f_t=pnp_f_t, pnp_attn_t=pnp_attn_t, 
                            prompt=prompt,
                            negative_prompt="",
                            audio_path=audio, 
                            image_path=image_path,
                            cfg_scale=cfg_scale,
                           )        
        
        return image 
        
    def update_audio_model(self, audio_model_update):
        if audio_model_update == "Landscape Model":
            audio_projector_path = "ckpts/audio_projector_landscape.pth"
            gate_dict_path = "ckpts/landscape.pt"
        else:
            audio_projector_path = "ckpts/audio_projector_gh.pth"
            gate_dict_path = "ckpts/greatest_hits.pt"            
            
        gate_dict = torch.load(gate_dict_path)    
        for name, param in self.pnp.unet.named_parameters():
            if "adapter" in name:
                param.data = gate_dict[name]            

        self.pnp.audio_projector.load_state_dict(torch.load(audio_projector_path))            
        self.pnp.unet.to(self.device)
        self.pnp.audio_projector.to(self.device)
        
        return gr.Dropdown()
   
controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            # [SonicDiffusion: Audio-Driven Image Generation and Editing with Pretrained Diffusion Models]
            """
        )    
        with gr.Row():
            audio_input = gr.Audio(sources="upload", type="filepath")
            prompt_textbox = gr.Textbox(label="Prompt", lines=2)
            
        with gr.Row():
            with gr.Column():
                pnp_f_t = gr.Slider(label="PNP Residual Injection",    step=0.1, value=0.8, minimum=0.0,   maximum=1.0)
                pnp_attn_t = gr.Slider(label="PNP Attention Injection", step=0.1, value=0.8, minimum=0.0,  maximum=1.0)
                
            with gr.Column():
                audio_model_dropdown = gr.Dropdown(
                    label="Select SonicDiffusion model",
                    choices=["Landscape Model", "Greatest Hits Model"],
                    interactive=True,
                )
                
                audio_model_dropdown.change(fn=controller.update_audio_model, inputs=[audio_model_dropdown], outputs=[audio_model_dropdown])
                cfg_scale_slider = gr.Slider(label="CFG Scale", step=0.5, value=7.5, minimum=0,   maximum=20)
                

        with gr.Row():
            preprocess_button = gr.Button(value="Preprocess", variant='primary')            
            generate_button = gr.Button(value="Generate", variant='primary')

        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Input Image Component", sources="upload", type="filepath")

            with gr.Column():
                output = gr.Image(label="Output Image Component", 
                                 height=512, width=512)

        with gr.Row():
            
            examples = [
                [Image.open("assets/house.png")],
                [Image.open("assets/pineapple.png")],                
            ]
            gr.Examples(examples=examples,inputs=[image_input])
                
            examples2 = [
                ['./assets/fire_crackling.wav'],
                ['./assets/plastic_bag.wav'],                
            ]
            gr.Examples(examples=examples2,inputs=[audio_input])           
        
        
        preprocess_button.click(
            fn=controller.preprocess,
            inputs=[
                image_input
            ],
            outputs=output
        )        
    
        
        generate_button.click(
            fn=controller.generate,
            inputs=[
                audio_model_dropdown,
                audio_input,
                prompt_textbox,
                cfg_scale_slider,
                image_input,
                pnp_f_t,
                pnp_attn_t,
            ],
            outputs=output
        )
            
    
    return demo



if __name__ == "__main__":
    demo = ui()
    demo.launch(share=True)
    