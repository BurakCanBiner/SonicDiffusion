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

from ldm.modules.encoders.audio_projector_res import Adapter
from diffusers import logging
logging.set_verbosity_error()


sample_idx     = 0

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 1.5em;
    min-width: 1.5em !important;
    height: 1.5em;
}
"""


def get_pipe(gate_dict_path="ckpts/landscape.pt", 
            clap_path = "CLAP/msclap",
                clap_weights = 'ckpts/CLAP_weights_2022.pth',
             adapter_ckpt_path="ckpts/audio_projector_landscape.pth",
            ):
    
    model_id = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        use_adapter_list=[False, True, True],
        low_cpu_mem_usage=False,
        device_map=None
    ).to("cuda")
    
    
    gate_dict = torch.load(gate_dict_path)
    for name, param in unet.named_parameters():
        if "adapter" in name:    
            param.data = gate_dict[name]
        
    unet.to("cuda");


    pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
    pipeline.to("cuda")

    pipeline.unet = unet
    
    
    import sys
    sys.path.append(clap_path)
    from CLAPWrapper import CLAPWrapper

    audio_encoder = CLAPWrapper(clap_weights, use_cuda=True)     
    audio_projector = Adapter(audio_token_count=77, transformer_layer_count=4).cuda()
    audio_projector.load_state_dict(torch.load(adapter_ckpt_path))
    audio_projector.eval()    
    

    return audio_encoder, audio_projector, pipeline


class AnimateController:
    def __init__(self):
        self.audio_encoder, self.audio_projector, self.pipeline = get_pipe()
        self.sr = 44100
        self.device = "cuda"
        
    
    def generate(self,  file=None, audio=None, prompt=None, cfg_scale=5, num_inference_steps=50):
        with torch.no_grad():

            audio_emb, _ = self.audio_encoder.get_audio_embeddings([audio], resample = self.sr)
            audio_proj = self.audio_projector(audio_emb.unsqueeze(1))

            audio_emb = torch.zeros(1, 1024).cuda()
            audio_uc = self.audio_projector(audio_emb.unsqueeze(1))    

            audio_context = torch.cat([audio_uc, audio_proj]).cuda()
            image = self.pipeline(
                        prompt=prompt, 
                        audio_context=audio_context,
                        guidance_scale=cfg_scale,
                        num_inference_steps=num_inference_steps)

            return image.images[0]

        
    def update_audio_model(self, audio_model_update):
        if audio_model_update == "Landscape Model":
            audio_projector_path = "ckpts/audio_projector_landscape.pth"
            gate_dict_path = "ckpts/landscape.pt"
        else:
            audio_projector_path = "ckpts/audio_projector_gh.pth"
            gate_dict_path = "ckpts/greatest_hits.pt"            
            
        gate_dict = torch.load(gate_dict_path)    
        for name, param in self.pipeline.unet.named_parameters():
            if "adapter" in name:
                param.data = gate_dict[name]            

        self.audio_projector.load_state_dict(torch.load(audio_projector_path))            
        self.pipeline.unet.to(self.device)
        self.audio_projector.to(self.device)
        
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
            with gr.Column():            
            
                audio_input = gr.Audio(sources="upload", type="filepath")

                audio_model_dropdown = gr.Dropdown(
                    label="Select SonicDiffusion model",
                    value="Landscape Model",
                    choices=["Landscape Model", "Greatest Hits Model"],
                    interactive=True,
                )
                audio_model_dropdown.change(fn=controller.update_audio_model, inputs=[audio_model_dropdown], outputs=[audio_model_dropdown])         

                prompt_textbox = gr.Textbox(label="Prompt", lines=2)

                with gr.Row():            
                    cfg_scale_slider = gr.Slider(label="CFG Scale",        value=7.5, minimum=0,   maximum=20)
                    num_steps_slider = gr.Slider(label="Number of steps",        value=50, minimum=20,  maximum=200)
                generate_button = gr.Button(value="Generate", variant='primary')
                                
            
            with gr.Column():            
                output = gr.Image(label="Output Image Component", height=512, width=512)
            
                
        
        generate_button.click(
            fn=controller.generate,
            inputs=[
                audio_model_dropdown,
                audio_input,
                prompt_textbox,
                cfg_scale_slider,
                num_steps_slider,
            ],
            outputs=output
        )
        
        with gr.Row():
                
            examples2 = [
                ['./assets/fire_crackling.wav'],
                ['./assets/plastic_bag.wav'],                
            ]
            gr.Examples(examples=examples2,inputs=[audio_input])          
        
            
    return demo



if __name__ == "__main__":
    demo = ui()
    demo.launch(share=True)
    