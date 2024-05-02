"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

# ddim inversion from pix2pix-zero 
from random import randrange
import os

# import wandb

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        # self.use_wandb = True


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        
#         print(f" number of ddim num steps {ddim_num_steps}")
#         print(f" number of ddpm num steps {self.ddpm_num_timesteps}")
        
        
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
#         self.ddim_timesteps -= 1
        
#         print(self.ddim_timesteps)
#         print("inside make schedule")
        
        
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameter
#         print("sampling parameters", flush=True)
        
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    #@torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               audio_context=None,
               inject_before_time_step=1001,
               schedule_flamingo=-1,
               schedule_flamingo_upper_limit=1002,
               orig_context=None,
                injected_features=None,
                c_negative=None,
                flamingo_multiplier=1.0,
                audio_unconditional_conditioning=None,
                # use_wandb=True,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
#         print("inside sample")
#         print(len(features_adapter))
        
        # wandb.login()
        # wandb.init(project="adapter_test_ca_audio_flamingo", name="test")

        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # self.use_wandb = use_wandb
        # if use_wandb:

        #     wandb.init(project="audio-ddim", name="testing",entity="audio-ddim")
        #     wandb.config.update(kwargs)


        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')
        # print(f"inject before time step is {inject_before_time_step}")
        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    audio_context=audio_context,
                                                    inject_before_time_step=inject_before_time_step,
                                                    schedule_flamingo=schedule_flamingo,
                                                    schedule_flamingo_upper_limit=schedule_flamingo_upper_limit,
                                                    orig_context=orig_context,
                                                    injected_features=injected_features,
                                                    c_negative=c_negative,
                                                    flamingo_multiplier=flamingo_multiplier,
                                                    audio_unconditional_conditioning=audio_unconditional_conditioning,
                                                   )
        return samples, intermediates

    #@torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, audio_context=None,
                        inject_before_time_step=1001, orig_context=None, injected_features=None, c_negative=None, 
                        flamingo_multiplier=1.0, schedule_flamingo=-1, schedule_flamingo_upper_limit=1002, 
                        audio_unconditional_conditioning=None, **kwargs):
        # print(f"inject before time step is {inject_before_time_step}")
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
#         print(f" Time range is {time_range}")
        #for visualization. callback_ddim_timesteps_list = save_features_timesteps usually 50

        # if 50 is not None:
        callback_ddim_timesteps_list = np.flip(make_ddim_timesteps("uniform", 50 , self.ddpm_num_timesteps))
        # else:
        #     np.flip(self.ddim_timesteps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            
            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            injected_features_i = injected_features[i]\
                if (injected_features is not None and len(injected_features) > 0) else None

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      audio_context=audio_context,
                                      inject_before_time_step=inject_before_time_step,
                                      schedule_flamingo=schedule_flamingo,
                                      schedule_flamingo_upper_limit=schedule_flamingo_upper_limit,
                                      orig_context=orig_context,
                                      injected_features=injected_features_i,
                                      c_negative=c_negative,
                                      flamingo_multiplier=flamingo_multiplier,
                                      audio_unconditional_conditioning=audio_unconditional_conditioning,)
            
            # for name, module in self.model.named_modules():
            #     # if module has norm_adapter_residual property
            #     if hasattr(module, "norm_adapter_residual") and module.norm_adapter_residual is not None:
            #         print(f"step {step} norm of {name} is {module.norm_adapter_residual}", flush=True)

            #         wandb.log({f"norm_{name}": module.norm_adapter_residual}, step=step)


            # # check if self.model has a property called norm 
            # if self.use_wandb:
            #     for name, module in self.model.named_modules():
            #         if hasattr(module, "norm_adapter_residual") and module.norm_adapter_residual is not None:
            #             print(f"norm of {name} is {module.norm_adapter_residual}")
            #             wandb.log({f"norm_{name}": module.norm_adapter_residual}, step=t)

            img, pred_x0 = outs
            if step in callback_ddim_timesteps_list:
                if callback: callback(i)
                if img_callback: img_callback(pred_x0, i, step)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    #@torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, audio_context=None, 
                        inject_before_time_step=1001, orig_context = None, injected_features=None, c_negative = None, 
                        flamingo_multiplier=1.0, schedule_flamingo=-1, schedule_flamingo_upper_limit=1002, audio_unconditional_conditioning=None, **kwargs):
        b, *_, device = *x.shape, x.device
        
        # print(f"inject before time step is {inject_before_time_step}")
        # print(f"schedule flamingo is {schedule_flamingo}")
        # print(f"flamingo multiplier is {flamingo_multiplier}")

        # WE ONLY INJECT BEFORE A CERTAIN TIMESTAMP THIS WAY
        
        if t < inject_before_time_step:
            injected_features = None            
            # print("injected features is none")
        if t < schedule_flamingo:
            flamingo_multiplier = 0.0

        if t > schedule_flamingo_upper_limit:
            flamingo_multiplier = 0.0
            #print(f" current t is {t} below limit {schedule_flamingo_upper_limit}")

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, audio_context=audio_context, injected_features=injected_features, flamingo_multiplier=flamingo_multiplier)
        else:
            # x_in = torch.cat([x] * 2)
            # t_in = torch.cat([t] * 2)
            # c_in = torch.cat([unconditional_conditioning, c])
            if c_negative is not None:
                raise NotImplementedError
            else:
                if audio_unconditional_conditioning is not None:
                    e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, audio_context=audio_unconditional_conditioning, injected_features=injected_features, flamingo_multiplier=flamingo_multiplier)
                else:
                    e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, injected_features=injected_features, flamingo_multiplier=flamingo_multiplier)
                
                e_t = self.model.apply_model(x, t, c, audio_context=audio_context, injected_features=injected_features, flamingo_multiplier=flamingo_multiplier, timesteps_norm=t.item())
    #             e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, audio_context=audio_context).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    
    #pix2pix-zero
    def auto_corr_loss(self, x, random_shift=True):
        
        B,C,H,W = x.shape
        assert B==1
        x = x.squeeze(0)
        # x must be shape [C,H,W] now
        reg_loss = 0.0
        for ch_idx in range(x.shape[0]):
            noise = x[ch_idx][None, None,:,:]
            while True:
                if random_shift: roll_amount = randrange(noise.shape[2]//2)
                else: roll_amount = 1
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=2)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=roll_amount, dims=3)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = torch.nn.functional.avg_pool2d(noise, kernel_size=2)
        return reg_loss
    
    
    #pix2pix-zero 
    def kl_divergence(self, x):
        _mu = x.mean()
        _var = x.var()
        return _var + _mu**2 - 1 - torch.log(_var+1e-7)
    
    
    #pix2pix-zero
    def invert_p2p_zero(self, x, c, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, audio_context=None, flamingo_multiplier=1.0, ddim_use_original_steps = True):
        
        self.make_schedule(ddim_num_steps=50, ddim_eta=0, verbose=True)
        save_inds = [0,2,5,10,25,45]

        b, *_, device = *x.shape, x.device
        
        num_inversion_step = 50
#         time_range = reversed(range(0, num_inversion_step)) if ddim_use_original_steps else np.flip(num_inversion_step)
        step_len = 1000//num_inversion_step
        time_range = range(1, 1000, step_len)
        
        print(f"time range is {time_range}", flush=True)
        for index in tqdm(range(num_inversion_step-1)):
            step = time_range[index]            
            # b is 1 for now
            ts = torch.full((1,), step, device=device, dtype=torch.long)

            if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, ts, c)
            else:
#                 x_in = torch.cat([x] * 2)
#                 t_in = torch.cat([s] * 2)
#                 c_in = torch.cat([unconditional_conditioning, c])

                # audio context is none now while inverting the image 

#                 e_t_uncond = self.model.apply_model(x, ts, unconditional_conditioning)
                e_t = self.model.apply_model(x, ts, c)

               # #for inversion we do not apply classifier free guidance
#                 e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

            if score_corrector is not None:
                assert self.model.parameterization == "eps"
                e_t = score_corrector.modify_score(self.model, e_t, x, ts, c, **corrector_kwargs)

            num_reg_steps = 5
            lambda_ac = 20.0
            num_ac_rolls= 5
            lambda_kl = 20.0


            # regularization of the noise prediction
    #         e_t = noise_pred
            for _outer in range(num_reg_steps):
                if lambda_ac>0:
                    for _inner in range(num_ac_rolls):
                        _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                        l_ac = self.auto_corr_loss(_var)
                        l_ac.backward()
                        _grad = _var.grad.detach()/num_ac_rolls
                        e_t = e_t - lambda_ac*_grad
                if lambda_kl>0:
                    _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                    l_kld = self.kl_divergence(_var)
                    l_kld.backward()
                    _grad = _var.grad.detach()
                    e_t = e_t - lambda_kl*_grad
                e_t = e_t.detach()

    #         noise_pred = e_t            

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
            sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

            a_next = torch.full((b, 1, 1, 1), alphas[index+1], device=device)
            
            
            # current prediction for x_02
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
                                
            # direction pointing to x_t
#             dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            
            dir_xt = (1. - a_next).sqrt() * e_t
        
            # should be zero 
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            
#             if noise_dropout > 0.:
#                 noise = torch.nn.functional.dropout(noise, p=noise_dropout)
                
            x_next = a_next.sqrt() * pred_x0 + dir_xt + noise
        
#             x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        
            x = x_next
            
            if index in save_inds:
                torch.save(x, os.path.join("./", f"inversion/latent_ind_{index}.pt"))


        return x, pred_x0
    
# #     @torch.no_grad()
#     def encode_ddim_p2p(self, img, num_steps,conditioning, unconditional_conditioning=None ,unconditional_guidance_scale=1.):
        
#         print(f"Running DDIM inversion of pix2piz-zero with {num_steps} timesteps")
        
#         T = 975  # 999 originally 
        
#         c = T // num_steps
#         iterator = tqdm(range(0,T ,c), desc='DDIM Inversion', total= num_steps)
#         steps = list(range(0,T + c,c))

#         for i, t in enumerate(iterator):
#             img, _ = self.reverse_ddim(img, t, t_next=steps[i+1] ,c=conditioning, unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale)

#         return img, _    
    
    
    
#     @torch.no_grad()
    def encode_ddim(self, img, num_steps, conditioning, unconditional_conditioning=None ,unconditional_guidance_scale=1.,
                     p2p=True, img_callback=None, save_inversion_step_count = None, save_inv_step_size = 20, img_ind=-1):
        
        print(f"Running DDIM inversion with {num_steps} timesteps")
        
        T = 985  # 999 originally but gives out of index if num_steps is not 1000
        
        c = T // num_steps
        # iterator = tqdm(range(0,T ,c), desc='DDIM Inversion', total= num_steps)
        # steps = list(range(0,T + c,c))

        if save_inversion_step_count is not None:
            iterator = tqdm(range(0, save_inv_step_size*save_inversion_step_count  ,save_inv_step_size), 
                             desc='DDIM Inversion', total= save_inv_step_size)
            steps = list(range(0,T + save_inv_step_size,save_inv_step_size))
            
        else:
            iterator = tqdm(range(0,T ,c), desc='DDIM Inversion', total= num_steps)
            steps = list(range(0,T + c,c))

        print(f"steps are {steps}")
        print(f"iterator is {iterator}")
        
        for i, t in enumerate(iterator):
#             print(f"current t {t} and next_t is {steps[i+1]}")
            img, _ = self.reverse_ddim(img, t, t_next=steps[i+1] ,c=conditioning, unconditional_conditioning=unconditional_conditioning, unconditional_guidance_scale=unconditional_guidance_scale, p2p=p2p)

            if save_inversion_step_count is not None:
                if img_callback: img_callback(steps[i+1], img_ind)

        return img, _

#     @torch.no_grad()
    def reverse_ddim(self, x, t,t_next, c=None, quantize_denoised=False, unconditional_guidance_scale=1.,
                     unconditional_conditioning=None, p2p=True):
        b, *_, device = *x.shape, x.device
        with torch.no_grad():
            t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
            if c is None:
                e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
            elif unconditional_conditioning is None or unconditional_guidance_scale == 1.:
                e_t = self.model.apply_model(x, t_tensor, c)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_tensor] * 2)
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)


        num_reg_steps = 5
        lambda_ac = 20.0
        num_ac_rolls= 5
        lambda_kl = 20.0

        if p2p :
            # regularization of the noise prediction
    #         e_t = noise_pred
#             print("inside p2p loss")
            for _outer in range(num_reg_steps):
                if lambda_ac>0:
                    for _inner in range(num_ac_rolls):
                        
#                         var_xs_h = Variable(xs_h.data, requires_grad=True)

                        _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                        l_ac = self.auto_corr_loss(_var)
        
#                         print(f" loss auto correlation {l_ac}")
        
                        l_ac.backward()
                        _grad = _var.grad.detach()/num_ac_rolls
                        e_t = e_t - lambda_ac*_grad
                if lambda_kl>0:
                    _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                    l_kld = self.kl_divergence(_var)
                    l_kld.backward()
                    _grad = _var.grad.detach()
                    e_t = e_t - lambda_kl*_grad
                e_t = e_t.detach()
            
            
        with torch.no_grad():    
            alphas = self.model.alphas_cumprod #.flip(0)
            sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod #.flip(0)
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[t], device=device)
            a_next = torch.full((b, 1, 1, 1), alphas[t_next], device=device) #a_next = torch.full((b, 1, 1, 1), alphas[t + 1], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[t], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1. - a_next).sqrt() * e_t
            x_next = a_next.sqrt() * pred_x0 + dir_xt
            return x_next, pred_x0   
    
    #@torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    #@torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec