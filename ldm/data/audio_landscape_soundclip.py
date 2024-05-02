import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

import librosa
# import glob
# import time
# import cv2
# import json


class Base(Dataset):
    def __init__(self,
                 img_txt_file,
                 audio_txt_file,
                 img_data_root,
                 audio_data_root,
                 size=256,
                 interpolation="bicubic",
                 flip_p=0.5,
                 track_len=441000
                 ):
        
        self.data_paths = img_txt_file
        self.data_root = img_data_root
        self.audio_data_paths = audio_txt_file
        self.audio_data_root = audio_data_root
        
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
            
        with open(self.audio_data_paths, "r") as af:
            self.audio_paths = af.read().splitlines()    
        
        self._length = len(self.image_paths)
        
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "audio_relative_file_path_": [l for l in self.audio_paths],
            "audio_file_path_": [os.path.join(self.audio_data_root, l)
                           for l in self.audio_paths],
        }
        
        self.track_len = track_len
        self.size = size
        
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        
#         paths_to_audio = "" # so we do this inside self.labels too ???
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32) 
        
        
#         track, _ = librosa.load(example["audio_file_path_"], dtype=np.float32)
#         print(example["audio_file_path_"],flush=True)
    
        track = np.load(example["audio_file_path_"], allow_pickle=True)    
#         temp_path = "/kuacc/users/bbiner21/Github/audioset-processing/audio_10s_npy/00wUFSUwR-g.npy"
#         track = np.load(temp_path, allow_pickle=True)
#         print(track.shape, flush=True)
    
        if len(track) >= self.track_len:
            track = track[:self.track_len]
        else:
            pad_amount = self.track_len - len(track)
#             print(pad_amount,flush=True)
#             example["audio"] = track
#             return example
            track = np.pad(track,((0,pad_amount),(0,0)),'constant')
            
            
#         print("after padding", flush=True)
#         print(track.shape, flush=True)

#         print(track.shape)
        if track.ndim > 1:
            track = np.mean(track,axis=1)
        
#         print(f" after dim reduce {track.shape}", flush=True)

        
        example["audio"] = track
        
        return example


    def __len__(self):
        return self._length
    
    
class AudioTrain(Base):
    def __init__(self, **kwargs):
        super().__init__(img_txt_file="/kuacc/users/bbiner21/Github/latent-diffusion/data/landscape/img_train.txt", audio_txt_file = "/kuacc/users/bbiner21/Github/latent-diffusion/data/landscape/audio_train.txt",img_data_root="/kuacc/users/bbiner21/Github/audioset-processing/landscape_frames", audio_data_root = "/kuacc/users/bbiner21/Github/audioset-processing/audio_10s_npy", **kwargs)
        
class AudioVal(Base):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(img_txt_file="data/landscape/img_val.txt", audio_txt_file = "data/landscape/audio_val.txt", img_data_root="/kuacc/users/bbiner21/Github/audioset-processing/landscape_frames", audio_data_root = "/kuacc/users/bbiner21/Github/audioset-processing/audio_10s_npy",
                         flip_p=flip_p, **kwargs)        
        