import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

# import librosa

import pandas as pd
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
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 track_len=441000,
                 label_csv_path = "landscape/landscape_final.csv",
                 caption_csv_path = None
                 ):
        
        self.data_paths = img_txt_file
        self.data_root = img_data_root
        self.audio_data_paths = audio_txt_file
        self.audio_data_root = audio_data_root
        
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
            
        with open(self.audio_data_paths, "r") as af:
            self.audio_paths = af.read().splitlines()    
        
        # with open(caption_path, "r") as cf:
        #     self.captions = cf.read().splitlines() 
        
        self._length = len(self.image_paths)
        
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "audio_relative_file_path_": [l for l in self.audio_paths],
            "audio_file_path_": [os.path.join(self.audio_data_root, l)
                           for l in self.audio_paths],
            # "caption": [l for l in self.captions],
        }
        
        # self.label_csv = pd.read_csv(label_csv_path)
        # self.label_dict = {}

        # for i in range(len(self.label_csv.index)):
        #     self.label_dict[self.label_csv.iloc[i,1]] = self.label_csv.iloc[i,0]
        
                
        caption_csv = pd.read_csv(caption_csv_path)
        self.caption_dict = {}
        for i in range(len(caption_csv.index)):
            self.caption_dict[caption_csv.iloc[i,0]] =  caption_csv.iloc[i,1].strip()


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
        
        image_id = example["relative_file_path_"].split('/')[1]
        image_id  = image_id[:-4]
#         example["input_text"] = "A photo of " +  self.label_dict[image_id]
        # example["input_text"] = example["caption"]
    
        audio_path =example["audio_file_path_"]  #  self.audio_data_root + 
#         audio_features = audio_encoder.get_audio_embeddings([audio_path], resample = 48000)
#         audio_features = audio_features.cuda()
        
#         track, _ = librosa.load(example["audio_file_path_"], dtype=np.float32)
# #         print(example["audio_file_path_"],flush=True)
    
#         track = np.load(example["audio_file_path_"], allow_pickle=True)    
# #         temp_path = "/kuacc/users/bbiner21/Github/audioset-processing/audio_10s_npy/00wUFSUwR-g.npy"
# #         track = np.load(temp_path, allow_pickle=True)
# #         print(track.shape, flush=True)
    
#         if len(track) >= self.track_len:
#             track = track[:self.track_len]
#         else:
#             pad_amount = self.track_len - len(track)
# #             print(pad_amount,flush=True)
# #             example["audio"] = track
# #             return example
#             track = np.pad(track,((0,pad_amount),(0,0)),'constant')
            
            
# #         print("after padding", flush=True)
# #         print(track.shape, flush=True)

# #         print(track.shape)
#         if track.ndim > 1:
#             track = np.mean(track,axis=1)
        
# #         print(f" after dim reduce {track.shape}", flush=True)

        example["input_text"] = self.caption_dict[example["file_path_"]]

        example["audio"] = audio_path
        
        return example


    def __len__(self):
        return self._length
    
class AudioTrain(Base):
    def __init__(self, img_txt_file, audio_txt_file, caption_csv_path, **kwargs):
        super().__init__(img_txt_file=img_txt_file, audio_txt_file = audio_txt_file, img_data_root="/datasets/audio-image/images", 
                         audio_data_root = "/datasets/audio-image/audios", caption_csv_path=caption_csv_path, **kwargs)
        
        # overfit_underwater_img, overfit_underwater_audio
        # dataset dataset_specific_split/image_train.txt
        # dataset dataset_specific_split/audio_train.txt

class AudioVal(Base):
    def __init__(self,  img_txt_file, audio_txt_file, caption_csv_path, flip_p=0., **kwargs):
        super().__init__(img_txt_file=img_txt_file, audio_txt_file = audio_txt_file, img_data_root="/datasets/audio-image/images", 
                         audio_data_root = "/datasets/audio-image/audios", flip_p=flip_p, caption_csv_path=caption_csv_path, **kwargs)        
        


# PREVIOUS data/mapper_small/img_train.txt
#          data/mapper_small/audio_train.txt
#          /kuacc/users/bbiner21/temp_data 
