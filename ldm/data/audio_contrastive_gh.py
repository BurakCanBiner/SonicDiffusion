import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import pandas as pd
import torch


class Base(Dataset):
    def __init__(self,
                 audio_txt_file,
                labels_path,
                audio_data_root,
                config,
                caption_csv_path,
                landscape_dataset=False,
                **kwargs
                 ):
        
        self.audio_data_paths = audio_txt_file
        self.audio_data_root = audio_data_root
        self.landscape_dataset = landscape_dataset
        self.config = config

        with open(self.audio_data_paths, "r") as af:
            self.audio_paths = af.read().splitlines()
        
        self.labels_df = pd.read_csv(labels_path)
        #create a labels_dict such that key: filename, value: label by first assigning unique index to each label
        self.labels_df['label'] = self.labels_df['label'].astype('category').cat.codes

        self.labels_dict = dict(zip(self.labels_df['filename'], self.labels_df['label'])) #key: filename: e.g 034-13.wav , value: label (as index 0-9)
        #replace .png with .wav in keys
        # self.labels_dict = {k.replace('.png', '.wav'): v for k, v in self.labels_dict.items()}
        
        self.num_classes = len(set(self.labels_dict.values()))

        # Precompute class samples
        self.class_samples = {cls: [k for k, v in self.labels_dict.items() if v == cls] for cls in range(self.num_classes)} #for negative samples mostly it is 
        self.num_negative_samples = 128 #self.config.train.num_neg 

        if caption_csv_path != "":
            caption_csv = pd.read_csv(caption_csv_path)
            self.caption_dict = {}
            for i in range(len(caption_csv.index)):
                self.caption_dict[caption_csv.iloc[i,0].split("/")[-1].replace(".png","")] =  caption_csv.iloc[i,1].strip()
        else:
            print("caption dict is None")
            self.caption_dict = None
 

    def __getitem__(self, i):
        #one anchor, one positive, 2 negative samples from each class
        anchor = self.audio_paths[i].split("/")[-1].replace(".wav","") #to only get wav
        anchor_label = self.labels_dict[anchor]

        # print(f"anchor: {anchor}, anchor_label: {anchor_label}", flush=True)

        positive_sample = random.choice(self.class_samples[anchor_label])

        while positive_sample == anchor:
            positive_sample = random.choice(self.class_samples[anchor_label])

        # print(f"positive_sample: {positive_sample}", flush=True)
        pos_audio_ID = positive_sample    #positive_sample.replace('.wav', '')
        anchor_audio_ID = anchor     #anchor.replace('.wav', '')

        #choose randomly x negative samples from all other classes
        class_samples = {k: v for k, v in self.class_samples.items() if k != anchor_label}
        values = list(class_samples.values())
        values = [item for sublist in values for item in sublist]
        negative_samples = np.random.choice(values, size=self.num_negative_samples,replace=False)

        # anchor = [os.path.join(self.audio_path, anchor)]
        # positive_sample = [os.path.join(self.audio_path, positive_sample)]


        # # Read CLAP embeddings
        positive_file = os.path.join(self.config.train.clap_embeddings, f'{pos_audio_ID}.pt')
        anchor_file = os.path.join(self.config.train.clap_embeddings, f'{anchor_audio_ID}.pt')
        # #replace .wav with .pt
        negative_files = [os.path.join(self.config.train.clap_embeddings, f'{neg}.pt') for neg in negative_samples]
        
        # #load embeddings
        positive_emb = torch.load(positive_file, map_location=torch.device('cpu'))
        anchor_emb = torch.load(anchor_file, map_location=torch.device('cpu'))
        negative_emb = torch.stack([torch.load(neg, map_location=torch.device('cpu')) for neg in negative_files])

        if self.caption_dict is not None:
            caption = self.caption_dict[anchor_audio_ID]
        else:
            caption = ""

        return { 'anchor': anchor_emb, 'pos': positive_emb, 'neg': negative_emb, 'caption': caption} # 'anchor': anchor_emb, 'pos': positive_emb, 'neg': negative_emb,

    def __len__(self):
        return len(self.audio_paths)

class AudioTrain(Base):
    def __init__(self, audio_txt_file, config, caption_csv_path, audio_data_root, **kwargs):
        super().__init__(audio_txt_file=audio_txt_file,
                         config=config,
                         audio_data_root=audio_data_root, 
                         caption_csv_path=caption_csv_path,
                         **kwargs)

class AudioVal(Base):
    def __init__(self, audio_txt_file, config, audio_data_root, caption_csv_path,**kwargs):
        super().__init__(audio_txt_file=audio_txt_file,
                         config = config,
                         audio_data_root=audio_data_root, 
                         caption_csv_path=caption_csv_path,
                         **kwargs)        
