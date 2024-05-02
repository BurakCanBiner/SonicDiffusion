import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import random
import pandas as pd


class Base(Dataset):
    def __init__(self,
                 img_txt_file,
                 audio_txt_file,
                 img_data_root,
                 audio_data_root,
                 labels_path,
                 size=512,
                 interpolation="bicubic",
                 flip_p=0.5,
                 track_len=441000,
                 caption_csv_path = None,
                 clap_embeddigs_path=None,
                 augmentation_probs = [0.1, 0.0, 0.0, 0.2, 0.0, 0.1]
                 ):
        
        self.img_data_paths = img_txt_file
        self.img_data_root = img_data_root
        self.audio_data_paths = audio_txt_file
        self.audio_data_root = audio_data_root
        self.clap_embeddigs_path = clap_embeddigs_path
        
        with open(self.img_data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
            
        with open(self.audio_data_paths, "r") as af:
            self.audio_paths = af.read().splitlines()    
        
        self._length = len(self.image_paths)
        
#         self.labels = {
#             "relative_file_path_": [l for l in self.image_paths],
#             "file_path_": [os.path.join(self.data_root, l)
#                            for l in self.image_paths],
#             "audio_relative_file_path_": [l for l in self.audio_paths],
#             "audio_file_path_": [os.path.join(self.audio_data_root, l)
#                            for l in self.audio_paths],
#             # "caption": [l for l in self.captions],
#         }
        
                
        caption_csv = pd.read_csv(caption_csv_path)
        self.caption_dict = {}
        for i in range(len(caption_csv.index)):
            self.caption_dict[caption_csv.iloc[i,0].split("/")[-1]] =  caption_csv.iloc[i,1].strip()

        self.track_len = track_len
        self.size = size
        
        # interpolation
        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.labels_df = pd.read_csv(labels_path)
        #create a labels_dict such that key: filename, value: label by first assigning unique index to each label
        self.labels_df['label'] = self.labels_df['label'].astype('category').cat.codes

        self.labels_dict = dict(zip(self.labels_df['filename'], self.labels_df['label'])) #key: filename: e.g 034-13.wav , value: label (as index 0-9)

        self.t_Grayscale = transforms.RandomGrayscale(p=1.0)
        self.t_Blur = transforms.GaussianBlur((11, 11), sigma=(0.1, 5.0))
        self.t_Jitter = transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2))

        self.random_perspective = transforms.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation=self.interpolation, fill=0)
        self.random_rotation = transforms.RandomRotation(degrees=30, interpolation=self.interpolation, fill=0)

        self.cropper = transforms.RandomCrop(size=(512, 512))
        self.cropper_2 = transforms.RandomCrop(size=(640, 640))

        self.padder = transforms.Pad(padding=(128, 128, 128, 128), padding_mode='edge')
        self.center_croper = transforms.CenterCrop(size=(512, 512))

        # self.blur_prob_th = 0.1
        # self.jitter_prob_th = 0.2
        # self.gray_prob_th = 0.1
        # self.random_crop_prob_th = 0.15 
        # self.random_perspective_prob_th = 0.2
        # self.random_rotation_prob_th = 0.2 

        print(f"augmentation probs arae {augmentation_probs}")

        self.blur_prob_th = augmentation_probs[0]
        self.jitter_prob_th = augmentation_probs[1]
        self.gray_prob_th = augmentation_probs[2]
        self.random_crop_prob_th = augmentation_probs[3]
        self.random_perspective_prob_th = augmentation_probs[4]
        self.random_rotation_prob_th = augmentation_probs[5]


    def __getitem__(self, i):
        example = dict()
        file_id = self.audio_paths[i].split("/")[-1].replace(".wav","")
        example["file_id"] = file_id

        image_path = self.image_paths[i]

        image = Image.open(self.img_data_root +  "/" + image_path ) # + ".png"
        if not image.mode == "RGB":
            image = image.convert("RGB")


        using_random_crop = False
        random_crop_prob = random.uniform(0, 1)
        if random_crop_prob < self.random_crop_prob_th:
            randomp_cropper_pick = random.uniform(0, 1)
            if randomp_cropper_pick < 0.5:
                image = self.cropper(image)
            else:
                image = self.cropper_2(image)
            using_random_crop = True


        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img)


        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)


        # AUGMENTATIONS
        image = self.flip(image)
        blur_prob = random.uniform(0, 1)
        if blur_prob < self.blur_prob_th:
            image = self.t_Blur(image)

        using_jitter = False
        jitter_prob = random.uniform(0, 1)
        if jitter_prob < self.jitter_prob_th:
            image = self.t_Jitter(image)
            using_jitter = True

        gray_prob = random.uniform(0, 1)
        if gray_prob < self.gray_prob_th and not using_jitter :
            image = self.t_Grayscale(image)

        using_random_rotation = False
        random_perspective_prob = random.uniform(0, 1)
        if random_perspective_prob < self.random_perspective_prob_th and not using_random_crop:
            image = self.random_perspective(image)
            using_random_rotation = True

        random_rotation_prob = random.uniform(0, 1)
        if random_rotation_prob < self.random_rotation_prob_th and not using_random_rotation and not using_random_crop: 
            image = self.padder(image)
            image = self.random_rotation(image)
            image = self.center_croper(image)
                        


        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32) 
            


        example["input_text"] = self.caption_dict[file_id+".png"]
        

        audio_path = os.path.join(self.clap_embeddigs_path, file_id + ".pt")
        audio_emb = torch.load(audio_path, map_location=torch.device('cpu'))
        example["audio"] = audio_emb

        example["label"] = self.labels_dict[file_id ]

        example["audio_path"] = os.path.join(self.audio_data_root, self.audio_paths[i])

        # if p_random_flip:
        #     instance_image = TF.hflip(instance_image)
        # if p_jitter:
        #     instance_image = self.t_Jitter(instance_image)
        # if p_blur:
        #     instance_image = self.t_Blur(instance_image)
        # if p_grayscale:
        #     instance_image = self.t_Grayscale(instance_image)


        return example


    def __len__(self):
        return self._length
    
class AudioTrain(Base):
    def __init__(self, img_txt_file, audio_txt_file, caption_csv_path, img_data_root, audio_data_root, clap_embeddings_path, labels_path, augmentation_probs,**kwargs):
        super().__init__(img_txt_file=img_txt_file, audio_txt_file = audio_txt_file, img_data_root=img_data_root, 
        audio_data_root = audio_data_root, caption_csv_path=caption_csv_path, clap_embeddigs_path=clap_embeddings_path, labels_path=labels_path, augmentation_probs=augmentation_probs,**kwargs)
        

class AudioVal(Base):
    def __init__(self,  img_txt_file, audio_txt_file, caption_csv_path, img_data_root, audio_data_root, clap_embeddings_path, labels_path, augmentation_probs ,flip_p=0.,  **kwargs):
        super().__init__(img_txt_file=img_txt_file, audio_txt_file = audio_txt_file, img_data_root=img_data_root, 
        audio_data_root = audio_data_root, flip_p=flip_p, caption_csv_path=caption_csv_path, clap_embeddigs_path=clap_embeddings_path, labels_path=labels_path, augmentation_probs=augmentation_probs, **kwargs)        
        

