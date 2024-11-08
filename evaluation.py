import numpy as np
import torch
from PIL import Image

import os 
from glob import glob

import clip

import argparse 
import pandas as pd

import librosa
import wav2clip

import torchvision as tv
from omegaconf import OmegaConf


class ToTensor1D(tv.transforms.ToTensor):

    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])
        return tensor_2d.squeeze_(0)

@torch.no_grad()
def get_wav2clip_audio_embeddings(audio_paths, audio_encoder):

    tracks = []
    audio_transforms = ToTensor1D()
    final_audio_tensor_len = 442763

    for path_to_audio in audio_paths:
        track, _ = librosa.load(path_to_audio, sr=44100, dtype=np.float32)
        tracks.append(track)

    audio = torch.stack([torch.nn.ConstantPad1d((0,final_audio_tensor_len-track.shape[0]),0)(audio_transforms(track.reshape(1, -1))) for track in tracks])
    audio_features = torch.from_numpy(wav2clip.embed_audio(np.array(audio.squeeze(1)), audio_encoder)).to("cuda")

    return audio_features
    
@torch.no_grad()
def get_text_clip_embeddings(labels, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_features = []
    for label in labels:
        text = clip.tokenize([label]).to(device)
        with torch.no_grad():
            text_features.append(model.encode_text(text))

    return torch.stack(text_features, dim=0)

@torch.no_grad()
def get_clip_image_embeddings(edit_paths, clip_model, preprocess):
    
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    image_features_list = []
    with torch.no_grad():
        for ind, edited_path in enumerate(edit_paths):
                image_features_list.append(clip_model.encode_image(preprocess(Image.open(edited_path)).unsqueeze(0).to("cuda")).float())
    return torch.stack(image_features_list, dim=0)


def main(args):

    ## AUDIO LIST ###
    edit_config = OmegaConf.load(args.cfg_path)
    audio_list = edit_config.config.audio
    audio_list = sorted(audio_list)
    print("DONE READING AUDIO LIST LEN IS ", len(audio_list))

    ### CORRESPONDING DATASET IMAGES TO AUDIOS ###
    dataset_images_list = []
    parent_folder_dict = dict()

    for x in audio_list:
        audio_id = x.split("/")[-1].split(".")[0] 

        if args.dataset_type == "landscape":
            audio_parent_folder = x.split("/")[-2]
            print(f"audio id is {audio_id} and parent folder is {audio_parent_folder}")
            parent_folder_dict[audio_id] = audio_parent_folder

            if audio_parent_folder == "landscape-audios":
                image_parent_folder = "landscape-images"
            else:
                image_parent_folder = "yt-images"

            img_path = args.dataset_image_path + "/" + image_parent_folder+"/" + audio_id + ".png"
        elif args.dataset_type == "ravdess":
            audio_id = audio_id.split("_")[0]
            img_path = args.dataset_image_path + "/" + audio_id + "_frame0.png"
        else:
            img_path = args.dataset_image_path + "/" + audio_id + ".png"

        dataset_images_list.append(img_path)

    print("DONE READING DATASET IMAGES - LIST LEN IS ", len(dataset_images_list))
    
    ### GENERATED IMAGES ###
    edit_paths = sorted(glob(os.path.join(args.edited_images_path, "**", args.extension ), recursive=True))
    print("DONE READING GENERATED IMAGES - LIST LEN IS ", len(edit_paths))


    ### LABELS ###
    labels_df = pd.read_csv(args.labels_path)
    labels_df['filename'] = labels_df['filename'] + '.wav'
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))
    label_texts = sorted(list(set(labels_dict.values())))

    # REMOVE inversion images if any
    edit_paths = [x for x in edit_paths if "inversion" not in x]
    print("EDIT PATHS AFTER INVERSION REMOVAL LEN IS ", len(edit_paths))
    # Remove source images if any
    edit_paths = [x for x in edit_paths if "source" not in x.split("/")[-1].split("_")[0]]
    print("EDIT PATHS AFTER SOURCE REMOVAL LEN IS ", len(edit_paths))


    # create a properties dict for each image
    img_props_dict = {}
    for edit_path in edit_paths:
        audio_id = edit_path.split("/")[-1].split(".")[0]
        
        audio_id_wav = audio_id + ".wav"

        if args.dataset_type == "landscape":
            audio_path = os.path.join(args.dataset_audio_path, parent_folder_dict[audio_id], audio_id_wav )
        else:
            audio_path = os.path.join(args.dataset_audio_path, audio_id_wav)

        img_folder = edit_path.split("/")[-2]

        img_props = {}
        img_props["audio_path"] = audio_path
        img_props["audio_index"] = audio_list.index(audio_path)
        img_props["audio_class"] = labels_dict[audio_id_wav]
        img_props["audio_class_index"] = label_texts.index(img_props["audio_class"])
        img_props_dict[edit_path] = img_props


    ## NOT USING THIS PART SINCE WE CACHE DATASET EMBEDDINGS

    # ###  audio validation set txt ###
    # with open(args.validation_set_txt, "r") as f:
    #     validation_set = f.read().splitlines()
    
    # # append root path to validation set
    # validation_set = [os.path.join(args.dataset_audio_path, x) for x in validation_set]
    # print("VALIDATION SET LEN IS ", len(validation_set))

    # ###  image validation set txt ###
    # with open(args.validation_set_img_txt, "r") as f:
    #     validation_set_img = f.read().splitlines()

    # # append root path to validation set
    # validation_set_img = [os.path.join(args.dataset_image_path, x) for x in validation_set_img]
    # print("VALIDATION SET IMG LEN IS ", len(validation_set_img))


    # AUDIO EMBEDDINGS FROM WAV2CLIPIMAGES
    if args.wav2clip:
        print("Calculating Wav2clip score for audios" )
        audio_model = wav2clip.get_model()
        audio_model = audio_model.cuda()
        audio_model.eval()        

        audio_embeddings_wav2clip = get_wav2clip_audio_embeddings(audio_list, audio_model)
        audio_embeddings_wav2clip = audio_embeddings_wav2clip / audio_embeddings_wav2clip.norm(dim=-1, keepdim=True)
        print(f"wav2clip audo embeddings shape is {audio_embeddings_wav2clip.shape}")

        if args.dataset_type == "landscape":
            random_audio_embeddings_wav2clip_cat = torch.load("metrics_tensors/landscape_w2c.pt").cuda()
        elif args.dataset_type == "gh":
            random_audio_embeddings_wav2clip_cat = torch.load("metrics_tensors/gh_w2c.pt").cuda()
        else:
            random_audio_embeddings_wav2clip_cat = torch.load("metrics_tensors/ravdess_w2c.pt").cuda()


    # IMAGE EMBEDDINGS FROM VIT-B/32
    if args.wav2clip:
        clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
        image_embeddings_vit_b32 = get_clip_image_embeddings(edit_paths, clip_model, preprocess).squeeze(1)
        image_embeddings_vit_b32 = image_embeddings_vit_b32 / image_embeddings_vit_b32.norm(dim=-1, keepdim=True)
        print(f"image embeddings b_32 shape is {image_embeddings_vit_b32.shape}")
        
        image_embeddings_dataset_vit_b32 = get_clip_image_embeddings(dataset_images_list, clip_model, preprocess).squeeze(1)
        image_embeddings_dataset_vit_b32 = image_embeddings_dataset_vit_b32 / image_embeddings_dataset_vit_b32.norm(dim=-1, keepdim=True)
        print(f"image embeddings dataset b_32 shape is {image_embeddings_dataset_vit_b32.shape}")

    if args.iis or args.aic:
        clip_model, preprocess = clip.load("ViT-L/14", device="cuda")

        image_embeddings_edits_vit_l14 = get_clip_image_embeddings(edit_paths, clip_model, preprocess).squeeze(1)
        image_embeddings_edits_vit_l14 = image_embeddings_edits_vit_l14 / image_embeddings_edits_vit_l14.norm(dim=-1, keepdim=True)
        print(f"image embeddings l_14 shape is {image_embeddings_edits_vit_l14.shape}")

        image_embeddings_dataset_vit_l14 = get_clip_image_embeddings(dataset_images_list, clip_model, preprocess).squeeze(1)
        image_embeddings_dataset_vit_l14 = image_embeddings_dataset_vit_l14 / image_embeddings_dataset_vit_l14.norm(dim=-1, keepdim=True)
        print(f"dataset image embeddings shape is {image_embeddings_dataset_vit_l14.shape}")
    
        text_label_embeddings = get_text_clip_embeddings(label_texts, clip_model).squeeze(1).float()
        text_label_embeddings = text_label_embeddings / text_label_embeddings.norm(dim=-1, keepdim=True)
        print(f"text embeddings shape is {text_label_embeddings.shape}")

        if args.dataset_type == "landscape":
            random_image_embeddings_dataset_vit_l14_cat = torch.load("metrics_tensors/landscape_imgs.pt").cuda()
        elif args.dataset_type == "gh":
            random_image_embeddings_dataset_vit_l14_cat = torch.load("metrics_tensors/gh_imgs.pt").cuda()
        else:
            random_image_embeddings_dataset_vit_l14_cat = torch.load("metrics_tensors/ravdess_imgs.pt").cuda()

        #random_image_embeddings_dataset_vit_l14_cat = torch.load("metrics_tensors/landscape_imgs.pt").cuda()


    # CALCULATE SIMILARITY SCORES
    if args.wav2clip:
        # calculate all similarities
        wav2clip_score = torch.einsum('ij,kj->ik', image_embeddings_vit_b32, audio_embeddings_wav2clip)
        wav2clip_score = wav2clip_score.cpu().detach().numpy()

        # get right indices for each image and find similarties
        wav2clip_score_pair_scores = [wav2clip_score[img_ind][audio_ind] for img_ind, audio_ind in enumerate([img_props_dict[edited_img]["audio_index"] for edited_img in edit_paths])]

        # create a list for each image in edit_paths
        results_dict = {}
        for img_ind, _ in enumerate(edit_paths):
            results_dict[img_ind] = []

        random_wav2clip_scores = torch.einsum('ij,kj->ik', image_embeddings_vit_b32, random_audio_embeddings_wav2clip_cat)
        random_wav2clip_scores = random_wav2clip_scores.cpu().detach().numpy()

        val_set_len = random_audio_embeddings_wav2clip_cat.shape[0]

        for img_ind, _ in enumerate([img_props_dict[edited_img]["audio_index"] for edited_img in edit_paths]):
            for random_ind in range(val_set_len):
                if wav2clip_score_pair_scores[img_ind] > random_wav2clip_scores[img_ind][random_ind]:
                    results_dict[img_ind].append(1)
                else:
                    results_dict[img_ind].append(0)
        
        # take average of list in results_dict 
        wav2clip_ais_score = []
        for img_ind, _ in enumerate(edit_paths):
            wav2clip_ais_score.append(sum(results_dict[img_ind]) / len(results_dict[img_ind]))

        wav2clip_ais_score = sum(wav2clip_ais_score) / len(wav2clip_ais_score)
        print("WAV2CLIP AIS SCORE IS ", wav2clip_ais_score)


    if args.iis:
        clip_score_image_d = torch.einsum('ij,kj->ik', image_embeddings_edits_vit_l14, image_embeddings_dataset_vit_l14)
        clip_score_image_d = clip_score_image_d.cpu().detach().numpy()
        clip_score_image_d_pair_scores = [clip_score_image_d[img_ind][audio_ind] for img_ind, audio_ind in enumerate([img_props_dict[edited_img]["audio_index"] for edited_img in edit_paths])]

        # create a list for each image in edit_paths
        results_dict = {}
        for img_ind, _ in enumerate(edit_paths):
            results_dict[img_ind] = []

        random_clip_score_image_d = torch.einsum('ij,kj->ik', image_embeddings_edits_vit_l14, random_image_embeddings_dataset_vit_l14_cat)
        random_clip_score_image_d = random_clip_score_image_d.cpu().detach().numpy()

        val_set_len = random_image_embeddings_dataset_vit_l14_cat.shape[0]

        for img_ind, _ in enumerate([img_props_dict[edited_img]["audio_index"] for edited_img in edit_paths]):
            for random_ind in range(val_set_len):
                if clip_score_image_d_pair_scores[img_ind] > random_clip_score_image_d[img_ind][random_ind]:
                    results_dict[img_ind].append(1)
                else:
                    results_dict[img_ind].append(0)
        
        # take average of list in results_dict
        iss_scores = []
        for img_ind, _ in enumerate(edit_paths):
            iss_scores.append(sum(results_dict[img_ind]) / len(results_dict[img_ind]))

        score_iis = sum(iss_scores) / len(iss_scores)


    if args.aic:
        # calculate all similarities
        clip_score_text = torch.einsum('ij,kj->ik', image_embeddings_edits_vit_l14, text_label_embeddings)
        clip_score_text = clip_score_text.cpu().detach().numpy()
        pairs_list = np.argmax(clip_score_text, axis=1)

        # compare with gt labels
        text_classifier_score = 0.0
        for img_ind, audio_class_index in enumerate([img_props_dict[edited_img]["audio_class_index"] for edited_img in edit_paths]):
            if pairs_list[img_ind] == audio_class_index:
                text_classifier_score += 1

        text_classifier_score = text_classifier_score / len(edit_paths)



    ## SAVE SCORES TO FILE
    print(f"iis score at the end: {score_iis}")
    print(f"wav2clip ais score at the end: {wav2clip_ais_score}")
    print(f"text classifier score at the end: {text_classifier_score}")
    with open(os.path.join(args.edited_images_path, f"_scores_ref_based.txt"), "w") as f:
        f.write(f"wav2clip_ais_score: {wav2clip_ais_score}\n")
        f.write(f"iis_score: {score_iis}\n")
        f.write(f"text_classifier_score: {text_classifier_score}\n")



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--edited_images_path',  type=str, default="edited_image_path")
                      #"/kuacc/users/bbiner21/share_folder/edited_image_results/gh_default_setup_ep110_gh_source_2_pnp_configuration")
    args.add_argument('--ais' , action='store_true')
    args.add_argument('--iis' , action='store_true')
    args.add_argument('--aic' , action='store_true')
    args.add_argument('--dataset_image_path', type=str, default="/datasets/audio-image/images/landscape-images")
    args.add_argument('--dataset_audio_path', type=str, default="/datasets/audio-image/audios/landscape-audios")
    args.add_argument('--labels_path', type=str, default='data/greatest_hits/labels.csv')
    args.add_argument('--extension', type=str, default='*.png')
    args.add_argument('--dataset_type', type=str, default='landscape')


    args.add_argument('--wav2clip', action='store_true')
    args.add_argument('--clip_img_text', action='store_true')

    args.add_argument('--validation_set_txt', type=str, default='')
    args.add_argument('--validation_set_img_txt', type=str, default='')

    args.add_argument('--cfg_path', type=str, default="/kuacc/users/bbiner21/hpc_run/Github/Audio_stable_diffusion_v1/configs/feature_visualization/feature-extraction-gh.yaml")




    args = args.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(f"cuda device available {torch.cuda.is_available()}")

    main(args)
