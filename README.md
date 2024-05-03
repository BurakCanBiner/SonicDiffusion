# SonicDiffusion: Audio-Driven Image Generation and Editing with Pretrained Diffusion Models

<a href="https://arxiv.org/abs/2405.00878"><img src="https://img.shields.io/badge/arXiv-2307.08397-b31b1b.svg"></a> <a <a href="https://cyberiada.github.io/SonicDiffusion/"><img src="https://img.shields.io/badge/Project_Page-purple"></a>

>Official Implementation of SonicDiffusion: Audio-Driven Image Generation and Editing with Pretrained Diffusion Models.

<p align="center">
<img src="assets/teaser_image_v3.jpg"/>  
<br>
We present CLIPInverter that enables users to easily perform semantic changes on images using free natural text. Our approach is not specific to a certain category of images and can be applied to many different domains (e.g., human faces, cats, birds) where a pretrained StyleGAN generator exists (top). Our approach specifically gives more accurate results for multi-attribute edits as compared to the prior work (middle). Moreover, as we utilize CLIP’s semantic embedding space, it can also perform manipulations based on reference images without any training or finetuning (bottom).
</br>
</p>

## Updates
**03.05.2024**: Our code and demo are released.

## Demo


## Getting Started

### Prerequisites

```bash
$ conda create -n "myenv" python=3.11
$ git clone git@github.com:BurakCanBiner/SonicDiffusion.git
$ cd SonicDiffusion
$ pip install -r requirements.txt
```

### Pretrained Models

Pretrained models are available on the following links. 

| Path | Description
| :--- | :----------
|[Landscapes Audio Projector](https://drive.google.com/file/d/1ilIDUjGdScJD4UIG-cq3rKwW9yk5S2In/view?usp=sharing) |  should be named ckpts/audio_projector_landscapes.pt
|[Greatest Hits Audio Projector ](https://drive.google.com/file/d/1uoOsJcT0bC-_zNDbhcj6iaxLJBN-LFao/view?usp=sharing) |  should be named ckpts/audio_projector_gh.pt
|[Landscapes Adapter](https://drive.google.com/file/d/1kxYtrg4YQCudxL5f9xmCzOdJRITH5UXB/view?usp=share_link) | should be named ckpts/landscape.pt
|[Greatest Hits Adapter](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view) | should be named greatest_hits.pt


By default, we assume that all models are downloaded and saved to the directory `ckpts`.


## Citation

```
@misc{biner2024sonicdiffusion,
      title={SonicDiffusion: Audio-Driven Image Generation and Editing with Pretrained Diffusion Models}, 
      author={Burak Can Biner and Farrin Marouf Sofian and Umur Berkay Karakaş and Duygu Ceylan and Erkut Erdem and Aykut Erdem},
      year={2024},
      eprint={2405.00878},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```