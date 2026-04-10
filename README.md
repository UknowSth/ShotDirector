## <div align="center"> ShotDirector: Directorially Controllable Multi-Shot Video Generation with Cinematographic Transitions </div>

<div align="center">

<a href="https://arxiv.org/abs/2512.10286">
  <img src="https://img.shields.io/badge/Paper-arXiv-red" />
</a>
<a href="https://uknowsth.github.io/ShotDirector/">
  <img src="https://img.shields.io/badge/Project-Website-blue" />
</a>
<a href="https://huggingface.co/NumlockUknowSth/ShotDirector">
  <img src="https://img.shields.io/static/v1?label=Model&message=HuggingFace&color=yellow&logo=huggingface" />
</a>

</div>


### 🔥 Updates
- [x] Release [arXiv paper](https://arxiv.org/pdf/2512.10286) 
- [x] Release [project page](https://uknowsth.github.io/ShotDirector/)
- 🎉🎉🎉 Our work has been accepted to CVPR 2026!
- [x] Release inference code
- [x] Release model [checkpoints](https://huggingface.co/NumlockUknowSth/ShotDirector)
- [ ] Release Dataset ShotWeaver


### 📑 Introduction

<img width="1130" height="414" alt="Method" src="https://github.com/user-attachments/assets/8d1d69c9-0962-43c5-b4a6-348d2e40c4e1" />

We introduce **ShotDirector**, a controllable multi-shot video generation framework that models diverse cinematographic transition types by combining parameter-level camera control with editing-pattern-aware prompting. Through 6-DoF camera conditioning and a shot-aware mask mechanism, it enables intentional, film-like transitions beyond simple shot changes.

### 📥 Install

Clone the Repo

```
git clone https://github.com/UknowSth/ShotDirector.git
cd ShotDirector
```

Set up Environment
```
conda create -n shotdirector python==3.11.9
conda activate shotdirector

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 🤗 Checkpoint

Download the weights of [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) and the weights required for [Shotdirector](https://huggingface.co/NumlockUknowSth/ShotDirector). Place them in the `.ckpt/` folder as shown in the following diagram.

```
ckpt/
│── Wan2.1/Wan2.1-T2V-1.3B/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors
│   ├── google/  
│   │── models_t5_umt5-xxl-enc-bf16.pth
│   └── Wan2.1_VAE.pth
│── encoder.pt
│── model.pt
│── trans.pt
```

### 🖥️ Inference

Use the following instructions to perform model inference:

```
python generate.py
```

On the single A800, it takes 15 min to sample a video sample and requires 30GB.


### 🖼️ Gallery  

| ![A Kid](https://github.com/user-attachments/assets/4e6803d3-8411-4abb-9fc0-8701e79879d1) | ![Smitty](https://github.com/user-attachments/assets/a24ad3ec-4698-48ce-8f89-984b7c6f0a08) | ![Prince Andrew](https://github.com/user-attachments/assets/116fb53a-3c36-46d3-84d6-f04d7d985921) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Multi-Angle | Shot/Reverse Shot | Cut-in |
| ![Beethoven](https://github.com/user-attachments/assets/c9eac0ee-2ba2-429e-89b3-f1264ce095d2) | ![Incredible](https://github.com/user-attachments/assets/31a71ea3-89ce-4ad9-bfd2-61315b70c160) | ![Leatherheads](https://github.com/user-attachments/assets/e74c0724-6e67-4d45-a2f0-725627db4540) |
| Cut-out | Multi-Angle | Shot/Reverse Shot |


## 📖  BiTeX  
If you find [ShotDirector](https://github.com/UknowSth/ShotDirector.git) useful for your research and applications, please cite using this BibTeX:
```
@misc{wu2025shotdirectordirectoriallycontrollablemultishot,
      title={ShotDirector: Directorially Controllable Multi-Shot Video Generation with Cinematographic Transitions}, 
      author={Xiaoxue Wu and Xinyuan Chen and Yaohui Wang and Yu Qiao},
      year={2025},
      eprint={2512.10286},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.10286}, 
}
```




