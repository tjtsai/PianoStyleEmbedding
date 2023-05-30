# PianoStyleEmbedding

This repository contains the code for the ISMIR 2020 paper "Composer Style Classification of Piano Sheet Music Images Using Language Model Pretraining."

The goal of this project is to predict the composer of an unseen page (image) of piano sheet music.

You can find the paper [here](https://drive.google.com/file/d/19jHQnAE8dCCFqy0un7W4HG2yvb4_gwgx/view?usp=sharing) and the pretrained models [here](https://drive.google.com/drive/folders/1Y-u3p8z5blISM06U7TXZaLLu52opoND_).

## Usage

### Environment Setup
Start by creating two environments. We recommend using [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) as it will greatly speed up the installation process, but conda also works.

`PianoStyleEmbedding_Prep` from [`environment_prep.yaml`](environment_prep.yaml) and `PianoStyleEmbedding_Train` from [`environment_train.yaml`](environment_train.yaml) to be used for data preparation and training respectively.

Using micromamba an environment can be created from a file using
```
micromamba create -f [env.yaml]
```

### Running the code
In general, the jupyter notebooks should be run in increasing order by name.

Note that notebooks [00]-[02] should be run in the `PianoStyleEmbedding_Prep` environment while notebooks [03]-[07] should be run in the `PianoStyleEmbedding_Train` environment.

Notes for specific notebooks are included below:
- 00_downloadScores
    - this notebook requires a *paid* imslp account to use. Without one it will not throw an error, but will just download empty PDFs that cause future notebooks to fail
- 01_extractFeatures
    - this notebook uses ImageMagick to convert PDFs to images. This process is very innefficient (it takes hours and uses up 100s of GB of disk space). There are instructions on how to change the temp storage directory in the notebook.
    - Here are instructions on how to [install](https://sites.google.com/g.hmc.edu/mir-lab-wiki/faq/install-imagemagick-locally?authuser=0) and [debug](https://sites.google.com/g.hmc.edu/mir-lab-wiki/faq/converting-pdfs-to-pngs-with-imagemagick?authuser=0) 
    - For future projects, we have switched to Ghostscript which is much faster and more efficient (https://sites.google.com/g.hmc.edu/mir-lab-wiki/faq/converting-pdfs-to-pngs-with-ghostscript?authuser=0)
- 04_roberta_lm
    - this should be run before any other of the 04 notebooks. It trains and saves the model that they use
- 05_gpt2_lm
    - this should be run before any other of the 04 notebooks. It trains and saves the model that they use

## Citation

TJ Tsai and Kevin Ji. "Composer Style Classification of Piano Sheet Music Images Using Language Model Pretraining" in Proceedings of the International Society for Music Information Retrieval Conference, 2020, pp. 176-183.
