# Multitrack Music Transformer

This repository contains the official implementation of "Multitrack Music Transformer" (ICASSP 2023).

__Multitrack Music Transformer__<br>
Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley and Taylor Berg-Kirkpatrick<br>
_IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, 2023<br>
[[homepage](https://salu133445.github.io/mmt/)]
[[paper](https://arxiv.org/pdf/2207.06983.pdf)]
[[code](https://github.com/salu133445/mmt)]
[[reviews](https://salu133445.github.io/pdf/mmt-icassp2023-reviews.pdf)]

## Content

- [Prerequisites](#prerequisites)
- [Preprocessing](#preprocessing)
  - [Preprocessed Datasets](#preprocessed-datasets)
  - [Preprocessing Scripts](#preprocessing-scripts)
- [Training](#training)
  - [Pretrained Models](#pretrained-models)
  - [Training Scripts](#training-scripts)
- [Evaluation](#evaluation)
- [Generation (Inference)](#generation-inference)
- [Citation](#citation)

## Prerequisites

We recommend using Conda. You can create the environment with the following command.

```sh
conda env create -f environment.yml
```

## Preprocessing

### Preprocessed Datasets

The preprocessed datasets can be found [here](https://drive.google.com/drive/folders/1owWu-Ne8wDoBYCFiF9z11fruJo62m_uK?usp=share_link). You can use [gdown](https://github.com/wkentaro/gdown) to download them via command line as follows.

```sh
gdown --id 1owWu-Ne8wDoBYCFiF9z11fruJo62m_uK --folder
```

Extract the files to `data/{DATASET_KEY}/processed/json` and `data/{DATASET_KEY}/processed/notes`, where `DATASET_KEY` is `sod`, `lmd`, `lmd_full` or `snd`.

### Preprocessing Scripts

__You can skip this section if you download the preprocessed datasets.__

#### Step 1 -- Download the datasets

Please download the [Symbolic orchestral database (SOD)](https://qsdfo.github.io/LOP/database.html). You may download it via command line as follows.

```sh
wget https://qsdfo.github.io/LOP/database/SOD.zip
```

We also support the following two datasets:

- [Lakh MIDI Dataset (LMD)](https://qsdfo.github.io/LOP/database.html):

  ```sh
  wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
  ```

- [SymphonyNet Dataset](https://symphonynet.github.io/):

  ```sh
  gdown https://drive.google.com/u/0/uc?id=1j9Pvtzaq8k_QIPs8e2ikvCR-BusPluTb&export=download
  ```

#### Step 2 -- Prepare the name list

Get a list of filenames for each dataset.

```sh
find data/sod/SOD -type f -name '*.mid' -o -name '*.xml' | cut -c 14- > data/sod/original-names.txt
```

> Note: Change the number in the cut command for different datasets.

#### Step 3 -- Convert the data

Convert the MIDI and MusicXML files into MusPy files for processing.

```sh
python mmt/convert_sod.py
```

> Note: You may enable multiprocessing with the `-j` option, for example, `python convert_sod.py -j 10` for 10 parallel jobs.

#### Step 4 -- Extract the note list

Extract a list of notes from the MusPy JSON files.

```sh
python mmt/extract.py -d sod
```

#### Step 5 -- Split training/validation/test sets

Split the processed data into training, validation and test sets.

```sh
python mmt/split.py -d sod
```

## Training

### Pretrained Models

The pretrained models can be found [here](https://drive.google.com/drive/folders/1HoKfghXOmiqi028oc_Wv0m2IlLdcJglQ?usp=share_link). You can use [gdown] to download all the pretrained models via command line as follows.

```sh
gdown --id 1HoKfghXOmiqi028oc_Wv0m2IlLdcJglQ --folder
```

### Training Scripts

Train a Multitrack Music Transformer model.

- Absolute positional embedding (APE):

  `python mmt/train.py -d sod -o exp/sod/ape -g 0`

- Relative positional embedding (RPE):

  `python mmt/train.py -d sod -o exp/sod/rpe --no-abs_pos_emb --rel_pos_emb -g 0`

- No positional embedding (NPE):

  `python mmt/train.py -d sod -o exp/sod/npe --no-abs_pos_emb --no-rel_pos_emb -g 0`

## Generation (Inference)

Generate new samples using a trained model.

```sh
python mmt/generate.py -d sod -o exp/sod/ape -g 0
```

## Evaluation

Evaluate the trained model using objective evaluation metrics.

```sh
python mmt/evaluate.py -d sod -o exp/sod/ape -ns 100 -g 0
```

## Acknowledgment

The code is based largely on the [x-transformers](https://github.com/lucidrains/x-transformers) library developed by [lucidrains](https://github.com/lucidrains).

## Citation

Please cite the following paper if you use the code provided in this repository.

 > Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley and Taylor Berg-Kirkpatrick, "Multitrack Music Transformer," _IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, 2023.

```bibtex
@inproceedings{dong2023mmt,
    author = {Hao-Wen Dong and Ke Chen and Shlomo Dubnov and Julian McAuley and Taylor Berg-Kirkpatrick},
    title = {Multitrack Music Transformer},
    booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    year = 2023,
}
```
