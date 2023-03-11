# Multitrack Music Transformer

## Prerequisites

### Set up development environment

We recommend using Conda. You can create the environment with the following command.

```sh
conda env create -f environment.yml
```

## Preprocessing

### Download the datasets

Please download the [Symbolic orchestral database (SOD)](https://qsdfo.github.io/LOP/database.html). You may also download in command line directly by `wget https://qsdfo.github.io/LOP/database/SOD.zip`.

> We also support the following two datasets:
>
> - [Lakh MIDI Dataset (LMD)](https://qsdfo.github.io/LOP/database.html)
>   - Download in command line directly via `wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz`
> - [SymphonyNet Dataset](https://symphonynet.github.io/)
>   - Download in command line directly via `gdown https://drive.google.com/u/0/uc?id=1j9Pvtzaq8k_QIPs8e2ikvCR-BusPluTb&export=download`

### Prepare the name list

Get a list of filenames for each dataset.

```sh
find data/sod/SOD -type f -name *.mid -o -name *.xml | cut -c 14- > data/sod/original-names.txt
```

> Note: Change the number in the cut command for different datasets.

### Convert the data

Convert the MIDI and MusicXML files into MusPy files for processing.

```sh
python convert_sod.py
```

> Note: You may enable multiprocessing via the `-j {JOBS}` option. For example, `python convert_sod.py -j 10` will run the script with 10 jobs.

### Extract the note list

Extract a list of notes from the MusPy JSON files.

```sh
python extract.py -d sod
```

### Split training/validation/test sets

Split the processed data into training, validation and test sets.

```sh
python split.py -d sod
```

## Training

Train a Multitrack Music Transformer model.

- Absolute positional embedding (APE):

  `python mmt/train.py -d sod -o exp/sod/ape -g 0`

- Relative positional embedding (RPE):

  `python mmt/train.py -d sod -o exp/sod/rpe --no-abs_pos_emb --rel_pos_emb -g 0`

- No positional embedding (NPE):

  `python mmt/train.py -d sod -o exp/sod/npe --no-abs_pos_emb --no-rel_pos_emb -g 0`

> Please run `python mmt/train.py -h` to see additional options.

## Evaluation

Evaluate the trained model.

```sh
python mmt/evaluate.py -d sod -o exp/sod/ape -ns 100 -g 0
```

> Please run `python mmt/evaluate.py -h` to see additional options.

## Generation (inference)

Generate new samples using a trained model.

```sh
python mmt/generate.py -d sod -o exp/sod/ape -g 0
```

> Please run `python mmt/generate.py -h` to see additional options.

## Citation

Please cite the following paper if you use the code provided in this repository.

Hao-Wen Dong, Ke Chen, Shlomo Dubnov, Julian McAuley and Taylor Berg-Kirkpatrick, "Multitrack Music Transformer: Learning Long-Term Dependencies in Music with Diverse Instruments," _Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_, 2023.
<br>
[[homepage](https://salu133445.github.io/mmt/)]
[[paper](https://arxiv.org/pdf/2207.06983.pdf)]
[[code](https://github.com/salu133445/mmt)]
[[reviews](https://salu133445.github.io/pdf/mmt-icassp2023-reviews.pdf)]
