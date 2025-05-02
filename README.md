# CSE 7850 - Final Project

Author: Sizhe Fang, Xinyue Huang, Xuanzhang Liu, Yiling Wu

## Introduction
Building upon the CLEAN-Contact method, our approach leverages the protein language model ESM-2 to encode amino acid sequences and the convolutional neural network ResNet50 to process structural data derived from contact maps. The sequence inputs can be CSV or FASTA files. The structure inputs must be PDB files. 

Please note that due to the limitation of GitHub file, the data file is stored in the GT Box.

## Installation and Setup
### Requirements
Python == 3.10.13, PyTorch == 2.1.1, torchvision == 0.16.1;
fair-esm == 2.0.0, pytorch-cuda == 12.1

### Installation
1. Clone the code and start setting up the conda environment
    ```bash
    git clone https://github.com/liuxuanzhang718/7850FinalProject.git
    cd CLEAN-Contact
    conda create -n clean-contact python=3.10 -y
    conda activate clean-contact
    conda install -c conda-forge biopython biotite matplotlib numpy pandas pyyaml scikit-learn scipy tensorboardx tqdm
    ```
2. Install PyTorch and torchvision with CUDA
   * Find your operating system's installation method here: https://pytorch.org/get-started/locally/
3. Install fair-esm
    ```
    python -m pip install fair-esm==2.0.0
    python build.py install
    git clone https://github.com/facebookresearch/esm.git
    ```
### Setup
1. Create required folders:

    ```
   python
   >>> from src.CLEAN.utils import ensure_dirs
   >>> ensure_dirs()
    ```
2. Download the precomputed embeddings and distance map for both training and test data from 
[here](https://gatech.box.com/s/rimcplz8ldf0y6z254e13eye88jkyw29) 


## Pre-inference

Before running the inference step, extract the sequence representations and structure representations for your own data 
and then merge them. 

Sequence inputs can be in a CSV format or FASTA format and should be placed in the `data` folder. CSVs must have the 
columns: "Entry", "EC number", and "Sequence", where only "EC number" should be empty. 

Structure inputs must be in PDB format. CLEAN-Contact will grab the PDBs from the Alphafold database if the structure 
is available, otherwise use your own pre-generated PDB files as input. In either case create your PDB folder, such 
as <pdb-dir>, in the top level directory of CLEAN-Contact where extract_structure_representation.py is.

### Extract sequence and structure representations
#### Data in CSV format

For example, your `<csv-file>` is `data/split100_reduced.csv`. Then run the following commands: 

```bash
python extract_structure_representation.py \
    --input data/split100_reduced.csv \
    --pdb-dir <pdb-dir> 
```

```
python
>>> from src.CLEAN.utils import csv_to_fasta, retrieve_esm2_embedding
>>> csv_to_fasta('data/selected_10k.csv', 'data/selected_10k.fasta') # fasta file will be 'data/selected_10k.fasta'
>>> retrieve_esm2_embedding('selected_10k')
```

#### Data in FASTA format

For example, your `<fasta-file>` is `data/split100_reduced.fasta`. Then run the following commands:

```
python
>>> from src.CLEAN.utils import fasta_to_csv, retrieve_esm2_embedding
>>> fasta_to_csv('selected_10k')
>>> retrieve_esm2_embedding('selected_10k')
```

```bash
python extract_structure_representation.py \
    --input data/selected_10k.csv \
    --pdb-dir <pdb-dir> 
```

### Merge representations and compute distance map

Run the following commands to merge the sequence and structure representations:

```
python
>>> from src.CLEAN.utils import merge_sequence_structure_emb
>>> merge_sequence_structure_emb(<csv-file>)
```

If your data will be used as training data, run the following commands to compute distance map:

```
python
>>> from src.CLEAN.utils import compute_esm_distance
>>> compute_esm_distance(<csv-file>)
```

## Inference

If your dataset is in `csv` format, you can use the following command to inference the model:

```bash
python inference.py \
    --train-data selected_10k \
    --test-data selected_2k_testset \
    --method <method>
```

Replace `<test-data>` with your test data name, `<method>` with the `maxsep`, `pvalue`, `knn`, `filtered_hierarchical_kNN`, `integrated`, `integrated_triple` or `softmax`.

Performance metrics measured by Precision, Recall, F-1, and AUROC will be printed out. Per sample predictions will be saved in `results` folder.

## Training

Sequences whose EC number has only one sequence are required to mutated to generate positive samples. We provide the mutated sequences in `data/selected_10k.csv`. To get your own mutated sequences, run the following command:

```
python
>>> from src.CLEAN.utils import mutate_single_seq_ECs
>>> mutate_single_seq_ECs('selected_10k')
```

```bash
python mutate_conmap_for_single_EC.py \
    --fasta data/selected_10k.fasta 
```

```
python
>>> from src.CLEAN.utils import fasta_to_csv, merge_sequence_structure_emb
>>> fasta_to_csv('selected_10k')
>>> merge_sequence_structure_emb('selected_10k')
```

To train the model mentioned in the main text (`addition` model), modify arguments in `train-addition-triplet.sh` and run the following command:

```bash
./train-addition-triplet.sh
```

