# UB-CSE635-2023SPRING 

University at Buffalo CSE 635: Natural Language Processing and Text Mining \
Spring 2023 \
Semester-Long Project \
Option 2: Fact Hallucinations Detection and Prevention \
by Changjae Lee 

## Requirements 

### Anaconda Environment 

If you don't have Anaconda distribution, please install it by refering to [this website](https://docs.anaconda.com/free/anaconda/install/index.html). 

```bash 
conda create -n hal python=3.8 
conda activate hal 

# PyTorch 
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia 
# Transformers (Huggingface) 
pip install transformers 
# Evaluate (Huggingface) 
pip install evaluate 
# BERT Score 
conda install -c conda-forge bert_score 
# ROUGE Score 
pip install rouge-score 
# Flair 
pip install flair 
``` 

### Data 

FaithDial dataset is necessary for this project. 
You can download the dataset on [this website](https://huggingface.co/datasets/McGill-NLP/FaithDial). 

`train.json`, `valid.json`, and `test.json` are expected to be under `/data`. 

After downloading, preprocessing is needed. 

```bash 
python data_preprocess.py 
``` 

There will be `hal_train.json`, `hal_valid.json`, and `hal_test.json` in `/data`. 

## Training 

### Baseliine 

**Milestone 2** 

```bash 
python baseline.py --t5_model t5-small --batch_size 64 --epochs 40 --max_input_length 512 --save_path _baseline_t5-small 
``` 

**Milestone 3** 

```bash 
python baseline.py --t5_model t5-small --batch_size 32 --epochs 50 --max_input_length 512 --save_path _baseline_t5-small 
``` 

### Baseline with POS 

First, we should build POS data. 

```bash 
python POS_data_create.py 
``` 

`train_pos.json`, `valid_pos.json`, and `test_pos.json` will be created under `/data`. 

Then, we can run 

```bash 
python baseline_with_POS.py --t5_model t5-small --batch_size 32 --epochs 50 --max_input_length 512 --save_path _baseline_pos_t5-small 
``` 

### Baseline with BERT 

```bash 
python baseline_with_BERT.py --t5_model t5-small --batch_size 32 --epochs 8 --max_input_length 64 --bert_max_input_length 512 --save_path _baseline_with_BERT_t5-small 
``` 

### Baseline with Contrastive Learning 

```bash 
python baseline_with_CLAPS.py --t5_model t5-small --batch_size 32 --epochs 50 --max_input_length 512 --save_path _baseline_with_CLAPS_t5-small 
``` 

## Evaluating 

There are `.ipynb` notebook files for evaluating models. 

### Baseliine 

`baseline_t5-small_test.ipynb` 

`test_baseline_t5-small_predicted.json` will be created under `/data`. 

### Baseline with POS 

`baseline_pos_t5-small_test.ipynb` 

`test_baseline_with_POS_t5-small_predicted.json` will be created under `/data`. 

### Baseline with BERT 

`baseline_with_BERT_t5-small_test.ipynb` 

`test_baseline_with_BERT_t5-small_predicted.json` will be created under `/data`. 

### Baseline with Contrastive Learning 

`baseline_with_CLAPS_t5-small_test.ipynb` 

`test_baseline_with_CLAPS_t5-small_predicted.json` will be created under `/data`. 

# References 

<a id="1">[1]</a> nunziati, “GitHub - nunziati/bert-vs-t5-for-question-answering: huggingface-based implementation of an open question answering model trained on the newsqa dataset.,” GitHub. https://github.com/nunziati/bert-vs-t5-for-question-answering 

<a id="2">[2]</a> seanie12, “GitHub - seanie12/CLAPS: [ICLR 2021] Contrastive Learning with Adversarial Perturbations for Conditional Text Generation,” GitHub. https://github.com/seanie12/CLAPS 

