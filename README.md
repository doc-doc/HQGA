#HQGA: Video as Conditional Graph Hierarchy for Multi-Granular Question Answering

![teaser](https://github.com/doc-doc/HQGA/blob/master/introduction.png)
![](https://github.com/doc-doc/HQGA/blob/master/model.png)

## Todo
2. [] Release data of other datasets.

## Environment

Anaconda 4.8.4, python 3.6.8, pytorch 1.6 and cuda 10.2. For other libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n videoqa python==3.6.8
>conda activate videoqa
>git clone https://github.com/doc-doc/HQGA.git
>pip install -r requirements.txt #may take some time to install
```
## Data Preparation
We use MSVD-QA as an example to help get farmilair with the code. Please download the pre-computed features and QA annotations from [here](https://drive.google.com/drive/folders/1gKRR2es8-gRTyP25CvrrVtV6aN5UxttF?usp=sharing). There are 4 zip files: 
- ```['vid_feat.zip']```: Appearance and motion feature for video representation. (With code provided by [HCRN](https://github.com/thaolmk54/hcrn-videoqa)).
- ```['qas_bert.zip']```: Finetuned BERT feature for QA-pair representation. (Based on [pytorch-pretrained-BERT](https://github.com/LuoweiZhou/pytorch-pretrained-BERT/)).
- ```['nextqa.zip']```: Annotations of QAs and GloVe Embeddings. 
- ```['models.zip']```: HGA model. 

After downloading the data, please create a folder ```['data/feats']``` at the same directory as ```['HQGA']```, then unzip the video and QA features into it. You will have directories like ```['data/msvd/' and 'HQGA/']``` in your workspace. Please move the model file ```[.ckpt]``` into ```['HQGA/models/msvd/']```. 


## Usage
Once the data is ready, you can easily run the code. First, to test the environment and code, we provide the prediction and model of the HQGA on MSVD-QA. 
You can get the results reported in the paper by running: 
```
>python eval_mc.py
```
The command above will load the prediction file under ['results/'] and evaluate it. 
You can also obtain the prediction by running: 
```
>./main.sh 0 val #Test the model with GPU id 0
```
The command above will load the model under ['models/'] and generate the prediction file.
If you want to train the model, please run
```
>./main.sh 0 train # Train the model with GPU id 0
```
It will train the model and save to ['models']. (*The results may be slightly different depending on the environments*)
## Results

## Citation
```

```
## Acknowledgement

