## HQGA: Video as Conditional Graph Hierarchy for Multi-Granular Question Answering

![teaser](https://github.com/doc-doc/HQGA/blob/main/introduction.png)
![](https://github.com/doc-doc/HQGA/blob/main/model.png)

## Todo
2. [ ] Release data of other datasets.

## Environment

Anaconda 4.8.4, python 3.6.8, pytorch 1.6 and cuda 10.2. For other libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n videoqa python==3.6.8
>conda activate videoqa
>git clone https://github.com/doc-doc/HQGA.git
>pip install -r requirements.txt
```
## Data Preparation
We use MSVD-QA as an example to help get farmiliar with the code. Please download the pre-computed features and trained models [here](https://drive.google.com/file/d/1bIWUqM9HtaJv2zDaEtz92v6UrTZtwGv9/view?usp=sharing)

After downloading the data, please create a folder ```['data/']``` at the same directory as ```['HQGA']```, then unzip the video and QA features into it. You will have directories like ```['data/msvd/' and 'HQGA/']``` in your workspace. Please move the model file ```[.ckpt]``` into ```['HQGA/models/msvd/']```. 


## Usage
Once the data is ready, you can easily run the code. First, to test the environment and code, we provide the prediction and model of the HQGA on MSVD-QA. 
You can get the results reported in the paper by running: 
```
>python eval_oe.py
```
The command above will load the prediction file under ['results/msvd/'] and evaluate it. 
You can also obtain the prediction by running: 
```
>./main.sh 0 test #Test the model with GPU id 0
```
The command above will load the model under ['models/msvd/'] and generate the prediction file.
If you want to train the model (Please follow our paper for details.), please run
```
>./main.sh 0 train # Train the model with GPU id 0
```
It will train the model and save to ['models/msvd']. 

## Citation
```

```

