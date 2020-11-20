# self-supervised-few-shot-medical-image-segmentation
the project implements few-shot image segmentation algorithm introduced in paper "self-supervised fewshot medical image segmentation" with TF2

## dataset preparation

download CHAOTS [here](https://chaos.grand-challenge.org/). unzip the CHAOTS_Train_Sets.zip and CHAOTS_Test_Sets.zip and execute the following command.

```python3
python3 create_datasets.py CHAOST2 </path/to/trainset> </path/to/testset>
```

upon executing the script successfully, there will be a directory named datasets containing trainset and testset.

## train with superpixel dataset

train with the following command

```python3
python3 train.py
```
