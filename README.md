# Descrition

This is implementation neural network for cloud classification.

The base paper: https://www.researchgate.net/publication/326873190_CloudNet_Ground-Based_Cloud_Classification_With_Deep_Convolutional_Neural_Network

# Development

## Prepea

You need to use the dataset CCSN (Cirrus Cumulus Stratus Nimbus).
It may be downloaded from: 
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD.
Extract it to root of folder 'dataset'.
Example:

```bash
:/dataset# ls
Ac
As
...
```

## Train

```bash
./cloudnet.py --gpus 1 --max_epochs 20 --learning_rate 0.00001
```

## Inference

```bash
# example
./infer.py --model-path lightning_logs/version_0/checkpoints/model_last.onnx \
  --img dataset/As/As-N009.jpg
```

## Logs

Show logs:

```bash
tensorboard --bind_all --logdir ./
```

Run notebook

```bash
jupyter notebook --allow-root --ip 0.0.0.0
```

# Model

- [Pretrained model](https://disk.yandex.ru/d/P0G4jV4S5AFgkA)
- [Full train log](https://disk.yandex.ru/d/GGF0UJaOyvJUKw)

Train info:

- Learning rate: 1e-05
- Model: Resnet18 (pretrain)
- Epoch: 20
- Validation percent: 10%
- Acc train: 0.99
- Add valid: 0.53
