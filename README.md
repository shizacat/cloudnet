# Descrition

This is implementation neural network for cloud classification.

The base paper: https://www.researchgate.net/publication/326873190_CloudNet_Ground-Based_Cloud_Classification_With_Deep_Convolutional_Neural_Network

# Development

## Prepea

You need to use the dataset CCSN (Cirrus Cumulus Stratus Nimbus).
It may be downloaded from: 
[https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD]
(https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CADDPD).
Extract it to root of folder 'dataset'.
Example:

```bash
:/dataset# ls
Ac
As
...
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