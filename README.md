# Graph Convolutional Matrix Completion

Tensorflow based implemention of Graph Convolutional Matrix Completion for recommender systems, based on our paper:

Rianne van den Berg, Thomas N. Kipf, Max Welling, [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (2017)

## Installation

```python setup.py install```

## Requirements

  * Python 2.7
  * TensorFlow (1.4)
  * pandas


## Usage

To reproduce the experiments mentioned in the paper you can run the following commands:


**Douban**
```bash
python train.py -d douban --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing 
```

**Flixster**
```bash
python train.py -d flixster --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing
```

**Yahoo Music**
```bash
python train.py -d yahoo_music --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing
```

**Movielens 100K on official split with features**
```bash
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing
```

**Movielens 100K on official split without features**
```bash
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --testing
```

**Movielens 1M**
```bash
python train.py -d ml_1m --data_seed 1234 --accum sum -do 0.7 -nsym -nb 2 -e 3500 --testing
```

**Movielens 10M** 
```bash
python train_mini_batch.py -d ml_10m --data_seed 1234 --accum stack -do 0.3 -nsym -nb 4 -e 20 --testing 
```
Note: 10M dataset training does not fit on GPU memory (12 Gb), therefore this script uses a naive version of mini-batching.
Script can take up to 24h to finish.

## Cite

Please cite our paper if you use this code in your own work:

```
@article{vdberg2017graph,
  title={Graph Convolutional Matrix Completion},
  author={van den Berg, Rianne and Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1706.02263},
  year={2017}
}
```
