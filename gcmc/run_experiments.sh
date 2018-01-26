#!/bin/bash

# Douban
python train.py -d douban --accum stack -do 0.7 -nleft -nb 2 -e 200 --features  --feat_hidden 64 --testing > douban_testing.txt  2>&1

# Flixster
python train.py -d flixster --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing > flixster_testing.txt  2>&1

# Yahoo Music
python train.py -d yahoo_music --accum stack -do 0.7 -nleft -nb 2 -e 200 --features --feat_hidden 64 --testing > yahoo_music_testing.txt  2>&1

# Movielens 100K on official split with features
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing > ml_100k_feat_testing.txt  2>&1

# Movielens 100K on official split without features
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --testing  > ml_1m_testing.txt  > ml_100k_testing.txt  2>&1

# Movielens 1M
python train.py -d ml_1m --data_seed 1234 --accum sum -do 0.7 -nsym -nb 2 -e 3500 --testing > ml_1m_testing.txt 2>&1

# Movielens 10M
python train_mini_batch.py -d ml_10m --data_seed 1234 --accum stack -do 0.3 -nsym -nb 4 -e 20 --testing > ml_10m_testing.txt 2>&1

