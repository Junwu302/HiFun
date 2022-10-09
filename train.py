# -*- coding: utf-8 -*-
"""
@File   : train.py
@Contact: junwu302@gmail.com
@Usage  : xxxxxx
@Time   : 2022/9/1 16:18 
@Version: 1.0
@Desciption:
"""
import fasttext
import pandas as pd
import numpy as np
from models import train_model
from utility import blosum_embedding, word2vec_embedding
fasttext.FastText.eprint = lambda x: None
def train():
    # load traing data
    train_mf = pd.read_pickle('db/uniprot_bacteria_mf.pkl')
    # extracting labels
    train_mf_label = np.array(train_mf.labels.tolist())
    word_index = np.load('db/word_index.npy', allow_pickle=True).item()
    embeddings_matrix = np.load("db/embeddings_matrix.npy")

    # get embedding matrices
    blosum_mat = blosum_embedding(train_mf.sequence)
    word2vec_mat = word2vec_embedding(train_mf.sequence, word_index, trim_len=1000)

    print("Traing HiFun model ...")
    mod_file = "models/hifun_mode.h5"
    train_model(train_word2vec=word2vec_mat, train_blosum=blosum_mat, embeddings_matrix=embeddings_matrix,
                train_labels = train_mf, model_file=mod_file)
