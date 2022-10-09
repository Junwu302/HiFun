# -*- coding: utf-8 -*-
"""
@File   : predict.py    
@Contact: junwu302@gmail.com
@Usage  : xxxxxx
@Time   : 2022/9/4 16:06 
@Version: 1.0
@Desciption:
"""
import os
import logging
import click as ck
import pandas as pd
import numpy as np
from keras.models import load_model
from utility import load_fasta, blosum_embedding, word2vec_embedding
from models import focal_loss, auc_tensor
from keras_self_attention import SeqSelfAttention
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
logging.basicConfig(level=logging.INFO)
@ck.command()
@ck.option(
    '--in', '-i', default='',
    help='input proteins in fasta format.')
@ck.option(
    '--out', '-o', default='',
    help='Result file with a list of proteins and annotations')
def main(in_fasta, th = 0.20):
    # load query proteins
    protein_id, protein_seq, protein_len = load_fasta(in_fasta = in_fasta)
    # load pre-build models
    label_index =pd.read_pickle('db/goterms_level34.pkl')
    word_index = np.load('db/word_index.npy', allow_pickle=True).item()
    model = load_model('models/hifun_mode.h5',
                       custom_objects={'SeqSelfAttention': SeqSelfAttention,
                                       'auc_tensor': auc_tensor,
                                       'multi_category_focal_loss2_fixed': focal_loss(gamma=2., alpha=.25)})
    embeddings_matrix = np.load("db/embeddings_matrix.npy")
    # get embedding matrices
    blosum_mat = blosum_embedding(protein_seq)
    word2vec_mat = word2vec_embedding(protein_seq, word_index, trim_len=1000)
    # load models
    predict_probs = model.predict([word2vec_mat, blosum_mat], verbose=1)
    predict_terms = []
    predict_names = []
    predict_levels = []
    for prob in predict_probs:
        ind = np.argwhere(prob >= th).flatten().tolist()
        predict_terms.append(';'.join(label_index.iloc[ind,0].to_list()))
        predict_names.append(';'.join(label_index.iloc[ind,1].to_list()))
        predict_levels.append(';'.join(map(str,label_index.iloc[ind,2].to_list())))
    predict_res = pd.DataFrame({'Protein_id': protein_id, 'GO_terms': predict_terms,
                                'GO_names': predict_names, 'GO_levels': predict_levels})
    predict_res = pd.concat([predict_res, pd.DataFrame(predict_probs, columns=label_index['terms'])], axis = 1)
    return predict_res

#predict_res.to_csv('nonhomology_proteins_MFpred.csv', index=None)
if __name__ == '__main__':
    main()
