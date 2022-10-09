# -*- coding: utf-8 -*-
"""
@File   : utility.py    
@Contact: junwu302@gmail.com
@Usage  : xxxxxx
@Time   : 2022/9/1 18:16 
@Version: 1.0
@Desciption: utility functions
"""
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences

IUPAC_CODES = {'A': 0.984, 'C': 0.906, 'E': 1.094, 'D': 1.068, 'G': 1.031, 'F': 0.915, 'I': 0.927, 'H': 0.950,
               'K': 1.102, 'M': 0.952, 'L': 0.935, 'N': 1.048, 'Q': 1.037, 'P': 1.049, 'S': 1.046, 'R': 1.008,
               'T': 0.997, 'W': 0.904, 'V': 0.931, 'Y': 0.929, '*': 0}

BLOSUM_CODES = {'A':[4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,-2,-1,-1,-1,-4],
                'R':[-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,-1,-2,0,-1,-4],
                'N':[-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,4,-3,0,-1,-4],
                'D':[-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,4,-3,1,-1,-4],
                'C':[0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-1,-3,-1,-4],
                'Q':[-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,0,-2,4,-1,-4],
                'E':[-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,1,-3,4,-1,-4],
                'G':[0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,-1,-4,-2,-1,-4],
                'H':[-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,0,-3,0,-1,-4],
                'I':[-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,-3,3,-3,-1,-4],
                'L':[-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,-4,3,-3,-1,-4],
                'K':[-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,0,-3,1,-1,-4],
                'M':[-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,-3,2,-1,-1,-4],
                'F':[-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,-3,0,-3,-1,-4],
                'P':[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,-2,-3,-1,-1,-4],
                'S':[1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,0,-2,0,-1,-4],
                'T':[0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,-1,-1,-1,-1,-4],
                'W':[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,-4,-2,-2,-1,-4],
                'Y':[-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,-3,-1,-2,-1,-4],
                'V':[0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4,-3,2,-2,-1,-4],
                'B':[-2,-1,4,4,-3,0,1,-1,0,-3,-4,0,-3,-3,-2,0,-1,-4,-3,-3,4,-3,0,-1,-4],
                'J':[-1,-2,-3,-3,-1,-2,-3,-4,-3,3,3,-3,2,0,-3,-2,-1,-2,-1,2,-3,3,-3,-1,-4],
                'Z':[-1,0,0,1,-3,4,4,-2,0,-3,-3,1,-1,-3,-1,0,-1,-2,-2,-2,0,-3,4,-1,-4],
                'X':[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-4],
                '*':[-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,1]}

def load_fasta(in_fasta): # copy form esm module
    sequence_labels, sequence_strs, sequence_lens = [], [], []
    cur_seq_label = None
    buf = []
    def _flush_current_seq():
        nonlocal cur_seq_label, buf
        if cur_seq_label is None:
            return
        sequence_labels.append(cur_seq_label)
        strs = "".join(buf)
        sequence_strs.append(strs)
        sequence_lens.append(len(strs))
        cur_seq_label = None
        buf = []
    with open(in_fasta, "r") as infile:
        for line_idx, line in enumerate(infile):
            if line.startswith(">"):  # label line
                _flush_current_seq()
                line = line[1:].strip()
                if len(line) > 0:
                    cur_seq_label = line
                else:
                    cur_seq_label = f"seqnum{line_idx:09d}"
            else:
                buf.append(line.strip())
    infile.close()
    _flush_current_seq()
    assert len(set(sequence_labels)) == len(
        sequence_labels
    ), "Found duplicate sequence labels"
    return sequence_labels, sequence_strs, sequence_lens

def blosum_embedding(sequences):
    blosum_mat = []
    for seq in sequences:
        vec = []
        for s in seq[0:1000]: # trim to 1000bp
            val = BLOSUM_CODES.get(s)
            if(val is None):
                val = BLOSUM_CODES.get('*')
            else:
                vec.append(val)
        vec = np.array(vec, np.float32)
        seq_len = vec.shape[0]
        pad_left = int((1000-seq_len)/2)
        pad_right = 1000 - pad_left - seq_len
        blosum_mat.append(np.pad(vec, ((pad_left,pad_right),(0,0))))
    blosum_mat =  np.expand_dims(np.array(blosum_mat), axis=-1) # add channel
    return blosum_mat

# convert sequnce to digital vectors
def get_sentence(seq):
    sentences = []
    for s in seq:
        _sentence = re.findall("." * 3, s)
        _frac = int(len(s) / 3) - len(_sentence)
        if _frac > 0:
            _sentence.append('')
        sentences.append(_sentence)
    return sentences

def tokenizer(texts, word_index, trim_len = 1000):
    data = []
    for sentence in texts:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(word_index[word])  # index
            except:
                new_txt.append(0)
        data.append(new_txt)
    texts = pad_sequences(data, maxlen=trim_len, padding="post", truncating="post")  # padding
    return texts

def word2vec_embedding(sequences,word_index,trim_len = 1000):
    texts = get_sentence(sequences)
    print("loading fasttext model ...")
    word2vec_mat = tokenizer(texts, word_index, trim_len=trim_len)
    return word2vec_mat

