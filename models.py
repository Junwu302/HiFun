# -*- coding: utf-8 -*-
"""
@File   : models.py    
@Contact: junwu302@gmail.com
@Usage  : xxxxxx
@Time   : 2022/9/2 21:58 
@Version: 1.0
@Desciption:
"""
import datetime
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model, optimizers
from keras import backend as K
from keras_self_attention import SeqSelfAttention
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
from sklearn.utils.class_weight import compute_class_weight

#### focal loss is application to  imbalance data
def focal_loss(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)尝试将alpha调大,鼓励模型进行预测出1。
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)
    def _focal_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss
    return _focal_loss

####### metrics  #######
def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    _recall = true_positives / (possible_positives + K.epsilon())
    return _recall

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    _precision = true_positives / (predicted_positives + K.epsilon())
    return _precision

def f1(y_true, y_pred):
    _precision = precision(y_true, y_pred)
    _recall = recall(y_true, y_pred)
    return 2 * ((_precision * _recall) / (_precision + _recall + K.epsilon()))

# mcc
def mcc(y_true, y_pred):
    y_pred_pos = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.30), 'float32')
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.cast(K.greater(K.clip(y_true, 0, 1), 0.30), 'float32')
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

def auc(y_true, y_pred):
    def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        N = K.sum(1 - y_true)
        FP = K.sum(y_pred - y_pred * y_true)
        return FP / N

    def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        P = K.sum(y_true)
        TP = K.sum(y_pred * y_true)
        return TP / P
    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)

def auc_tensor(y_true, y_pred):
    def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.3)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        N = K.sum(1 - y_true)
        FP = K.sum(y_pred - y_pred * y_true)
        return FP / N

    def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.3)):
        y_pred = K.cast(y_pred >= threshold, 'float32')
        P = K.sum(y_true)
        TP = K.sum(y_pred * y_true)
        return TP / P

    ptas = tf.stack([binary_PTA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k) for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)

def class_weight(labels):
    y = []
    for val in labels:
        ind = np.argwhere(val == 1).flatten().tolist()
        y.extend(ind)
    w = compute_class_weight('balanced', np.unique(y), y)
    w = dict(enumerate(w))
    return w

def build_model(blosum_dim, word2vec_dim, output_dim, embeddings_matrix):
    # 2D CNN for BLOSUM Input
    b_input = Input(shape=blosum_dim, dtype='float32')
    b_out = []
    layer_list = [[12, 8, 4], [16, 4, 4], [4, 4, 16]]
    for layer in layer_list:
        b = b_input
        for i in layer:
            b = Convolution2D(i, (1, 1), padding='same', activation='relu', strides=1)(b)
            b = MaxPooling2D(pool_size=(2, 2))(b)
            b_out.append(b)
    b_out = Flatten()(concatenate(b_out))

    # 1D-CNN + BiLSTM + Attention for Word2Vec Input
    w_input = Input(shape=word2vec_dim, dtype='float32')
    embedding_layer = Embedding(input_dim=embeddings_matrix.shape[0],  # length
                                output_dim=embeddings_matrix.shape[1],  # wv_length
                                weights=[embeddings_matrix],  #
                                input_length=word2vec_dim,  # padding
                                name='embedding_layer')
    embedded_sequences = embedding_layer(w_input)
    w_out = Convolution1D(filters=32, kernel_size=10, padding='valid',
                          activation='relu', strides=3)(embedded_sequences)
    w_out = Dropout(0.5)(w_out)
    w_out = Convolution1D(filters=16, kernel_size=10, padding='valid', activation='relu', strides=1)(w_out)
    w_out = MaxPooling1D(pool_size=2)(w_out)
    w_out = Convolution1D(filters=8, kernel_size=5, padding='valid', activation='relu', strides=1)(w_out)
    w_out = MaxPooling1D(pool_size=2)(w_out)

    w_out = Bidirectional(LSTM(32, return_sequences=True))(w_out)
    w_out = Bidirectional(LSTM(16))(w_out)
    w_out = Lambda(lambda x: K.reshape(x, (-1, 2, 16)))(w_out)

    w_out = SeqSelfAttention(attention_width=15, attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                             attention_activation=None, kernel_regularizer=l2(1e-6),
                             use_attention_bias=False, name='Attention', )(w_out)

    w_out = Lambda(lambda x: K.reshape(x, (-1, 32)))(w_out)
    w_out = Dense(16, activation='relu')(w_out)
    w_out = Dropout(0.3)(w_out)

    model = concatenate([w_out, b_out])
    model = Dense(16, activation='relu')(model)
    model = Dropout(0.3)(model)
    model = Dense(output_dim, activation='sigmoid')(model)
    model = Model([word2vec_dim, blosum_dim], model)
    return model

def train_model(train_word2vec, train_blosum, embeddings_matrix, train_labels, model_file):
    blosum_dim = train_blosum.shape[1:]
    word2vec_dim = train_word2vec.shape[1:]
    output_dim = train_labels.shape[1]
    model = build_model(blosum_dim = blosum_dim, word2vec_dim = word2vec_dim, output_dim = output_dim,
                        embeddings_matrix = embeddings_matrix)
    adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print('HiFun Compiling ...')
    model.compile(loss=[focal_loss(gamma=2., alpha=.25)],
                  optimizer=adam, metrics=['accuracy', auc, precision, recall, f1, mcc])
    model.summary()
    #####save each epoch models: set save_best_only=False #####
    checkpoint = ModelCheckpoint(model_file + format(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '.hdf5',
                                 monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    csv_logger = CSVLogger(model_file + 'training.log', separator=',', append=False)
    # class_weight =d
    # Training
    print('HiFun Training ...')
    model.fit(x=[train_word2vec, train_blosum], y= train_labels, validation_split = 0.3,epochs=100,
              class_weight=class_weight(train_labels),
              batch_size=32, shuffle=False, callbacks=[checkpoint, csv_logger, early_stopping])
    model.save(model_file)

