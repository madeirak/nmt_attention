import numpy as np
import tensorflow as tf
#from pyhanlp import *
import jieba
import os
import matplotlib.pyplot as plt


# attention_mechanism
def attention_mechanism_fn(attention_type, num_units, memory, encoder_length):
    if attention_type == 'luong':
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units,  #attention深度
            memory,  #通常是RNNencoder的输出
            memory_sequence_length=encoder_length #在memory(tensor)的行中超过memory_sequence_length部分补0
        )
    elif attention_type == 'bahdanau':
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=encoder_length)
    else:
        raise ValueError('unkown atteion type %s' % attention_type)
    return attention_mechanism


# create_rnn_cell
def create_rnn_cell(unit_type, num_units, num_layers, keep_prob):
    def single_rnn_cell():
        if unit_type == 'lstm':
            single_cell = tf.contrib.rnn.LSTMCell(num_units)  #num_units ”门“中的隐藏神经元个数
        elif unit_type == 'gru':
            single_cell = tf.contrib.rnn.GRUCell(num_units)
        elif unit_type == 'rnn':
            single_cell = tf.contrib.rnn.LSTMCell(num_units)
        else:
            raise ValueError("Unknown cell type %s" % unit_type)
        cell = tf.contrib.rnn.DropoutWrapper(#Wrapper 包装器
            single_cell,
            output_keep_prob=keep_prob)#keep_prob:每个元素被保留的概率
        return cell
    mul_cell = tf. contrib.rnn.MultiRNNCell(#堆叠RNN隐藏层
        [single_rnn_cell() for _ in range(num_layers)])
    return mul_cell



class GenData(object):
    """docstring for GenData."""
    def __init__(self, filepath='self.txt', mode='jieba', data_length=20000):
        super(GenData, self).__init__()
        self.filepath = filepath
        self.mode = mode
        self.data_length = data_length
        self.SOURCE_CODES = ['<PAD>', '<UNK>']
        self.TARGET_CODES = ['<PAD>', '<GO>', '<EOS>', '<UNK>']
        self._init_data()
        self._init_vocab()
        self._init_num_data()

    def _init_data(self):
        with open(self.filepath, 'r', encoding='utf8') as f:
            self.data = f.readlines()
        self.data = self.data[:self.data_length]

    def _init_vocab(self):
        self.cn_list = [line.split('\t')[0] for line in self.data]
        self.en_list = [line.split('\t')[1].strip('\n') for line in self.data]
        self.en_list = [str.lower(line) for line in self.en_list]

        #jieba
        self.en_list = [[char for char in jieba.cut(line) if char != ' ']
            for line in self.en_list]
        self.cn_list = [[char for char in jieba.cut(line) if char != ' ']
            for line in self.cn_list]
        self.en_vocab = [word for line in self.en_list for word in line]
        self.en_vocab = sorted(set(self.en_vocab))
        self.ch_vocab = [word for line in self.cn_list for word in line]
        self.ch_vocab = sorted(set(self.ch_vocab))

        #make_vocab
        self.id2en = self.TARGET_CODES + list(self.en_vocab)
        self.en2id = {c:i for i,c in enumerate(self.id2en)}
        self.id2cn = self.SOURCE_CODES + self.ch_vocab
        self.cn2id = {c:i for i,c in enumerate(self.id2cn)}

    def _init_num_data(self):
        self.en_inp_num = [[self.cn2id[cn] for cn in line]
            for line in self.cn_list]
        self.de_inp_num = [[self.en2id['<GO>']] + [self.en2id[en]
            for en in line] for line in self.en_list]
        self.de_out_num = [[self.en2id[en] for en in line]
            + [self.en2id['<EOS>']] for line in self.en_list]

    def generator(self, batch_size):
        batch_num = len(self.en_inp_num) // batch_size
        for i in range(batch_num):
            begin = i * batch_size
            end = begin + batch_size
            encoder_inputs = self.en_inp_num[begin:end]
            decoder_inputs = self.de_inp_num[begin:end]
            decoder_targets = self.de_out_num[begin:end]
            encoder_lengths = [len(line) for line in encoder_inputs]
            decoder_lengths = [len(line) for line in decoder_inputs]
            encoder_max_length = max(encoder_lengths)
            decoder_max_length = max(decoder_lengths)
            encoder_inputs = np.array([data
                + [self.cn2id['<PAD>']] * (encoder_max_length - len(data))
                for data in encoder_inputs]).T
            decoder_inputs = np.array([data
                + [self.en2id['<PAD>']] * (decoder_max_length - len(data))
                for data in decoder_inputs]).T
            decoder_targets = np.array([data
                + [self.en2id['<PAD>']] * (decoder_max_length - len(data))
                for data in decoder_targets]).T
            mask = decoder_targets > 0 #tensor，<PAD>位置填0，target位置填1
            #print('mask :',mask)
            target_weights = mask.astype(np.int32)#mask bool 转为int后赋给target_weights，用于计算loss
            yield encoder_inputs, decoder_inputs, decoder_targets, \
                    target_weights, encoder_lengths, decoder_lengths


# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    plt.show()




if (__name__ == '__main__'):
    pass
