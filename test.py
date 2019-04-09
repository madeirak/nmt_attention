from utils import GenData,plot_attention
from params import create_hparams
from model import BaseModel
import jieba
import tensorflow as tf
import matplotlib.pyplot as plt

data = GenData('self.txt','jieba',20)#(filepath, mode, data_length)
param = create_hparams()
# infer模式下需要改动
param.batch_size = 1
param.keepprob = 1
param.encoder_vocab_size = len(data.id2cn)
param.decoder_vocab_size = len(data.id2en)  #107

param.infer_mode = 'beam_search', # greedy | beam_search
g = BaseModel(param, 'infer')


def inference(data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint('model')
        saver.restore(sess, latest)
        while True:
            inputs = input('input chinese: ')
            if inputs == 'exit': break
            inputs = jieba.lcut(inputs)
            #encoder_inputs = [[data.cn2id.get(cn, 3)] for cn in inputs]
            encoder_inputs = [[data.cn2id.get(cn)] for cn in inputs]
            #print('encoder_inputs : ',encoder_inputs)

            encoder_length = [len(encoder_inputs)]
            feed = {
                g.encoder_inputs: encoder_inputs,
                g.encoder_input_lengths: encoder_length}
            predict = sess.run(g.translations, feed_dict=feed)
            #print(predict)
            outputs = ' '.join([data.id2en[i] for i in predict[0][:-1]])
            print('output english:', outputs)


            print()

            '''
            attention_plot = attention_plot[:len(outputs.split(' ')), :len(inputs.split(' '))]
            plot_attention(attention_plot, inputs.split(' '), outputs.split(' '))
            '''
            #print('inputs : ',inputs)
            #print('outputs : ',outputs)
            #attention_plot = attention_plot[:len(outputs.split(' ')), :len(inputs)]
            #plot_attention(attention_plot, inputs, outputs.split(' '))


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    fontdict = {'fontsize': 14}
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)
    plt.show()




inference(data)

if __name__ == '__main__':
    pass
