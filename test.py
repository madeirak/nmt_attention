from utils import GenData,plot_attention
from params import create_hparams
from model import BaseModel
import jieba
import tensorflow as tf


data = GenData('self.txt','jieba',20)#(filepath, mode, data_length)
param = create_hparams()
# infer模式下需要改动
param.batch_size = 1
param.keepprob = 1
param.encoder_vocab_size = len(data.id2cn)
param.decoder_vocab_size = len(data.id2en)
param.infer_mode = 'greedy', # greedy | beam_search
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


            #print('inputs : ',inputs)
            #print('outputs : ',outputs)
            #attention_plot = attention_plot[:len(outputs.split(' ')), :len(inputs)]
            #plot_attention(attention_plot, inputs, outputs.split(' '))

inference(data)

if __name__ == '__main__':
    pass
