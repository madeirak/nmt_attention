from utils import GenData
from params import create_hparams
from model import BaseModel
import jieba
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data = GenData('cn2en.txt','jieba',20)#(filepath, mode, data_length)
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
        print('输入exit，敲回车结束.')
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


            #print(predict)
            if param.infer_mode[0] == 'beam_search':
                predict = sess.run(g.translations, feed_dict=feed)
                predict = predict[0]
                predict = list(predict[i][0]for i in range(predict.shape[1]))
                outputs = ' '.join([data.id2en[i] for i in predict[:-1]])
                print('output english:', outputs)

            else:#greedy
                predict, attention = sess.run([g.translations, g.inference_attention_matrices], feed_dict=feed)
                outputs = ' '.join([data.id2en[i] for i in predict[0][:-1]])
                print('output english:', outputs)

                #attention 可视化
                i = 0 #第i个样本
                matrix = attention[:, i, :].T
                #print('matrix',matrix)
                lx = encoder_length
                y_ = [data.id2en[i] for i in predict[0][:-1]]

                #print('lx',lx)
                #print('y_',y_)
                #print('y_shape',y_.type)
                #print('x',x)
                y_len = len(y_)


                for idx in range(y_len):
                    if y_[idx] == 0:
                        y_len = idx + 1
                        break

                y_valid = y_[:y_len]
                #print('y_valid',y_valid)
                #print('x[3][0]',x[3][0])
                #print('inputs',inputs)
                x_valid = inputs

                #x_valid = np.array([data.cn2id.get(cn) for cn in inputs])
                #print('x_valid',x_valid)
                #print('y_valid',y_valid)
                plot_attention_matrix(
                                      src=x_valid, tgt=y_valid,
                                      matrix=matrix[:lx[i], :y_len]
                                      )

def plot_attention_matrix(src, tgt, matrix,name= "attention_matrix.png"):
    #src = [str(item) for item in src]
    #tgt = [str(item) for item in tgt]
    df = pd.DataFrame(matrix, index=src, columns=tgt)
    sns.set(font_scale=1.2, font='simhei')
    #ax = sns.heatmap(df, cmap = "YlGnBu",linewidths=.2,vmin = 0,vmax = 1)
    ax = sns.heatmap(df,linewidths=.2,vmin = 0,vmax = 1)
    ax.set_xlabel("target")
    ax.set_ylabel("source")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.set_title("Attention heatmap")
    #plt.savefig(name, bbox_inches='tight')
    plt.show()
    plt.gcf().clear()

    return matrix



if __name__ == '__main__':
    inference(data)


