import tensorflow as tf
import os
from utils import GenData
from params import create_hparams
from model import BaseModel
from tqdm import tqdm


data = GenData('cn2en.txt','jieba',10000)#(filepath, mode, data_length)
param = create_hparams()
#param.out_dir = 'model'
param.epochs = 10
param.batch_size = 16
param.encoder_vocab_size = len(data.id2cn)
param.decoder_vocab_size = len(data.id2en)
param.encoder_type = 'bi'    #uni | bi
g = BaseModel(param, 'train')


def train(self, data):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        add_num = 0
        if os.path.exists('model/checkpoint'):
            print('loading language model...')
            latest = tf.train.latest_checkpoint('model')
            add_num = int(latest.split('_')[-1])
            saver.restore(sess, latest)
        writer = tf.summary.FileWriter(
            'model/tensorboard', tf.get_default_graph())

        batch_num = len(data.data) // self.batch_size
        print('batch_num : ',batch_num)

        for k in range(self.epochs):
            total_loss = 0
            acc_sum = 0
            data_generator = data.generator(self.batch_size)
            for i in tqdm(range(batch_num)):
                en_inp, de_inp, de_tg, mask, en_len, de_len = next(
                    data_generator)
                #print(mask)
                feed = {
                    g.encoder_inputs: en_inp,
                    g.decoder_inputs: de_inp,
                    g.decoder_targets: de_tg,
                    g.mask: mask,#用于计算batch_loss
                    g.encoder_input_lengths: en_len,
                    g.decoder_input_lengths: de_len}
                cost, _, acc = sess.run([g.cost, g.update, g.acc], feed_dict=feed)
                total_loss += cost
                acc_sum += acc
                if (k * batch_num + i) % 10 == 0:
                    rs = sess.run(merged, feed_dict=feed)
                    writer.add_summary(rs, k * batch_num + i)

            print('epochs', k+add_num+1, ': average loss = ', round(total_loss / batch_num,6),': average acc = ',round(acc_sum / batch_num,5)*100,'%')

        saver.save(sess, 'model/model_%d' % (self.epochs + add_num))
        writer.close()



train(param,data)