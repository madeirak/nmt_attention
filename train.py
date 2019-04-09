import tensorflow as tf
import os
from utils import GenData
from params import create_hparams
from model import BaseModel



data = GenData('self.txt','jieba',20)#(filepath, mode, data_length)
param = create_hparams()
#param.out_dir = 'model'
param.epochs = 10
param.encoder_vocab_size = len(data.id2cn)
param.decoder_vocab_size = len(data.id2en)
param.encoder_type = 'uni'
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


        for k in range(self.epochs):
            total_loss = 0
            batch_num = len(data.data) // self.batch_size
            data_generator = data.generator(self.batch_size)
            for i in range(batch_num):
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
                cost, _ = sess.run([g.cost, g.update], feed_dict=feed)
                total_loss += cost
                if (k * batch_num + i) % 10 == 0:
                    rs = sess.run(merged, feed_dict=feed)
                    writer.add_summary(rs, k * batch_num + i)

            print('epochs', k+add_num+1, ': average loss = ', total_loss / batch_num)
        saver.save(sess, 'model/model_%d' % (self.epochs + add_num))
        writer.close()

train(param,data)