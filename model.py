import tensorflow as tf
import jieba
#from pyhanlp import *
from tensorflow.python.layers.core import Dense
import numpy as np
from utils import attention_mechanism_fn, create_rnn_cell
import os



class BaseModel():
    """docstring for BaseModel."""
    def __init__(self, params, mode):
        super(BaseModel, self).__init__()
        tf.reset_default_graph()
        self.mode = mode
        # networks
        self.num_units = params.num_units
        self.num_layers = params.num_layers
        # attention type and architecture
        self.attention_type = params.attention_type
        self.attention_architecture = params.attention_architecture
        # optimizer
        self.optimizer = params.optimizer
        self.learning_rate = params.learning_rate # sgd:1, Adam:0.0001
        self.decay_steps = params.decay_steps
        self.decay_rate = params.decay_rate
        self.epochs = params.epochs # 100000
        # Data
        self.out_dir = params.out_dir # log/model files
        # vocab
        self.encoder_vocab_size = params.encoder_vocab_size
        self.decoder_vocab_size = params.decoder_vocab_size
        self.share_vocab = params.share_vocab # False
        # Sequence lengths
        self.src_max_len = params.src_max_len # 50
        self.tgt_max_len = params.tgt_max_len # 50
        # Default settings works well (rarely need to change)
        self.unit_type = params.unit_type # lstm
        self.keep_prob = params.keep_prob # 0.8
        self.max_gradient_norm = params.max_gradient_norm # 1
        self.batch_size = params.batch_size # 32
        self.num_gpus = params.num_gpus # 1
        self.time_major = params.time_major
        # inference
        self.infer_mode = params.infer_mode # greedy / beam_search
        self.beam_width = params.beam_width # 0
        self.num_translations_per_input = params.num_translations_per_input # 1
        self._model_init()


    def _model_init(self):
        self._placeholder_init()
        self._embedding_init()
        self._encoder_init()
        self._decoder_init()


    def _placeholder_init(self):
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None])
        self.decoder_targets = tf.placeholder(tf.int32, [None, None])
        self.mask = tf.placeholder(tf.float32, [None, None])
        self.encoder_input_lengths = tf.placeholder(tf.int32, [None])
        self.decoder_input_lengths = tf.placeholder(tf.int32, [None])


    def _embedding_init(self):
        self.encoder_embedding = tf.get_variable(
            name='encoder_embedding',
            shape=[self.encoder_vocab_size, self.num_units])#num_units 嵌入维度
        self.encoder_emb_inp = tf.nn.embedding_lookup(
            self.encoder_embedding,
            self.encoder_inputs)
        self.decoder_embedding = tf.get_variable(
            name='decoder_embedding',
            shape=[self.decoder_vocab_size, self.num_units])
        self.decoder_emb_inp = tf.nn.embedding_lookup(
            self.decoder_embedding,
            self.decoder_inputs)


    def _encoder_init(self):
        with tf.name_scope('encoder'):
            encoder_cell = create_rnn_cell(
                self.unit_type,
                self.num_units,# #num_units ”门“中的隐藏神经元个数
                self.num_layers,# Network depth RNN隐藏层深度
                self.keep_prob) #for dropout
            encoder_init_state = encoder_cell.zero_state(#encoder_init_state.shape = [batch_size, state_size], filled with zeros
                self.batch_size, tf.float32)
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(#dynamic动态的RNN，通过循环动态构建网络，不需指定时序长度
                                                                         #encoder_state is N-tuple( N是时序长度 )，包含每个LSTMcell的 LSTMStateTuple
                                                                         #encoder_outputs.shape [batch_size, max_time, num_units(最后时间步的隐层unit_num)]
                encoder_cell,
                self.encoder_emb_inp,

                #sequence_length: 1-D 用来指定每个句子的有效长度（除去PAD） 超出的部分直接复制最后一个有效状态，并输出零向量
                sequence_length=self.encoder_input_lengths,

                time_major=self.time_major,#输入输出tensor格式，如果真，必须为[max_time, batch_size, depth]，否则[batch_size, max_time, depth]
                initial_state=encoder_init_state)#RNN初始状态，如果cell.state_size是整数，则必须形状是[batch_size, cell.state_size]的Tensor


    def _decoder_init(self):

        if self.time_major == True:
            memory = tf.transpose(self.encoder_outputs, [1, 0, 2])#[max_time, batch_size, depth] -> [batch_size, max_time, depth]
        else:
            memory = self.encoder_outputs  #[batch_size, max_time, depth]

        with tf.name_scope('decoder'):
            cell =create_rnn_cell(
                self.unit_type,
                self.num_units,
                self.num_layers,
                self.keep_prob)
            attention_mechanism = attention_mechanism_fn(
                self.attention_type,
                self.num_units,
                memory,
                self.encoder_input_lengths)
            cell = tf.contrib.seq2seq.AttentionWrapper(
                cell,
                attention_mechanism,
                attention_layer_size=self.num_units   #注意（输出）层的深度，不为None时，将上下文向量和单元输出送到关注层以在每个时间步产生注意力
            )

            init_state = cell.zero_state(self.batch_size, tf.float32).clone(
                cell_state=self.encoder_state)

            projection_layer = Dense(self.decoder_vocab_size, use_bias=False)

        if self.mode == 'train':
            train_helper = tf.contrib.seq2seq.TrainingHelper(#帮助建立Decoder，只在训练时使用
                inputs = self.decoder_emb_inp,
                sequence_length = self.decoder_input_lengths,
                time_major=True)
            train_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell,
                train_helper,
                init_state,
                output_layer=projection_layer#应用于RNN输出的层 （Dense）
            )
            outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(#return (final_outputs, final_state, final_sequence_lengths)
                train_decoder,
                output_time_major=True,
                swap_memory=True #swap交换，是否为此循环启用GPU-CPU内存交换
            )

            logits = outputs.rnn_output

            with tf.name_scope('optimizer'):
                # loss
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(#sparse 稀疏
                    labels=self.decoder_targets,
                    logits=logits
                )
                self.cost = tf.reduce_sum((loss * self.mask) / self.batch_size)#loss*mask 只算target位置loss，不算<PAD>
                tf.summary.scalar('loss', self.cost)
                # learning_rate decay
                self.global_step = tf.Variable(0)
                self.learning_rate = tf.train.exponential_decay(#将指数衰减应用于学习率
                    self.learning_rate,                         #decayed_learning_rate = learning_rate *
                                                                #   decay_rate ^ (global_step / decay_steps)
                    self.global_step,
                    self.decay_steps,
                    self.decay_rate,
                    staircase=True)#如果为True,那么global_step / decay_steps是整数除法，并且衰减学习率遵循阶梯函数

                # clip_by_global_norm 梯度裁剪
                self.trainable_variables = tf.trainable_variables()#tf.trainable_variables返回图中所有trainable=True的变量
                self.grads, _ = tf.clip_by_global_norm(#tf.clip_by_global_norm(t_list, clip_norm)
                                                       #t_list[i] * clip_norm / max(global_norm, clip_norm)
                                                       #global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

                    tf.gradients(self.cost, self.trainable_variables),#tf.gradients(y,x)返回len(x)的tensor列表，
                                                                      #返回列表第i个tensor是y对列表x第i个值求导的值
                    self.max_gradient_norm
                )

                # OPTIMIZE: adam | sgd
                if self.optimizer == 'adam':
                    opt = tf.train.AdamOptimizer(self.learning_rate)
                elif self.optimizer == 'sgd':
                    opt = tf.train.GradientDescentOptimizer(
                        self.learning_rate)
                else:
                    raise ValueError('unkown optimizer %s' % self.optimizer)

                self.update = opt.apply_gradients(
                    zip(self.grads, self.trainable_variables),#zip 对应元素打包成二元组，返回元组列表
                    global_step=self.global_step)

        elif self.mode == 'infer':
            start_tokens = tf.ones([self.batch_size,], tf.int32) * 1
            end_token = 2
            self.infer_mode = self.infer_mode[0]

            if self.infer_mode == 'greedy':
                infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(#使用输出logits的argmax并将结果传递给嵌入层以获取下一个输入
                    embedding=self.decoder_embedding,
                    start_tokens=start_tokens,
                    end_token=end_token)
                infer_decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell,
                    infer_helper,
                    init_state,
                    output_layer=projection_layer  #应用于RNN'输出的层 （Dense）
                )
            elif self.infer_mode == 'beam_search':
                infer_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=cell,
                    embedding=self.decoder_embedding,
                    start_tokens=start_tokens,
                    end_token=end_token,
                    initial_state=init_state,
                    beam_width=self.beam_width,
                    output_layer=projection_layer#应用于RNN'输出的层 （Dense）
                )
            else:
                raise ValueError('unkown infer mode %s' % self.infer_mode)

            decoder_outputs,_,_ = tf.contrib.seq2seq.dynamic_decode(
                decoder=infer_decoder,
                maximum_iterations=50)#允许的最大解码步数

            #print('final_state : ',final_state)

            if self.infer_mode == 'greedy':
                self.translations = decoder_outputs.sample_id

                logits = decoder_outputs.rnn_output
                print('logits',logits)

            elif self.infer_mode == 'beam_search':
                translations = decoder_outputs.predicted_ids
                self.translations = tf.reduce_sum(translations,-1)

                self.logits = tf.no_op()


            else:
                raise ValueError('unkown infer mode %s' % self.infer_mode)






