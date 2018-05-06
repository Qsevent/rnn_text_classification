#-*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import math
class BiRNN(object):
    """
    用于文本分类的双向RNN
    """
    def __init__(self, embedding_size, rnn_size, layer_size,
        vocab_size, attn_size, sequence_length, n_classes, interaction_rounds,batch_size, embeddings=None, grad_clip=5.0, learning_rate=0.001):
        """
        - embedding_size: word embedding dimension
        - rnn_size : hidden state dimension
        - layer_size : number of rnn layers
        - vocab_size : vocabulary size
        - attn_size : attention layer dimension
        - sequence_length : max sequence lengths
        - n_classes : number of target labels
        - interaction_rounds: interaction rounds of sequence
        - embeddings: embedding vector
        - grad_clip : gradient clipping threshold
        - learning_rate : initial learning rate
        """
#        self.embeddings = 0
        one_batch_whole_interaction = math.ceil(batch_size * 1.0 / interaction_rounds)
        batch_size = int(one_batch_whole_interaction * interaction_rounds)
        self.output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')
#        self.input_data = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_data')
        self.input_data = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_data')
#        self.targets = tf.placeholder(tf.float32, shape=[None, n_classes], name='targets')
        self.targets = tf.placeholder(tf.float32, shape=[None, n_classes], name='targets')

        self.word_point = tf.placeholder(tf.int32,shape=[None,interaction_rounds],name='word_point')
        self.sentence_point = tf.placeholder(tf.int32,shape=[None],name='sentence_point')

        # multi layer rnn
        # define forword RNN Cell
        with tf.name_scope('fw_rnn'), tf.variable_scope('fw_rnn'):
            print tf.get_variable_scope().name
            lstm_fw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in xrange(layer_size)]
            lstm_fw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_fw_cell_list), output_keep_prob=self.output_keep_prob)

        # define backword RNN Cell
        with tf.name_scope('bw_rnn'), tf.variable_scope('bw_rnn'):
            print tf.get_variable_scope().name
            lstm_bw_cell_list = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in xrange(layer_size)]
            lstm_bw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_bw_cell_list), output_keep_prob=self.output_keep_prob)


        #with tf.device('/cpu:0'):
        if embeddings == None:
            embedding = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1),trainable=True,name='embedding')
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        else:
            embedding = tf.Variable(tf.convert_to_tensor(embeddings), trainable=True, name='embedding')
            inputs = tf.nn.embedding_lookup(embedding,self.input_data)





        # self.input_data shape: (batch_size , sequence_length)
        # inputs shape : (batch_size , sequence_length , rnn_size)
        # bidirection rnn 的inputs shape 要求是(sequence_length, batch_size, rnn_size)
        # 因此这里需要对inputs做一些变换
        # 经过transpose的转换已经将shape变为(sequence_length, batch_size, rnn_size)
        # 只是双向rnn接受的输入必须是一个list,因此还需要后续两个步骤的变换
        inputs = tf.transpose(inputs, [1,0,2])
        # 转换成(batch_size * sequence_length, rnn_size)
        inputs = tf.reshape(inputs, [-1, rnn_size])
        # 转换成list,里面的每个元素是(batch_size, rnn_size)
        inputs = tf.split(inputs, sequence_length, 0)

        with tf.name_scope('bi_rnn'), tf.variable_scope('bi_rnn'):
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs, dtype=tf.float32)
        # mask for sentence(word padding)
        outputs_tran = tf.transpose(outputs,[1,0,2])
        word_p = tf.reshape(self.word_point,[-1])
        mask_word = tf.sequence_mask(word_p,maxlen=sequence_length,dtype=tf.float32)
        mask_word = tf.expand_dims(mask_word,-1)
        outputs_mask = outputs_tran*mask_word
        outputs_mask = tf.transpose(outputs_mask,[1,0,2])
        outputs_mask = tf.reshape(outputs_mask,[-1,2*rnn_size])
        outputs_mask = tf.split(outputs_mask,sequence_length,0)
        # define attention layer
        attention_size = attn_size
        with tf.name_scope('attention'), tf.variable_scope('attention'):
            attention_w = tf.Variable(tf.truncated_normal([2*rnn_size, attention_size], stddev=0.1), name='attention_w')
            attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
            u_list = []
            for t in xrange(sequence_length):
                u_t = tf.tanh(tf.matmul(outputs_mask[t], attention_w) + attention_b)
                u_list.append(u_t)
            u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
            attn_z = []
            for t in xrange(sequence_length):
                z_t = tf.matmul(u_list[t], u_w)
                attn_z.append(z_t)
            # transform to batch_size * sequence_length
            attn_zconcat = tf.concat(attn_z, axis=1)
            self.alpha = tf.nn.softmax(attn_zconcat)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            alpha_trans = tf.reshape(tf.transpose(self.alpha, [1,0]), [sequence_length, -1, 1])
            self.final_output = tf.reduce_sum(outputs_mask * alpha_trans, 0)

        # define forword RNN Cell for interaction
        with tf.name_scope('fw_i_rnn'), tf.variable_scope('fw_i_rnn'):
            print tf.get_variable_scope().name
            lstm_ifw_cell_list = [tf.contrib.rnn.LSTMCell(2*rnn_size) for _ in xrange(layer_size)]
            lstm_ifw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_ifw_cell_list), output_keep_prob=self.output_keep_prob)

        # define backword RNN Cell for interaction
        with tf.name_scope('bw_i_rnn'), tf.variable_scope('bw_i_rnn'):
            print tf.get_variable_scope().name
            lstm_ibw_cell_list = [tf.contrib.rnn.LSTMCell(2*rnn_size) for _ in xrange(layer_size)]
            lstm_ibw_cell_m = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(lstm_ibw_cell_list), output_keep_prob=self.output_keep_prob)

            sentence_input = tf.reshape(self.final_output,[-1,interaction_rounds,2*rnn_size])
            sentence_input = tf.transpose(sentence_input,[1,0,2])
            sentence_input = tf.reshape(sentence_input,[-1,2*rnn_size])
            sentence_input = tf.split(sentence_input,interaction_rounds)
        # sentence_outputs shape (interaction_rounds, batch_size/interaction_rounds, 4*rnn_size)
        with tf.name_scope('bi_i_rnn'), tf.variable_scope('bi_i_rnn'):
            sentence_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_ifw_cell_m, lstm_ibw_cell_m, sentence_input, dtype=tf.float32)
        # mask for interaction(sentence padding)
        sentence_outputs_tran = tf.transpose(sentence_outputs,[1,0,2])
        mask_sentence = tf.sequence_mask(self.sentence_point,maxlen=interaction_rounds,dtype=tf.float32)
        mask_sentence = tf.expand_dims(mask_sentence,-1)
        outputs_mask_sentence = sentence_outputs_tran*mask_sentence
        outputs_mask_sentence = tf.transpose(outputs_mask_sentence,[1,0,2])
        outputs_mask_sentence = tf.reshape(outputs_mask_sentence,[-1,4*rnn_size])
        outputs_mask_sentence = tf.split(outputs_mask_sentence,interaction_rounds,0)

        # define attention layer
        i_attention_size = 2*attn_size
        with tf.name_scope('i_attention'), tf.variable_scope('i_attention'):
            i_attention_w = tf.Variable(tf.truncated_normal([4 * rnn_size, i_attention_size], stddev=0.1),
                                      name='i_attention_w')
            i_attention_b = tf.Variable(tf.constant(0.1, shape=[i_attention_size]), name='i_attention_b')
            i_u_list = []
            for t in xrange(interaction_rounds):
                i_u_t = tf.tanh(tf.matmul(outputs_mask_sentence[t], i_attention_w) + i_attention_b)
                i_u_list.append(i_u_t)
            i_u_w = tf.Variable(tf.truncated_normal([i_attention_size, 1], stddev=0.1), name='i_attention_uw')
            i_attn_z = []
            for t in xrange(interaction_rounds):
                z_t = tf.matmul(i_u_list[t], i_u_w)
                i_attn_z.append(z_t)
            # transform to batch_size * sequence_length
            i_attn_zconcat = tf.concat(i_attn_z, axis=1)
            self.i_alpha = tf.nn.softmax(i_attn_zconcat)
            # transform to sequence_length * batch_size * 1 , same rank as outputs
            i_alpha_trans = tf.reshape(tf.transpose(self.i_alpha, [1, 0]), [interaction_rounds, -1, 1])
            self.i_final_output = tf.reduce_sum(outputs_mask_sentence * i_alpha_trans, 0)

        # dirction use softmax to classify
 #       sentence_outputs_s = tf.convert_to_tensor(sentence_outputs)
#        sentence_outputs_s = tf.transpose(sentence_outputs_s,[1,0,2])
 #       sentence_outputs_s = tf.reshape(sentence_outputs_s,[-1,4*rnn_size])
 #       one_batch_whole_interaction = int(math.ceil(batch_size * 1.0 / interaction_rounds))
 #       sentence_outputs_s = tf.split(sentence_outputs_s,one_batch_whole_interaction,0)
#        accuracy = 0
 #       for i in range(len(sentence_outputs_s)):
        #ot_interaction = sentence_outputs_s[i]
        fc_w = tf.Variable(tf.truncated_normal([4*rnn_size, n_classes], stddev=0.1), name='ifc_w')
        fc_b = tf.Variable(tf.zeros([n_classes]), name='ifc_b')
        self.logits = tf.matmul(self.i_final_output, fc_w) + fc_b
        self.prob = tf.nn.softmax(self.logits)
        self.cost = tf.losses.softmax_cross_entropy(self.targets, self.logits)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)

        # dirction use softmax to classify
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.targets, axis=1), tf.argmax(self.prob, axis=1)), tf.float32))
        self.accuracy = accuracy
        self.y_results = tf.argmax(self.prob,axis=1)
        self.y_tr = tf.argmax(self.targets,axis= 1)



    def inference(self, sess, labels, inputs,w_p,s_p):

        prob = sess.run(self.prob, feed_dict={self.input_data:inputs,self.word_point:w_p,self.sentence_point:s_p, self.output_keep_prob:1.0})
        ret = np.argmax(prob, 1)
        ret = [labels[i] for i in ret]
        return ret


if __name__ == '__main__':
    em = [0.123,0.3243,0.435435,0.43564,0.345345,0.345345,0.32423,0.56464,0.24234,0.3454354]
    model = BiRNN(10, 10, 2, 5, 20, 5, 3,4,em,5, 0.001)
