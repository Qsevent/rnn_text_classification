#-*- coding:utf-8 -*-


import tensorflow as tf
from utils import InputHelper
from model import BiRNN
from log import Logger
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.flags.DEFINE_string('embedding_file','data/all_vectors.bin','the word embedding file')
tf.flags.DEFINE_integer('embedding_size', 100, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN , as well as dimensionality of character embedding (default: 100)')
tf.flags.DEFINE_float('dropout_keep_prob', 1.0, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 1, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 30, 'Sequence length (default : 32)')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 30, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_float('decay_rate', 0.97, 'decay rate for rmsprop')
tf.flags.DEFINE_string('train_file', 'train.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'test.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'data directory')
tf.flags.DEFINE_string('log_file', 'data/valid_log.txt', 'data directory')
tf.flags.DEFINE_string('label_dic','data/train_s_embedding_temp','label_dictionary')
tf.flags.DEFINE_string('valid_file','data/valid.txt','valid data file')
tf.flags.DEFINE_integer('interaction_rounds',6,'num of the document interaction rounds')
tf.flags.DEFINE_integer('attn_size', 200, 'attention layer size (default: 2*rnn_size)')
tf.flags.DEFINE_string('result_file','result/result.txt','result file')
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

log = Logger(FLAGS.log_file)
def main():
    data_loader = InputHelper(log=log)
    data_loader.load_embedding(FLAGS.embedding_file, FLAGS.embedding_size)
    data_loader.load_label_dictionary(FLAGS.label_dic)
    x, y, x_w_p, x_s_p = data_loader.load_valid(FLAGS.valid_file,FLAGS.interaction_rounds,FLAGS.sequence_length)
    FLAGS.embeddings = data_loader.embeddings
    FLAGS.vocab_size = len(data_loader.word2idx)
    FLAGS.n_classes = len(data_loader.label_dictionary)
    model = BiRNN(embedding_size=FLAGS.embedding_size, rnn_size=FLAGS.rnn_size, layer_size=FLAGS.layer_size,
                  vocab_size=FLAGS.vocab_size, attn_size=FLAGS.attn_size, sequence_length=FLAGS.sequence_length,
                  n_classes=FLAGS.n_classes, interaction_rounds=FLAGS.interaction_rounds, batch_size=FLAGS.batch_size,
                  embeddings=FLAGS.embeddings, grad_clip=FLAGS.grad_clip, learning_rate=FLAGS.learning_rate)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(FLAGS.save_dir)
        model_path = FLAGS.save_dir+'/model.ckpt-45'
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, model_path)
        labels = model.inference(sess, y,x,x_w_p,x_s_p)
        corrcet_num = 0
        for i in range(len(labels)):
            if labels[i] == y[i]:
                corrcet_num+=1
        print('eval_acc = {:.3f}'.format(corrcet_num*1.0/len(labels)))
        data_loader.output_result(labels,FLAGS.valid_file,FLAGS.result_file)
        
if __name__ == '__main__':
	main()
