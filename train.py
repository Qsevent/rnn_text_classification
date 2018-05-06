#-*- coding:utf-8 -*-

import tensorflow as tf
from model import BiRNN
from utils import InputHelper
import time
import os
import numpy as np
from log import Logger
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


# Parameters
# =================================================
tf.flags.DEFINE_integer('embedding_size', 100, 'embedding dimension of tokens')
tf.flags.DEFINE_integer('rnn_size', 100, 'hidden units of RNN , as well as dimensionality of character embedding (default: embedding_size)')
tf.flags.DEFINE_float('dropout_keep_prob', 1.0, 'Dropout keep probability (default : 0.5)')
tf.flags.DEFINE_integer('layer_size', 1, 'number of layers of RNN (default: 2)')
tf.flags.DEFINE_integer('batch_size', 128, 'Batch Size (default : 32)')
tf.flags.DEFINE_integer('sequence_length', 30, 'Sequence length (default : 32)')
tf.flags.DEFINE_integer('attn_size', 200, 'attention layer size (default: 2*rnn_size)')
tf.flags.DEFINE_float('grad_clip', 5.0, 'clip gradients at this value')
tf.flags.DEFINE_integer("num_epochs", 50, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_string('train_file', 'train_s_1k.txt', 'train raw file')
tf.flags.DEFINE_string('test_file', 'valid_t.txt', 'train raw file')
tf.flags.DEFINE_string('data_dir', 'data', 'data directory')
tf.flags.DEFINE_string('save_dir', 'save', 'model saved directory')
tf.flags.DEFINE_string('log_dir', 'log', 'log info directiory')
tf.flags.DEFINE_integer('save_steps', 1000, 'num of train steps for saving model')
tf.flags.DEFINE_integer('interaction_rounds',6,'num of the document interaction rounds')
tf.flags.DEFINE_string('embedding_file','data/all_vectors.bin','the word embedding file')
tf.flags.DEFINE_string('log_file','log.txt','log of program')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
# log
log = Logger(FLAGS.log_dir + '/' + FLAGS.log_file)
log.info( '\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    log.info('{0}={1}'.format(attr.upper(), value))


def train():
    #train data load
    data_loader = InputHelper(log=log)
    data_loader.load_embedding(FLAGS.embedding_file,FLAGS.embedding_size)
    train_data = data_loader.load_data(FLAGS.data_dir+'/'+FLAGS.train_file, FLAGS.data_dir+'/',FLAGS.interaction_rounds,FLAGS.sequence_length)
    x_batch,y_batch,train_interaction_point, train_word_point = data_loader.generate_batches(train_data,FLAGS.batch_size,FLAGS.interaction_rounds)
    FLAGS.vocab_size = len(data_loader.word2idx)
    FLAGS.n_classes = len(data_loader.label_dictionary)
    print FLAGS.n_classes
    FLAGS.num_batches = data_loader.num_batches
    FLAGS.embeddings = data_loader.embeddings
    # test data load
    test_data_loader = InputHelper(log=log)
    test_data_loader.load_info(embeddings=FLAGS.embeddings,word2idx=data_loader.word2idx,idx2word=data_loader.idx2word,
                                   label_dictionary=data_loader.label_dictionary)
    test_data = test_data_loader.load_data(FLAGS.data_dir + '/' + FLAGS.test_file, FLAGS.data_dir + '/',
                                       FLAGS.interaction_rounds, FLAGS.sequence_length)
    test_x_batch, test_y_batch, test_interaction_point,test_word_point = test_data_loader.generate_batches(test_data, FLAGS.batch_size, FLAGS.interaction_rounds)
    # Define specified Model
    model = BiRNN(embedding_size=FLAGS.embedding_size, rnn_size=FLAGS.rnn_size, layer_size=FLAGS.layer_size,
        vocab_size=FLAGS.vocab_size, attn_size=FLAGS.attn_size, sequence_length=FLAGS.sequence_length,
                n_classes=FLAGS.n_classes, interaction_rounds=FLAGS.interaction_rounds, batch_size=FLAGS.batch_size,
                  embeddings=FLAGS.embeddings,grad_clip=FLAGS.grad_clip, learning_rate=FLAGS.learning_rate)
    # define value for tensorboard
    tf.summary.scalar('train_loss', model.cost)
    tf.summary.scalar('accuracy', model.accuracy)
    merged = tf.summary.merge_all()

    # 调整GPU内存分配方案
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1000)
        
        total_steps = FLAGS.num_epochs * FLAGS.num_batches
        for e in xrange(FLAGS.num_epochs):
            data_loader.reset_batch()
            e_avg_loss = []
            t_acc = []
            start = time.time()
            num_tt = []
#            w=open('temp/pre'+str(e)+'.txt','w')
            for b in xrange(FLAGS.num_batches):

                x, y, z,m = data_loader.next_batch(x_batch,y_batch,train_interaction_point,train_word_point)
                feed = {model.input_data:x, model.targets:y, model.output_keep_prob:FLAGS.dropout_keep_prob, model.word_point:m, model.sentence_point:z}
                train_loss,t_accs,yy,yyy, summary,  _ = sess.run([model.cost, model.accuracy,model.y_results,
                model.y_tr, merged, model.train_op], feed_dict=feed)
                e_avg_loss.append(train_loss)
                t_acc.append(t_accs)
                global_step = e * FLAGS.num_batches + b
                if global_step % 20 == 0:
                    train_writer.add_summary(summary, e * FLAGS.num_batches + b)
                num_t = 0
                for i in range(len(yy)):
                    if yy[i] == yyy[i] and yy[i] != 4:
                        num_t+=1
                num_tt.append(num_t*1.0/len(yy))
#                w.write('predict '+str(len(yy))+'\n')
#                for y in yy:
#                    w.write(str(y)+'\t')
#                w.write('\ntrue '+str(len(yyy))+'\n')
#                for ys in yyy:
#                    w.write(str(ys)+'\t')
#                w.write('\n')
#           w.close()


            # model test
            test_data_loader.reset_batch()
            test_accuracy = []
            test_a = []
            for i in xrange(test_data_loader.num_batches):
                test_x, test_y, test_z, test_m = test_data_loader.next_batch(test_x_batch,test_y_batch,test_interaction_point,test_word_point)
                feed = {model.input_data:test_x, model.targets:test_y, model.output_keep_prob:1.0,model.word_point:test_m, model.sentence_point:test_z}
                accuracy,y_p,y_r = sess.run([model.accuracy,model.y_results,model.y_tr],feed_dict=feed)
                test_accuracy.append(accuracy)
                num_test = 0
                for j in range(len(y_p)):
                    if y_p[j] == y_r[j] and y_p[j] != 4:
                        num_test+=1
                test_a.append(num_test*1.0/len(y_p))
            end = time.time()
            num_tt_acc = np.average(num_tt)
            num_test_acc = np.average(test_a)
            avg_loss = np.average(e_avg_loss)
            print('e{},loss = {:.3f}, train_acc = {:.3f}, test_acc = {:.3f}, time/epoch'.format(e,avg_loss,num_tt_acc,num_test_acc,end - start ))
            #print and save
#            avg_loss = np.average(e_avg_loss)
#            t_avg_acc = np.average(t_acc)
#            log.info('epoch {}, train_loss = {:.3f},train_acc = {:.3f} test_accuracy:{:.3f}, time/epoch = {:.3f}'.format(e, avg_loss,t_avg_acc,np.average(test_accuracy), end - start))
            checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=e)


if __name__ == '__main__':
    train()
