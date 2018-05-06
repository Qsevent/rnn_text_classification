# -*- coding:utf-8 -*-
"""
用于文本分类任务
train_file为已经分好词的文本 如 'token1 token2 ... \t label' 
token之间使用空格分开, 与label使用\t隔开
"""

import numpy as np
import cPickle
from log import Logger
import codecs
import math


class InputHelper():
    def __init__(self, log=None):
        self.padding_label = 'null'
        self.embeddings = []
        self.word2idx = {}
        self.idx2word = {}
        self.label_dictionary = {}
        if log == None:
            self.log = Logger('temp_log.txt')
        else:
            self.log = log
        self.batch_index = []
        self.num_batches = 0
        self.interaction_point = []
        self.word_point = []

    def load_info(self, embeddings, word2idx, idx2word, label_dictionary):
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.label_dictionary = label_dictionary
        self.log.info("load infomation finished")

    def load_embedding(self, embedding_file, embedding_size):
        '''
        load embedding file
        - embedding_file: perparatory train word dictionary file
        - emdedding_size: dimension of embeding
        note: <a> is a padding, 'UNK' represent unknows word
        '''

        count = 0
        words_num = 0
        for line in open(embedding_file):
            line = line.strip().replace('  ', ' ')
            line = line.strip().split(' ')
            if count == 0:
                words_num = int(line[0])
                file_embedding_size = int(line[1])
                if embedding_size != file_embedding_size:
                    self.log.error('file embedding dimension is false')
                self.embeddings.append(np.random.uniform(-0.1, 0.1, embedding_size).tolist())
                self.word2idx['<a>'] = len(self.word2idx)
                self.idx2word[len(self.word2idx)] = '<a>'

                self.embeddings.append(np.random.uniform(-0.1, 0.1, embedding_size).tolist())
                self.word2idx['UNK'] = len(self.word2idx)
                self.idx2word[len(self.word2idx)] = 'UNK'
            else:
                self.embeddings.append([float(var) for var in line[1:]])
                self.word2idx[line[0]] = len(self.word2idx)
                self.idx2word[len(self.word2idx)] = line[0]
            count += 1
        self.log.info('load embedding finish, %d words in total ' % (words_num + 2))

    def output_result(self, result, file, outfile):
        tmp_dic = {}
        for k,v in self.label_dictionary.items():
            tmp_dic[v] =k
        r = codecs.open(file, 'r', 'UTF-8')
        lines = r.readlines()
        w = codecs.open(outfile, 'w', 'UTF-8')
        count = 0
        for i in range(len(lines)):
            s = lines[i]
            if s.strip() != '':
                if len(s.strip().split('\t')) == 2:
                    w.write(s.strip() + '\t' + tmp_dic.get(result[count]) + '\n')
                else:
                    w.write(s.strip() + '\t\t' + tmp_dic.get(result[count])+ '\n')
                count+=1
            else:
                w.write('\n')

    def format_dialog(self, file):
        all = []
        one = []
        for line in codecs.open(file, 'r', 'UTF-8'):
            if line.strip() != '':
                one.append(line.strip())
            else:
                all.append(one)
                one = []
        all.append(one)
        return all

    def em_on(self, list_c, interaction_rounds, sequence_length):
        round_num = 0
        data_id = []
        one_interaction_word_point = []
        if len(list_c[-1].strip().split('\t'))==2:
            t = list_c[-1].strip().split('\t')[-1].strip()
            if len(t.split('&'))==2:
                label = t.split('&')[0]
            else:
                label = t
        else:
            label = self.padding_label
        for line in list_c:
            line = line.strip().split('\t')
            text = line[0]
            tokens = text.split(' ')
            round_num += 1
            tokens_id, word_point = self.embedding_sentence(tokens, sequence_length)
            data_id.append(tokens_id)
            one_interaction_word_point.append(word_point)

        if round_num < interaction_rounds:
            for i in range(interaction_rounds - round_num):
                tokens = []
                tokens_id, word_point = self.embedding_sentence(tokens, sequence_length)
                data_id.append(tokens_id)
                one_interaction_word_point.append(word_point)
        labels_id = self.label_dictionary.get(label)
        if len(one_interaction_word_point) != interaction_rounds:
            self.log.error(' valid word point error in one interaction')
        if round_num > interaction_rounds:
            round_num = interaction_rounds
        return data_id, labels_id, one_interaction_word_point, round_num

    def load_label_dictionary(self, file):
        self.label_dictionary = cPickle.load(open(file, "rb"))

    def load_valid(self, file, interaction_rounds, sequence_length):
        all = self.format_dialog(file)
        step = 1
        x = []
        y = []
        x_w_p = []
        x_s_p = []

        for one in all:
            star = 0
            length = len(one)
            end = star + step
            last = True
            flag = False
            while (last):
                if flag:
                    last = False
                else:
                    last = True
                tmp = one[star:end]
                data_id, labels_id, one_interaction_word_point, round_num = self.em_on(tmp, interaction_rounds,
                                                                                       sequence_length)
                x.extend(data_id)
                y.append(labels_id)
                x_w_p.append(one_interaction_word_point)
                x_s_p.append(round_num)
                if end - star == interaction_rounds:
                    star += step
                end += step
                if end >= length:
                    end = length
                    flag = True
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        x_w_p = np.array(x_w_p, dtype=int)
        x_s_p = np.array(x_s_p, dtype=int)

        return x, y, x_w_p, x_s_p

    def load_data(self, data_file, save_dir, interaction_rounds, sequence_length):
        '''
        create data id though embeding list
        - data_file: train data
        - save_dir: path to save id file of data
        - interaction_rounds: num of document interaction rounds
        - sequence_length: words num of sequence
        '''
        data_id = []
        labels_id = []
        round_num = 0
        label = ''
        one_interaction_word_point = []
        count = 0
        for line in codecs.open(data_file, 'r', 'UTF-8'):
            count += 1
            line = line.strip().split('\t')
            if line[0] != '':
                text = line[0]
                if len(line) == 2:
                    label = line[1]
                    if label not in self.label_dictionary:
                        self.label_dictionary[label] = len(self.label_dictionary)
                tokens = text.split(' ')
                round_num += 1
                tokens_id, word_point = self.embedding_sentence(tokens, sequence_length)
                data_id.append(tokens_id)
                one_interaction_word_point.append(word_point)
                if round_num > interaction_rounds:
                    self.log.error('data line is much than interaction')
                    print count
            else:
                if round_num < interaction_rounds:
                    for i in range(interaction_rounds - round_num):
                        tokens = []
                        tokens_id, word_point = self.embedding_sentence(tokens, sequence_length)
                        data_id.append(tokens_id)
                        one_interaction_word_point.append(word_point)
                labels_id.append(self.label_dictionary.get(label))

                if len(one_interaction_word_point) != interaction_rounds:
                    self.log.error('word point error in one interaction')
                if round_num > interaction_rounds:
                    round_num = interaction_rounds
                self.interaction_point.append(round_num)
                round_num = 0
                label = ''
                self.word_point.append(one_interaction_word_point)
                one_interaction_word_point = []
        if len(data_id) / interaction_rounds != len(labels_id):
            self.log.error('generate data false, the data lines num not match data labels num')
        self.log.info('data classes: %d' % len(self.label_dictionary))
        save_file = data_file.split('/')[-1].split('.')[0] + '_embedding_temp'
        with open(save_dir + save_file, 'w') as f:
            cPickle.dump(self.label_dictionary, f)
        print 'null',self.label_dictionary.get(self.padding_label)
        w = codecs.open('temp/label_dic.txt','w','utf-8')
        for k,v in self.label_dictionary.items():
            w.write(k+'\t'+str(v)+'\n')
        w.close()
        return (data_id, labels_id)

    def embedding_sentence(self, tokens, sequence_length):
        '''
        get id of sequence
        - tokens: a sequence
        - sequence_length: sequence length
        '''
        tokens_id = []
        tokens_length = len(tokens)
        word_point = tokens_length
        if tokens_length >= sequence_length:
            tokens = tokens[:sequence_length]
            word_point = sequence_length
        else:
            supmount = sequence_length - tokens_length
            tokens = tokens + ['<a>' for i in range(supmount)]
        for token in tokens:
            if token not in self.word2idx:
                tokens_id.append(self.word2idx.get('UNK'))
            else:
                tokens_id.append(self.word2idx.get(token))
        if len(tokens_id) != sequence_length:
            self.log.error('generate sequence length false')
        return tokens_id, word_point

    def generate_batches(self, data_tuple, batch_size, interaction_rounds):
        data = data_tuple[0]
        label = data_tuple[1]
        #        print data
        #        print label
        if len(data) % interaction_rounds != 0:
            self.log.error('padding interaction num false')

        one_batch_whole_interaction = math.ceil(batch_size * 1.0 / interaction_rounds)
        batch_size = int(one_batch_whole_interaction * interaction_rounds)
        num_batches = len(data) / batch_size
        self.num_batches = num_batches

        data = data[:num_batches * batch_size]
        label = label[:int(num_batches * one_batch_whole_interaction)]
        interaction_point = self.interaction_point[:int(num_batches * one_batch_whole_interaction)]
        word_point = self.word_point[:int(num_batches * one_batch_whole_interaction)]
        x_data = np.split(np.array(data, dtype=int), num_batches, 0)
        y_data = np.split(np.array(label, dtype=int), num_batches, 0)
        interaction_point_ = np.split(np.array(interaction_point, dtype=int), num_batches, 0)
        word_point_ = np.split(np.array(word_point, dtype=int), num_batches, 0)
        #        self.batch_index = np.array(range(num_batches))
        self.batch_index = np.random.permutation(num_batches)
        self.pointer = 0
        return x_data, y_data, interaction_point_, word_point_

    def next_batch(self, x_data, y_data, interaction_point, word_point):
        index = self.batch_index[self.pointer]
        self.pointer += 1
        x_batch, y_batch, interaction_point_batch, word_point = x_data[index], y_data[index], interaction_point[index], \
                                                                word_point[index]
        y_batch = [self.label_one_hot(y) for y in y_batch]
        return x_batch, y_batch, interaction_point_batch, word_point

    def label_one_hot(self, label_id):
        y = [0] * len(self.label_dictionary)
        y[int(label_id)] = 1.0
        return np.array(y)

    def reset_batch(self):
        self.batch_index = np.random.permutation(self.num_batches)
        self.pointer = 0


if __name__ == '__main__':
    data_loader = InputHelper()
    # data_loader.load_embedding('data/all_vectors.bin',100)
    # da,la =data_loader.load_data('data/train_s.txt','data/',6,100)
    # x_batch,y_batch,c,v = data_loader.generate_batches((da,la),128,6)
    # x,y,z,m = data_loader.next_batch(x_batch,y_batch,c,v)
    # print x
    # print y
    # print z
    # print m
