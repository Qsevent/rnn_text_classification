#-*- coding:utf-8 -*-
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

	def __init__(self):
            self.padding_label = 'null'
            self.embeddings = []
            self.word2idx = {}
            self.idx2word = {}
            self.label_dictionary = {}
            self.log = Logger('temp_log.txt')
        def load_embedding(self,embedding_file,embedding_size):
            '''
            load embedding file
            embedding_file:perparatory train word dictionary file
            emdedding_size:dimension of embeding
            note:<a> is a padding 
                 UNK represent unknows word
            '''
            embeddings =[]
            count = 0
            words_num = 0
            for line in open(embedding_file):
                line = line.strip().split(' ')
                if count == 0:
                    words_num = int(line[0])
                    file_embedding_size = int(line[1])
                    if embedding_size != file_embedding_size:
                        self.log.error('file embedding dimension is false')
                    self.embeddings.append(np.random.uniform(-0.1,0.1,embedding_size))
                    self.word2idx['<a>']=len(self.word2idx)
                    self.idx2word[len(self.word2idx)]='<a>'
                    
                    self.embeddings.append(np.random.uniform(-0.1,0.1,embedding_size))
                    self.word2idx['UNK']=len(self.word2idx) 
                    self.idx2word[len(self.word2idx)]='UNK'
                else:
                    self.embeddings.append(float(var) for var in line[1:])
                    self.word2idx[line[0]]=len(self.word2idx)
                    self.idx2word[len(self.word2idx)]=line[0]
                count += 1
            self.log.info('load embedding finish, %d words in total '%(words_num+2))

        def load_data(self,data_file,save_dir,interaction_rounds,sequence_length):
            '''
            create data id though embeding list
            data_file:train data
            save_dir:path to save id file of data
            interaction_rounds:num of document interaction rounds
            sequence_length:words num of sequence
            '''
            data_id = []
            labels_id = []
            round_num = 0
            for line in codecs.open(data_file,'r','UTF-8'):
                line = line.strip().split('\t')
                if line[0] != '':
                    text = line[0]
                    label = line[1]
                    if label not in self.label_dictionary:
                        self.label_dictionary[label] = len(self.label_dictionary)
                    labels_id.append(self.label_dictionary.get(label))
                    tokens = text.split(' ')
                    tokens_id = self.embedding_sentence(tokens,sequence_length)
                    round_num+= 1
                    if round_num <= interaction_rounds:
                        data_id.append(tokens_id)
                else:
                    if round_num < interaction_rounds:
                        for i in range(interaction_rounds-round_num):
                            tokens = []
                            tokens_id = self.embedding_sentence(tokens,sequence_length)
                            data_id.append(tokens_id)
                            if self.padding_label not in self.label_dictionary:
                                self.label_dictionary[self.padding_label] =len(self.label_dictionary)
                            labels_id.append(self.label_dictionary.get(self.padding_label))
                    round_num = 0
            if len(data_id) != len(labels_id):
                self.log.error('generate data false, the data lines num not match data labels num')
            self.log.info('classes: %d'%len(self.label_dictionary))
            save_file = data_file.split('/')[-1].split('.')[0]+'embedding_temp'
            with open(save_dir+save_file, 'w') as f:
		cPickle.dump((data_id,labels_id,self.label_dictionary), f)
            return (data_id,labels_id)
        


        def embedding_sentence(self,tokens,sequence_length):
            '''
            get id of sequence
            tokens:a sequence
            sequence_length:sequence length
            '''
            tokens_id = []
            tokens_length = len(tokens)
            if tokens_length >= sequence_length:
                tokens = tokens[:sequence_length]
            else:
                supmount = sequence_length-tokens_length
                tokens = tokens + ['<a>' for i in range(supmount)]
            for token in tokens:
                if token not in self.word2idx:
                    tokens_id.append(self.word2idx.get('UNK'))
                else:
                    tokens_id.append(self.word2idx.get(token))
            if len(tokens_id) != sequence_length:
                self.log.error('generate sequence length false')
            return tokens_id


	def create_dictionary(self, train_file, save_dir):
		"""
		从原始文本文件中创建字典
		train_file : 原始训练数据文档
		save_dir : 词典保存路径
		"""
		token_dictionary = {}
		token_index = 0

		label_dictionary = {}
		label_index = 0

		labels = []

		for line in open(train_file):
			# 使用unicode编码
			line = line.decode('utf-8')
			text, label = line.rstrip().split('\t')
			tokens = text.split(' ')
			if label not in label_dictionary:
				label_dictionary[label] = label_index
				labels.append(label)
				label_index += 1

			for token in tokens:
				if token not in token_dictionary:
					token_dictionary[token] = token_index
					token_index += 1


		token_dictionary['</s>'] = token_index
		token_index += 1
		self.vocab_size = len(token_dictionary)
		self.n_classes = len(label_dictionary)
		print 'Corpus Vocabulary:{0}, Classes:{1}'.format(self.vocab_size, self.n_classes)

		with open(save_dir+'dictionary', 'w') as f:
			cPickle.dump((token_dictionary, label_dictionary), f)

		self.token_dictionary = token_dictionary
		self.label_dictionary = label_dictionary
		self.labels = labels

	def load_dictionary(self, dictionary_file):

		with open(dictionary_file) as f:
			self.token_dictionary, self.label_dictionary = cPickle.load(f)
			self.vocab_size = len(self.token_dictionary)
			self.n_classes = len(self.label_dictionary)

			self.labels = [None for i in xrange(self.n_classes)]

			for key in self.label_dictionary:
				self.labels[self.label_dictionary[key]] = key


        def generate_batches(self,data_tuple,batch_size,interaction_rounds):
            data = data_tuple[0]
            label = data_tuple[1]
            if len(data) % interaction_rounds != 0:
                self.log.error('padding interaction num false')

            one_batch_whole_interaction = math.ceil(batch_size*1.0/interaction_rounds)
            batch_size = int(one_batch_whole_interaction * interaction_rounds)
            num_batches = int(math.floor(len(data)*1.0/batch_size))

            data = data[:num_batches*batch_size]
            label = label[:num_batches*batch_size]


            x_data = np.split(np.array(data,dtype=int),num_batches,0)
            y_data = np.split(np.array(label,dtype=int),num_batches,0)
            self.batch_index = np.random.permutation(num_batches)
            self.pointer = 0
            return x_data,y_data


        def next_batch(self,x_data,y_data):
	    index = self.batch_index[self.pointer]
	    self.pointer += 1		
	    x_batch, y_batch = x_data[index], y_data[index]
	    y_batch = [self.label_one_hot(y) for y in y_batch]
	    return x_batch, y_batch








	def create_batches(self, train_file, batch_size, sequence_length):

		self.x_data = []
		self.y_data = []
		padding_index = self.vocab_size - 1
		for line in open(train_file):
			line = line.decode('utf-8')
			text, label = line.rstrip().split('\t')
			tokens = text.split(' ')
			seq_ids = [self.token_dictionary.get(token) for token in tokens if self.token_dictionary.get(token) is not None]
			seq_ids = seq_ids[:sequence_length]
			for _ in xrange(len(seq_ids), sequence_length):
				seq_ids.append(padding_index)

			self.x_data.append(seq_ids)
			self.y_data.append(self.label_dictionary.get(label))

		self.num_batches = len(self.x_data) / batch_size
		self.x_data = self.x_data[:self.num_batches * batch_size]
		self.y_data = self.y_data[:self.num_batches * batch_size]

		self.x_data = np.array(self.x_data, dtype=int)
		self.y_data = np.array(self.y_data, dtype=int)
		self.x_batches = np.split(self.x_data.reshape(batch_size, -1), self.num_batches, 1)
		self.y_batches = np.split(self.y_data.reshape(batch_size, -1), self.num_batches, 1)
		self.pointer = 0





	def label_one_hot(self, label_id):

		y = [0] * len(self.label_dictionary)
		y[int(label_id)] = 1.0

		return np.array(y)
            
#	def next_batch(self):
#		index = self.batch_index[self.pointer]
#		self.pointer += 1		
#		x_batch, y_batch = self.x_batches[index], self.y_batches[index]
#		y_batch = [self.label_one_hot(y) for y in y_batch]
#		return x_batch, y_batch

	def reset_batch(self):
		self.batch_index = np.random.permutation(self.num_batches)
		self.pointer = 0

	def transform_raw(self, text, sequence_length):

		if not isinstance(text, unicode):
			text = text.decode('utf-8')

		x = [self.token_dictionary.get(token) for token in text]
		x = x[:sequence_length]
		padding_index = self.vocab_size - 1
		for _ in xrange(len(x), sequence_length):
			x.append(padding_index)

		return x


if __name__ == '__main__':
	data_loader = InputHelper()
        data_loader.load_embedding('data/test_embedding.bin',3)
        da,la =data_loader.load_data('data/t.txt','data/',4,5)
        x_batch,y_batch = data_loader.generate_batches((da,la),4,4)
        x,y = data_loader.next_batch(x_batch,y_batch)
