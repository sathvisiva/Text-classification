import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
import json

def clean_str(string):

	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"ain't", " is not ", string.lower())
	string = re.sub(r"can't", "cannot", string)
	string = re.sub(r"shan't", "shall not", string)
	string = re.sub(r"sha'n't", "shall not", string)
	string = re.sub(r"won't", "will not", string)
	string = re.sub(r"let's", "let us", string)
	string = re.sub(r"how'd", "how did", string)
	string = re.sub(r"how'd'y", "how do you", string)
	string = re.sub(r"where'd", "where did", string)
	string = re.sub(r"'m", " am ", string)
	string = re.sub(r"'d", " would had ", string)
	string = re.sub(r"n\'t", " not ", string)
	string = re.sub(r"\'ve", " have ", string)
	string = re.sub(r"\'re", " are ", string)
	string = re.sub(r"\'ll", " will ", string)
	string = re.sub(r"\'s", " is ", string)
	string = re.sub(r"\'cause", "because", string)
	string = re.sub(r"ma'am", "madam", string)
	string = re.sub(r"o'clock", "of the clock", string)
	string = re.sub(r"y'all", "you all", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	string = re.sub(r"1st", " first ", string)
	string = re.sub(r"2nd", " second ", string)
	string = re.sub(r"3rd", " third ", string)
	string = re.sub(r"4th", " fourth ", string)
	string = re.sub(r"5th", " fifth ", string)
	string = re.sub(r"6th", " sixth ", string)
	string = re.sub(r"7th", " seventh ", string)
	string = re.sub(r"8th", " eighth ", string)
	string = re.sub(r"9th", " ninth ", string)
	string = re.sub(r"10th", " tenth ", string)
	string = re.sub(r"0", " zero ", string)
	string = re.sub(r"1", " one ", string)
	string = re.sub(r"2", " two ", string)
	string = re.sub(r"3", " three ", string)
	string = re.sub(r"4", " four ", string)
	string = re.sub(r"5", " five ", string)
	string = re.sub(r"6", " six ", string)
	string = re.sub(r"7", " seven ", string)
	string = re.sub(r"8", " eight ", string)
	string = re.sub(r"9", " nine ", string)
	string = re.sub(r"1\/2", "half", string)
	return string.strip().lower()

def load_data_and_labels(file_path):
	f = open(file_path)
	labels = []
	text = []
	for line in f:
		sent = line.split("\t",1)
		labels.append(sent[0])
		text.append(sent[1].rstrip())
	data = pd.DataFrame({'text' : text, 'labels' : labels})
	data = data.reindex(np.random.permutation(data.index))

	labels = sorted(list(set(data['labels'].tolist())))
	with open('data/labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent = 4)
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot,1)
	label_dict = dict(zip(labels , one_hot))
	x_raw = data['text'].apply(lambda x : clean_str(x)).tolist()
	y_raw = data['labels'].apply(lambda y: label_dict[y]).tolist()

	return x_raw, y_raw, data




def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    print("filename" + filename)
    #f = open('../data/utterances.txt')
    f = open('/home/ameex/ML/projects/text-classification/data/glove.txt')
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    num_batches_per_epoch = data_size // batch_size
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            # if end_index - start_index != batch_size:
                # yield shuffled_data[end_index-batch_size:end_index]
            print(shuffled_data[start_index:end_index])
            yield shuffled_data[start_index:end_index]




