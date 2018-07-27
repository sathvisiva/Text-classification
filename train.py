import numpy as np
import pandas as pd
import os
import math
import yaml
import datetime
import json
import time
import tensorflow as tf
import logging
import utils.data_helpers as data_helpers
from model.text_cnn import TextCNN
from tensorflow.contrib import learn
#import tensorflow_transform as tft

logging.getLogger().setLevel(logging.INFO)


with open("config/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

default_dataset = cfg["datasets"]["default"]
dataset_path = cfg["datasets"][default_dataset]["path"]
xraw, y_raw, df = data_helpers.load_data_and_labels(dataset_path)
logging.info('Loading parameters file')
params = json.loads(open('config/parameters.json').read())

max_doc_length = max([len(x.split(' ')) for x in xraw])
logging.info('The maximum length of all sentences: {}'.format(max_doc_length))

vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_length)
x = np.array(list(vocab_processor.fit_transform(xraw)))
y = np.array(y_raw)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_train = x[shuffle_indices]
y_train = y[shuffle_indices]

logging.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

with tf.Graph().as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement = True,
		log_device_placement = False
		)
	sess = tf.Session(config = session_conf)
	with sess.as_default():
		nn = TextCNN(
				sequence_length=x_train.shape[1],
				num_classes=y_train.shape[1],
				vocab_size=len(vocab_processor.vocabulary_),
				embedding_size=params['embedding_dim'],
				filter_sizes=list(map(int, params['filter_sizes'].split(","))),
				num_filters=params['num_filters'],
				l2_reg_lambda=params['l2_reg_lambda'])

		global_step = tf.Variable(0, name = "global_step", trainable = False)
		optimizer = tf.train.AdamOptimizer(nn.learning_rate)
		tvars = tf.trainable_variables()
		grads,_ = tf.clip_by_global_norm(tf.gradients(nn.loss, tvars), params['grad_clip'])
		grads_and_vars = tuple(zip(grads,tvars))
		train_op = optimizer.apply_gradients(grads_and_vars, global_step = global_step)


		grad_summaries = []
		for g,v in grads_and_vars:
			if g is not None:
				grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name),g)
				sparsity_summary = tf.summary.scalar("{}grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
				grad_summaries.append(grad_hist_summary)
				grad_summaries.append(sparsity_summary)
		grad_summaries_merged = tf.summary.merge(grad_summaries)

		timestamp = str(int(time.time()))

		out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
		print("Writing to {}\n".format(out_dir))

		loss_summary = tf.summary.scalar("loss", nn.loss)
		acc_summary = tf.summary.scalar("accuracy", nn.accuracy)

		train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
		train_summary_dir = os.path.join(out_dir, "summaries", "train")
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

		checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
		checkpoint_prefix = os.path.join(checkpoint_dir, "model")
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=params['num_checkpoints'])
		vocab_processor.save(os.path.join(out_dir, "vocab"))
		sess.run(tf.global_variables_initializer())
		vocabulary = vocab_processor.vocabulary_

		initW = None
		print("Load glove file {}".format(cfg['word_embeddings']['glove']['path']))
		embedding_dimension = cfg['word_embeddings']['glove']['dimension']
		initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
		print("glove file has been loaded\n")
		sess.run(nn.W.assign(initW))

		def train_step(x_batch, y_batch, learning_rate):

			feed_dict = {
				nn.input_x : x_batch,
				nn.input_y : y_batch,
				nn.dropout_keep_prob : params['dropout_keep_prob'],
				nn.learning_rate : learning_rate
			}

			_, step, summaries, loss, accuracy = sess.run(
				[train_op, global_step, train_summary_op, nn.loss, nn.accuracy],
				feed_dict
				)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}, lr {:g}".format(time_str, step, loss, accuracy, learning_rate))
			train_summary_writer.add_summary(summaries, step)

		batches = data_helpers.batch_iter(list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
		max_learning_rate = 0.005
		min_learning_rate = 0.0001
		decay_speed = params['decay_coefficient']*len(y_train)/params['batch_size']

		counter = 0
		for batch in batches:
			learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-counter/decay_speed)
			counter += 1
			x_batch, y_batch = zip(*batch)
			train_step(x_batch, y_batch, learning_rate)
			current_step = tf.train.global_step(sess, global_step)
			if current_step % params['checkpoint_every'] == 0:
				path = saver.save(sess, checkpoint_prefix, global_step=current_step)
				print("Saved model checkpoint to {}\n".format(path))



#data = 
#labels = data_helpers.label_to_int(data['labels'])
#convert_to_text = data_helpers.int_to_label(data['labels'])
