import numpy as np
import pandas as pd
import os
import time
import csv
import yaml
import datetime
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn import metrics
import utils.data_helpers as data_helpers

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

with open("config/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

x_raw = ["tata"]
y_test = None

vocab_path = os.path.join('/home/ameex/ML/projects/text-classification/runs/1532540562/vocab')
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))
print("\nPredicting...\n")
graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto( allow_soft_placement = True, log_device_placement = False)
	sess = tf.Session(config = session_conf)
	with sess.as_default():
		checkpoint_file = '/home/ameex/ML/projects/text-classification/runs/1532540562/checkpoints/model-700'
		saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
		saver.restore(sess, checkpoint_file)

		input_x = graph.get_operation_by_name("input_x").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
		scores = graph.get_operation_by_name("output/scores").outputs[0]
		predictions = graph.get_operation_by_name("output/predictions").outputs[0]
		batches = data_helpers.batch_iter(list(x_test), 37, 1, shuffle=False)
		all_predictions = []
		all_probabilities = None
		batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test, dropout_keep_prob: 1.0})
		all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
		probabilities = softmax(batch_predictions_scores[1])
		print(all_predictions)
		print(all_predictions.shape)
		print(max(all_predictions))
		
		for index, x_test_batch in enumerate(batches):
			print(x_test_batch)
			print("Predicting label")
			batch_predictions_scores = sess.run([predictions, scores], {input_x: x_test, dropout_keep_prob: 1.0})
			print(batch_predictions_scores)
			all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
			probabilities = softmax(batch_predictions_scores[1])
			if all_probabilities is not None:
				all_probabilities = np.concatenate([all_probabilities, probabilities])
			else:
				all_probabilities = probabilities
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}".format(time_str, (index+1)*FLAGS.batch_size))

