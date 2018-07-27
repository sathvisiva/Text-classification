import tensorflow as tf
import numpy as np
import os
import time
import datetime
import utils.data_helpers as data_helpers
from model.text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import yaml
import logging
import json

logging.getLogger().setLevel(logging.INFO)


with open("config/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

params = json.loads(open('config/parameters.json').read())
labels = json.loads(open('data/labels.json').read())
one_hot = np.zeros((len(labels), len(labels)), int)
np.fill_diagonal(one_hot, 1)


default_dataset = cfg["datasets"]["default"]
dataset_path = cfg["datasets"][default_dataset]["path"]
x_raw, y_raw, df = data_helpers.load_data_and_labels(dataset_path)
#_raw = ['do you support black people']
logging.info('Loading parameters file')


vocab_path = os.path.join('/home/ameex/ML/projects/text-classification/runs/1532540562/vocab')
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
logging.info("loaded vocab processor")
x_test = np.array(list(vocab_processor.transform(x_raw)))
checkpoint_file = tf.train.latest_checkpoint('home/ameex/ML/projects/text-classification/runs/1532540562/checkpoints/checkpoint')
logging.info("loaded checkpoint file")

graph = tf.Graph()
with graph.as_default():
	session_conf = tf.ConfigProto(
		allow_soft_placement = True,
		log_device_placement = False
		)

	sess = tf.Session(config = session_conf)

	with sess.as_default():
		#print("checkpoint file" + checkpoint_file)
		saver = tf.train.import_meta_graph("/home/ameex/ML/projects/text-classification/runs/1532536239/checkpoints/model-700.meta")
		saver.restore(sess, 'home/ameex/ML/projects/text-classification/runs/1532536239/checkpoints/model-700')

		input_x = graph.get_operation_by_name("input_x").outputs[0]
		dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
		predictions = graph.get_operation_by_name("output/predictions").outputs[0]
		batches = data_helpers.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
		all_predictions = []

		for x_test_batch in batches:
			batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
			all_predictions = np.concatenate([all_predictions, batch_predictions])

predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))

out_path = os.path.join("home/ameex/ML/projects/text-classification/runs/1532536239/", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
	csv.writer(f).writerows(predictions_human_readable)
