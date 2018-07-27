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

logging.getLogger().setLevel(logging.INFO)


with open("config/config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)



if __name__ == '__main__':


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
	vocabulary = vocab_processor.vocabulary_


	logging.info("loading glove")
	embedding_dimension = cfg['word_embeddings']['glove']['dimension']
	initW = data_helpers.load_embedding_vectors_glove(vocabulary,
                                                                  cfg['word_embeddings']['glove']['path'],
                                                                  embedding_dimension)
	logging.info("successfully loaded")
