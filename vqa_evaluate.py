import time
import logging
import os
import sys
import h5py
import argparse
from datetime import datetime
import json

from utils import *

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

class Config():
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    dropout = 0.5
    
    rnn_size = 512   # Number of hidden states in an rnn layer
    rnn_layer = 2

    input_embed_size = 200
    image_size = 4096
    common_embed_size = 1024  #size of common embdding vector

    ques_max_words = 26
    num_output = 1000    #number of output answers

    batch_size = 500

    max_iterations = 50000

    lr = 0.0003

    img_norm = True  # normalize image

    gpu_id = 0

    ### checkpoints
    save_checkpoint_every = 5000

    ### number of checkpoints to keep
    keep = 1

    input_image_h5  = './data_img.h5'
    input_ques_h5   = './data_prepro.h5'
    input_json      = './data_prepro.json'

    max_gradient_norm = 10.0

    #### other
    print_every = 10

    ### Learning rate decay factor used in the GT implementation 
    decay_factor = 0.99997592083

    def __init__(self, output_path=None):
        if output_path:
            # Where to save things.
            self.output_path = output_path
        else:
            self.output_path = "results/cnn_lstm_baseline/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"

class VQAEvaluator(object):
    """Top-level Question Answering module"""

    def __init__(self, config, vocab_size):
        """
        Initializes the QA model.
        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the VQAEvaluator..."
        self.config = config
        self.vocab_size = vocab_size

        # Add all parts of the graph
        with tf.variable_scope("VQAEvaluator"):
            self.add_placeholders()
            self.add_variables()
            scores_emb = self.build_graph()
            self.generated_ans = scores_emb


    def add_variables(self):
        random_initializer = tf.random_uniform_initializer(-0.08, 0.08)
        self.embed_ques_W =  tf.get_variable("embed_ques_W", shape=[self.vocab_size, self.config.input_embed_size], initializer=random_initializer)

        self.lstm_1 = tf.nn.rnn_cell.LSTMCell(self.config.rnn_size, use_peepholes=True)
        self.lstm_dropout_1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = self.keep_prob)
        
        self.lstm_2 = tf.nn.rnn_cell.LSTMCell(self.config.rnn_size, use_peepholes=True)
        self.lstm_dropout_2 = tf.nn.rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = self.keep_prob)
        
        self.stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

        # state-embedding
        self.embed_state_W = tf.get_variable('embed_state_W', shape=[2 * self.config.rnn_size * self.config.rnn_layer, self.config.common_embed_size], initializer=random_initializer)
        self.embed_state_b = tf.get_variable('embed_state_b', shape=[self.config.common_embed_size], initializer=random_initializer)
        
        # image-embedding
        self.embed_image_W = tf.get_variable('embed_image_W', shape=[self.config.image_size, self.config.common_embed_size], initializer=random_initializer)
        self.embed_image_b = tf.get_variable('embed_image_b', shape=[self.config.common_embed_size], initializer=random_initializer)
        
        # score-embedding
        self.embed_score_W = tf.get_variable('embed_score_W', shape=[self.config.common_embed_size, self.config.num_output], initializer=random_initializer)
        self.embed_score_b = tf.get_variable('embed_score_b', shape=[self.config.num_output], initializer=random_initializer)

    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.image_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.image_size])
        self.ques_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.ques_max_words])
        self.labels = tf.placeholder(tf.int64, shape=[None,])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def build_graph(self):
        """Builds the main part of the graph for the model
        """
        state = self.stacked_lstm.zero_state(self.config.batch_size, tf.float32)
        for i in range(self.config.ques_max_words):
            if i==0:
                ques_emb_linear = tf.zeros([self.config.batch_size, self.config.input_embed_size])
            else:
                tf.get_variable_scope().reuse_variables()
                ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, self.ques_placeholder[:,i-1])

            ques_emb = tf.tanh(tf.nn.dropout(ques_emb_linear, self.keep_prob))
            output, state = self.stacked_lstm(ques_emb, state)

        state = tf.transpose(state, [2,0,1,3]) #get batch size first
        state = tf.reshape(state, [self.config.batch_size,-1] )
        
        state_drop = tf.nn.dropout(state, self.keep_prob)

        state_emb = tf.tanh(tf.matmul(state_drop, self.embed_state_W) + self.embed_state_b)

        image_drop = tf.nn.dropout(self.image_placeholder, self.keep_prob)
        image_emb = tf.tanh(tf.matmul(image_drop, self.embed_image_W) + self.embed_image_b)

        # fuse question & image
        scores = tf.multiply(state_emb, image_emb)
        scores_drop = tf.nn.dropout(scores, self.keep_prob)
        scores_emb = tf.matmul(scores_drop, self.embed_score_W) + self.embed_score_b
        return scores_emb
        
    def add_loss(self, logits):
        """
        Add loss computation to the graph.
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
        self.loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', self.loss)

    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)
        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard
        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.ques_placeholder] = batch["questions"]
        input_feed[self.labels] = batch["answers"]
        input_feed[self.image_placeholder] = batch["images"]
        input_feed[self.keep_prob] = 1 - self.config.dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)
        self.config.lr = self.config.lr*self.config.decay_factor

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm

    def generate(self, session, dataset, img_features, test_data, model_path):
        num_test = test_data['question'].shape[0]
        saver = tf.train.Saver()
        saver.restore(session, model_path)

        tStart_total = time.time()
        result = []
        for current_batch_start_idx in xrange(0, num_test-1, batch_size):

            tStart = time.time()
            # set data into current*
            if current_batch_start_idx + batch_size < num_test:
                current_batch_file_idx = range(current_batch_start_idx,current_batch_start_idx+batch_size)
            else:
                current_batch_file_idx = range(current_batch_start_idx,num_test)

            current_question = test_data['question'][current_batch_file_idx,:]
            current_length_q = test_data['length_q'][current_batch_file_idx]
            current_img_list = test_data['img_list'][current_batch_file_idx]
            current_ques_id  = test_data['ques_id'][current_batch_file_idx]
            current_img = img_features[current_img_list,:] # (batch_size, dim_image)

            # deal with the last batch
            if(len(current_img)<500):
                    pad_img = np.zeros((500-len(current_img),dim_image),dtype=np.int)
                    pad_q = np.zeros((500-len(current_img),max_words_q),dtype=np.int)
                    pad_q_len = np.zeros(500-len(current_length_q),dtype=np.int)
                    pad_q_id = np.zeros(500-len(current_length_q),dtype=np.int)
                    pad_ques_id = np.zeros(500-len(current_length_q),dtype=np.int)
                    pad_img_list = np.zeros(500-len(current_length_q),dtype=np.int)
                    current_img = np.concatenate((current_img, pad_img))
                    current_question = np.concatenate((current_question, pad_q))
                    current_length_q = np.concatenate((current_length_q, pad_q_len))
                    current_ques_id = np.concatenate((current_ques_id, pad_q_id))
                    current_img_list = np.concatenate((current_img_list, pad_img_list))


            generated_ans = sess.run(
                    self.generated_ans,
                    feed_dict={
                        self.image_placeholder: current_img,
                        self.ques_placeholder: current_question,
                        self.keep_prob: 1
                        })

            top_ans = np.argmax(generated_ans, axis=1)


            # initialize json list
            for i in xrange(0,500):
                ans = dataset['ix_to_ans'][str(top_ans[i]+1)]
                if(current_ques_id[i] == 0):
                    continue
                result.append({u'answer': ans, u'question_id': str(current_ques_id[i])})

            tStop = time.time()
            print ("Testing batch: ", current_batch_file_idx[0])
            print ("Time Cost:", round(tStop - tStart,2), "s")
        print ("Testing done.")
        tStop_total = time.time()
        print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")
        # Save to JSON
        print 'Saving result...'
        my_list = list(result)
        dd = json.dump(my_list,open('data.json','w'))

if __name__ == '__main__':

    config = Config()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    # Some GPU settings
    gpu_config=tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    # create a directory for best checkpoint if it does not exist
    bestmodel_dir = os.path.join(config.output_path, "best_checkpoint")

    if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
            os.makedirs(config.log_output)
    
    print("Output path", config.log_output)
    file_handler = logging.FileHandler(os.path.join(config.log_output, "log.txt"))

    logging.getLogger().addHandler(file_handler)

    print("Loading test dataset")
    dataset, img_features, test_data = get_test_data(config)
    vocab_size = len(dataset['ix_to_word'].keys())

    vqa_evaluator = VQAEvaluator(config, vocab_size)
    init = tf.global_variables_initializer()

    model_path = os.path.join(config.output_path, "vqa.ckpt")

    ### Train
    with tf.Session(config=gpu_config) as sess:
            sess.run(init)
            # Train
            vqa_evaluator.generate(sess, dataset, img_features, test_data, model_path)

    ### Test


    
