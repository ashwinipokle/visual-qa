import time
import logging
import os
import sys
import h5py
import argparse
from datetime import datetime

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
    n_epochs = 300
    max_iterations = 150000

    lr = 0.0003

    img_norm = True  # normalize image

    gpu_id = 0

    ### checkpoints
    save_checkpoint_every = 15000

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

class VQAModel(object):
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
        print "Initializing the VQAModel..."
        self.config = config
        self.vocab_size = vocab_size

        # Add all parts of the graph
        with tf.variable_scope("VQAModel"):
            self.add_placeholders()
            self.add_variables()
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_variables(self):
        """
        add variables of the model to the graph
        
        """
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
        self.embed_score_W = tf.get_variable('embed_score_W', shape=[self.config.common_embed_size, self.config.common_embed_size], initializer=random_initializer)
        self.embed_score_b = tf.get_variable('embed_score_b', shape=[self.config.common_embed_size], initializer=random_initializer)

    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.image_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.image_size])
        self.ques_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.ques_max_words])
        self.labels = tf.placeholder(tf.int32, shape=[None,])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def build_graph(self):
        """Builds the main part of the graph for the model
        """
        state = self.stacked_lstm.zero_state(self.config.batch_size, tf.float32)
        loss = 0.0
        for i in range(self.config.ques_max_words):
            if i==0:
                ques_emb_linear = tf.zeros([self.config.batch_size, self.config.input_embed_size])
            else:
                tf.get_variable_scope().reuse_variables()
                ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, self.ques_placeholder[:,i-1])

            ques_emb = tf.tanh(tf.nn.dropout(ques_emb_linear, self.keep_prob))
            output, state = self.stacked_lstm(ques_emb, state)

        state_drop = tf.nn.dropout(state, self.keep_prob)
        state_emb = tf.tanh(tf.matmul(state_drop, self.embed_state_W) + self.embed_state_b)

        image_drop = tf.nn.dropout(image, self.keep_prob)
        image_emb = tf.tanh(tf.matmul(image_drop, self.embed_image_W, self.embed_image_b))

        # fuse question & image
        scores = tf.mul(state_emb, image_emb)
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

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm

    def train(self, session, dataset, img_features, train_data):
        """
        Main training loop.
        Inputs:
          session: TensorFlow session
        """

        tic = time.time()

        # Print number of model parameters
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.config.output_path, "vqa.ckpt")
        bestmodel_dir = os.path.join(self.config.output_path, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "vqa_best.ckpt")

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.config.output_path, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while epoch < self.config.max_iterations:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in makebatches(config.batch_size, img_features, train_data):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Sometimes print info to screen
                if global_step % self.config.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.config.save_checkpoint_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                # if global_step % self.config.eval_every == 0:

                #     # Get loss for entire dev set and log to tensorboard
                #     dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                #     logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                #     write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                #     # Get F1/EM on train set and log to tensorboard
                #     train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                #     logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                #     write_summary(train_f1, "train/F1", summary_writer, global_step)
                #     write_summary(train_em, "train/EM", summary_writer, global_step)


                #     # Get F1/EM on dev set and log to tensorboard
                #     dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                #     logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                #     write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                #     write_summary(dev_em, "dev/EM", summary_writer, global_step)


                #     # Early stopping based on dev EM. You could switch this to use F1 instead.
                #     if best_dev_em is None or dev_em > best_dev_em:
                #         best_dev_em = dev_em
                #         logging.info("Saving to %s..." % bestmodel_ckpt_path)
                #         self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()

def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)


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

    print("Loading dataset")
    dataset, img_features, train_data = get_data(config)
    vocab_size = len(dataset['ix_to_word'].keys())

    ### Train
    with tf.Session(config=gpu_config) as sess:

            vqa_model = VQAModel(config, vocab_size)
            # Train
            vqa_model.train(sess, dataset, img_features, train_data)

    ### Test


    
