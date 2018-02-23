import time
import logging
import os
import sys
import h5py
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

logging.basicConfig(level=logging.INFO)

class Config:
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
    seed = 123

    ### checkpoints
    save_checkpoint_every = 25000

    input_image_h5  = './data_img.h5'
    input_ques_h5   = './data_prepro.h5'
    input_json      = './data_prepro.json'

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
        with tf.variable_scope("VQAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.add_variables()
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_variables(self):
        """
        add variables of the model to the graph
        
        """
        self.embed_ques_W =  tf.get_variable("embed_ques_W", tf.random_uniform([self.vocabulary_size, self.input_embedding_size], -0.08, 0.08))

        self.lstm_1 = rnn_cell.LSTMCell(rnn_size, input_embedding_size, use_peepholes=True)
        self.lstm_dropout_1 = rnn_cell.DropoutWrapper(self.lstm_1, output_keep_prob = self.keep_prob)
        
        self.lstm_2 = rnn_cell.LSTMCell(rnn_size, rnn_size, use_peepholes=True)
        self.lstm_dropout_2 = rnn_cell.DropoutWrapper(self.lstm_2, output_keep_prob = self.keep_prob)
        
        self.stacked_lstm = rnn_cell.MultiRNNCell([self.lstm_dropout_1, self.lstm_dropout_2])

        # state-embedding
        random_embed_initializer = tf.random_uniform([2 * self.config.rnn_size * self.config.rnn_layer, self.config.common_embed_size], -0.08, 0.08)
        self.embed_state_W = tf.get_variable('embed_state_W', initializer=random_embed_initializer)
        self.embed_state_b = tf.get_variable('embed_state_b', tf.random_uniform([self.config.common_embed_size], -0.08, 0.08))
        
        # image-embedding
        self.embed_image_W = tf.get_variable('embed_image_W', tf.random_uniform([self.config.image_size, self.config.common_embed_size], -0.08, 0.08))
        self.embed_image_b = tf.get_variable('embed_image_b', tf.random_uniform([self.config.common_embed_size], -0.08, 0.08))
        
        # score-embedding
        self.embed_score_W = tf.get_variable('embed_score_W', tf.random_uniform([self.config.hidden_size, self.config.common_embed_size], -0.08, 0.08))
        self.embed_score_b = tf.get_variable('embed_score_b', tf.random_uniform([self.config.common_embed_size], -0.08, 0.08))

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

    def add_embedding_layer(self):
        embeddings = tf.nn.embedding_lookup(self.embed_ques_W, self.ques_placeholder)
        return embeddings

    def build_graph(self):
        """Builds the main part of the graph for the model
        """
        state = tf.zeros([self.batch_size, self.stacked_lstm.state_size])
        loss = 0.0
        for i in range(max_words_q):
            if i==0:
            ques_emb_linear = tf.zeros([self.batch_size, self.input_embedding_size])
            else:
                tf.get_variable_scope().reuse_variables()
            ques_emb_linear = tf.nn.embedding_lookup(self.embed_ques_W, question[:,i-1])

            ques_emb_drop = tf.nn.dropout(ques_emb_linear, 1-self.drop_out_rate)
            ques_emb = tf.tanh(ques_emb_drop)

            output, state = self.stacked_lstm(ques_emb, state)

        state_drop = tf.nn.dropout(state, self.keep_prob)
        state_emb = tf.tanh(tf.matmul(state_drop, self.embed_state_W). + self.embed_state_b)

        image_drop = tf.nn.dropout(image, self.keep_prob)
        image_emb = tf.tanh(tf.nn.xw_plus_b(image_drop, self.embed_image_W, self.embed_image_b))

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

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

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
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.config.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.
        Inputs:
          session: TensorFlow session
          batch: a Batch object
        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss

    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.
        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files
        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.
        Inputs:
          session: TensorFlow session
        """

        config = Config()

        tic = time.time()

        # Print number of model parameters
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.config.train_dir, "vqa.ckpt")
        bestmodel_dir = os.path.join(self.config.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "vqa_best.ckpt")

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.config.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while epoch < self.config.n_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in minibatches()

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()

def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)