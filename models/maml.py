import numpy as np
import sys
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import xent, conv_block

FLAGS = flags.FLAGS


class MAML:
    def __init__(self, dim_input=1, dim_output=1, meta_test_num_inner_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.inner_update_lr = FLAGS.inner_update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.meta_test_num_inner_updates = meta_test_num_inner_updates
        self.loss_func = xent  # cross entropy loss
        self.dim_hidden = FLAGS.num_filters
        self.forward = self.forward_conv
        self.construct_weights = self.construct_conv_weights
        self.channels = 1
        self.img_size = int(np.sqrt(self.dim_input / self.channels))

    def construct_model(self, prefix='maml'):
        # a: group of data for calculating inner gradient
        # b: group of data for evaluating modified weights and computing meta gradient
        self.inputa = tf.placeholder(tf.float32)
        self.inputb = tf.placeholder(tf.float32)
        self.labela = tf.placeholder(tf.float32)
        self.labelb = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as training_scope:
            # outputbs[i] and lossesb[i] are the output and loss after i+1 inner gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            # number of loops in the inner training loop
            num_inner_updates = max(self.meta_test_num_inner_updates, FLAGS.num_inner_updates)
            outputbs = [[]] * num_inner_updates
            lossesb = [[]] * num_inner_updates
            accuraciesb = [[]] * num_inner_updates

            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights - these should NOT be directly modified by the
                # inner training loop
                self.weights = weights = self.construct_weights()

            # self.weights means meta params in MAML

            def task_inner_loop(inp, reuse=True):
                """
					Perform gradient descent for one task in the meta-batch (i.e. inner-loop).
					Args:
						inp: a tuple (inputa, inputb, labela, labelb), where inputa and labela are the inputs and
							labels used for calculating inner loop gradients and inputb and labelb are the inputs and
							labels used for evaluating the model after inner updates.
						reuse: reuse the model parameters or not. Hint: You can just pass its default value to the 
							forwawrd function
					Returns:
						task_output: a list of outputs, losses and accuracies at each inner update
				"""
                inputa, inputb, labela, labelb = inp
                ### inputa for [N, K, 784]. labela for [N, K, N]
                labela = tf.reshape(labela, shape=[-1, FLAGS.n_way])
                labelb = tf.reshape(labelb, shape=[-1, FLAGS.n_way])
                #############################
                #### YOUR CODE GOES HERE ####
                # perform num_inner_updates to get modified weights
                # modified weights should be used to evaluate performance
                # Note that at each inner update, always use inputa and labela for calculating gradients
                # and use inputb and labelb for evaluating performance
                # HINT: you may wish to use tf.gradients()

                # output, loss, and accuracy of group a before performing inner gradientupdate

                # lists to keep track of outputs, losses, and accuracies of group b for each inner_update
                # where task_outputbs[i], task_lossesb[i], task_accuraciesb[i] are the output, loss, and accuracy
                # after i+1 inner gradient updates
                # inner gradient descent
                task_outputa, task_lossa, task_accuracya = None, None, None
                task_outputbs, task_lossesb, task_accuraciesb = [], [], []
                outputa = self.forward_conv(inputa, weights=weights, reuse=reuse)
                lossa = self.loss_func(outputa, labela)
                task_outputa = outputa
                task_lossa = lossa
                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa, dim=1), axis=1),
                                                             tf.argmax(labela, axis=1))
                # first inner_loop update
                grads = tf.gradients(lossa, list(weights.values()))
                inner_grads = dict(zip(weights.keys(), grads))
                new_weights = dict(zip(weights.keys(),
                                   [weights[key] - self.inner_update_lr * inner_grads[key] for key in
                                    weights.keys()]))
                outputb = self.forward(inputb, new_weights, reuse=True)
                # meta-test loss
                lossb = self.loss_func(outputb, labelb)
                # record T0 pred and loss for meta-test
                task_outputbs.append(outputb)
                task_lossesb.append(lossb)

                for _ in range(1, num_inner_updates):
                    outputa = self.forward_conv(inputa, weights=new_weights, reuse=True)
                    lossa = self.loss_func(outputa, labela)
                    grads = tf.gradients(lossa, list(new_weights.values()))
                    inner_grads = dict(zip(new_weights.keys(), grads))
                    new_weights = dict(zip(new_weights.keys(),
                                       [new_weights[key] - self.inner_update_lr * inner_grads[key] for key in
                                        new_weights.keys()]))
                    outputb = self.forward_conv(inputb, weights=new_weights, reuse=True)

                    lossb = self.loss_func(outputb, labelb)
                    task_outputbs.append(outputb)
                    task_lossesb.append(lossb)
                for i in range(num_inner_updates):
                    task_accuraciesb.append(
                        tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[i], dim=1), axis=1),
                                                    tf.argmax(labelb, axis=1)))
                #############################

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb, task_accuracya, task_accuraciesb]

                return task_output

            # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
            unused = task_inner_loop((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)
            out_dtype = [tf.float32, [tf.float32] * num_inner_updates, tf.float32, [tf.float32] * num_inner_updates]
            out_dtype.extend([tf.float32, [tf.float32] * num_inner_updates])
            result = tf.map_fn(task_inner_loop, elems=(self.inputa, self.inputb, self.labela, self.labelb),
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result

        ## Performance & Optimization
        self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
        self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in
                                              range(num_inner_updates)]
        # after the map_fn
        self.outputas, self.outputbs = outputas, outputbs
        self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
        self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size)
                                                      for j in range(num_inner_updates)]

        if FLAGS.meta_train_iterations > 0:
            optimizer = tf.train.AdamOptimizer(self.meta_lr)
            self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_inner_updates - 1])
            # optimizer.compute_gradients Return A list of (gradient, variable) pairs. Variable is always present, but gradient can be None.
            self.metatrain_op = optimizer.apply_gradients(gvs)
            # Apply gradients to variables.

        ## Summaries
        tf.summary.scalar(prefix + 'Pre-update loss', total_loss1)
        tf.summary.scalar(prefix + 'Pre-update accuracy', total_accuracy1)

        for j in range(num_inner_updates):
            tf.summary.scalar(prefix + 'Post-update loss, step ' + str(j + 1), total_losses2[j])
            tf.summary.scalar(prefix + 'Post-update accuracy, step ' + str(j + 1), total_accuracies2[j])

    ### Network construction functions
    def construct_conv_weights(self):
        '''represent weights as a dictionary'''
        weights = {}

        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden],
                                           initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output]), name='w5')
        weights['b5'] = tf.Variable(tf.zeros([self.dim_output]), name='b5')
        return weights

    def forward_conv(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        # B*H*W*dim_hidden
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        # B*H*W*dim_hidden
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        # B*H*W*dim_hidden
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
        # B*H*W*dim_hidden
        hidden4 = tf.reduce_mean(hidden4, [1, 2])
        # B*dim_hidden

        return tf.matmul(hidden4, weights['w5']) + weights['b5']
    # B*dim_out
