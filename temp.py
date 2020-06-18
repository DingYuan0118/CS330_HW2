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
            N = labela.shape[-1]
            ### inputa for [N, K, 784]. labela for [N, K, N]
            inputa = tf.reshape(inputa, shape=[-1, self.img_size, self.img_size, 1])
            inputb = tf.reshape(inputb, shape=[-1, self.img_size, self.img_size, 1])
            labela = tf.reshape(inputb, shape=[-1, N])
            labelb = tf.reshape(inputb, shape=[-1, N])
            #############################
            #### YOUR CODE GOES HERE ####
            # perform num_inner_updates to get modified weights
            # modified weights should be used to evaluate performance
            # Note that at each inner update, always use inputa and labela for calculating gradients
            # and use inputb and labelb for evaluating performance
            # HINT: you may wish to use tf.gradients()

            # output, loss, and accuracy of group a before performing inner gradientupdate
            task_outputa, task_lossa, task_accuracya = None, None, None
            task_outputa = self.forward_conv(inputa, weights=weights)
            task_lossa = self.loss_func(task_outputa, labela)
            task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa, dim=1), axis=1),
                                                         tf.argmax(labela, axis=1))

            # lists to keep track of outputs, losses, and accuracies of group b for each inner_update
            # where task_outputbs[i], task_lossesb[i], task_accuraciesb[i] are the output, loss, and accuracy
            # after i+1 inner gradient updates
            # inner gradient descent

            task_outputbs, task_lossesb, task_accuraciesb = [], [], []
            outputa = self.forward_conv(inputa, weights=weights)
            lossa = self.loss_func(outputa, labela)
            grads = tf.gradients(lossa, weights.values())
            inner_grads = dict(zip(weights.keys(), grads))
            weights = dict(zip(self.weights.keys(),
                               [self.weights[key] - self.inner_update_lr * inner_grads[key] for key in
                                self.weights.keys()]))
            for _ in range(num_inner_updates):
                outputa = self.forward_conv(inputa, weights=weights)
                lossa = self.loss_func(outputa, labela)
                grads = tf.gradients(lossa, weights.values())
                inner_grads = dict(zip(weights.keys(), grads))
                weights = dict(zip(self.weights.keys(),
                                   [self.weights[key] - self.inner_update_lr * inner_grads[key] for key in
                                    self.weights.keys()]))
                outputb = self.forward_conv(inputb, weights=weights)

                lossb = self.loss_func(outputb, labelb)
                accb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputb, dim=1), axis=1),
                                                   tf.argmax(labelb, axis=1)))
                task_outputbs.append(outputb)
                task_lossesb.append(lossb)
                task_accuraciesb.append(accb)
                #############################

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb, task_accuracya, task_accuraciesb]

            return task_output