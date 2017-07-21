# -*- coding: utf-8 -*-
# TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
# print("started...")
import tensorflow as tf


class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size,
                 is_training, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False,
                 clip_gradients=5.0, decay_rate_big=0.50):
        """init all hyperparameter here"""
        # set hyperparamter
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")  # ADD learning_rate
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.multi_label_flag = multi_label_flag
        self.clip_gradients = clip_gradients

        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")  # X
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],
                                                 name="input_y_multilabel")
        self.input_y = self.input_y_multilabel
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        with tf.name_scope("embedding"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            self.W_projection = tf.get_variable("W_projection", shape=[self.num_filters_total, self.num_classes],
                                                initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection",
                                                shape=[self.num_classes])

        self.logits = self.inference()
        self.loss_val = self.loss_multi_label()
        self.train_op = self.train
        self.predictions = tf.argmax(self.logits, 1, name="predictions")

    def inference(self):
        """main computation graph here: 1.embedding-->2.average-->3.linear classifier"""
        # 1. get embedding of words in the sentence
        embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        sentence_embeddings_expanded = tf.expand_dims(embedded_words, -1)

        # 2. loop each filter size.
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                filter = tf.get_variable(
                    "filter-%s" % filter_size,
                    [filter_size, self.embed_size, 1, self.num_filters],
                    initializer=self.initializer
                )
                conv = tf.nn.conv2d(
                    sentence_embeddings_expanded,
                    filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool"
                )
                pooled_outputs.append(pooled)
        # 3. combine all pooled features, and flatten the feature.output' shape is a [1,None]
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])

        # 4. add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)

        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(h_drop, self.W_projection) + self.b_projection
        return logits

    def loss_multi_label(self, l2_lambda=0.00001):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,
                                                             logits=self.logits)
            print("sigmoid_cross_entropy_with_logits.losses:", losses)
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss += l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        self.learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True
        )
        train_op = tf.contrib.layers.optimize_loss(
            self.loss_val,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer="Adam",
            clip_gradients=self.clip_gradients
        )
        return train_op

    def reset_learning_rate(self, lr):
        self.learning_rate = tf.train.exponential_decay(
            lr,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True
        )
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss_val,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer="Adam",
            clip_gradients=self.clip_gradients
        )
