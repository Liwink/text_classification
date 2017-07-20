import tensorflow as tf
import numpy as np


class TextCNN:
    def __init__(self, filter_sizes, num_filters, num_classes,
                 learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size, is_training=True,
                 initializer=tf.random_normal_initializer(stddev=0.1),
                 clip_gradients=5.0):
        # set hyperparamters
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.num_classes = num_classes
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate')
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.initializer = initializer
        self.clip_gradients = clip_gradients

        # add placeholder
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # initialize embedding variable
        with tf.name_scope('embedding'):
            self.Embedding = tf.get_variable(
                'Embedding',
                shape=[self.vocab_size, self.embed_size],
                initializer=self.initializer
            )
            self.W_projection = tf.get_variable(
                'W_projection',
                shape=[self.num_filters_total, self.num_classes],
                initializer=self.initializer
            )
            self.b_projection = tf.get_variable(
                'b_projection',
                shape=[self.num_classes]
            )

        self.embedded_words = None
        self.sentence_embeddings_expanded = None
        self.h_pool = None
        self.h_pool_flat = None
        self.h_drop = None
        self.logits = None
        self.inference()

        self.loss_val = self.loss_multi_label()
        self.train_op = self.train()

        self.accuracy = tf.constant(0.5)

        self.saver = tf.train.Saver()

    def inference(self):
        # 1. embedding of words in the sentence
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)

        # 2. loop each filter size
        pooled_output = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('convolution-pooling-{0}'.format(filter_size)):
                filter_w = tf.get_variable('filter-{}'.format(filter_size),
                                           [filter_size, self.embed_size, 1, self.num_filters],
                                           initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded,
                                    filter_w,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                b = tf.get_variable('b-{}'.format(filter_size), [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), 'relu')
                pooled = tf.nn.max_pool(h, strides=[1, 1, 1, 1], padding='VALID', name='pool',
                                        ksize=[1, self.sequence_length - filter_size + 1, 1, 1])
                pooled_output.append(pooled)
        # 3. combine all pooled features and flatten the feature.output
        self.h_pool = tf.concat(pooled_output, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        # 4. add dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        # 5. logits
        with tf.name_scope('output'):
            self.logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection

    def loss_multi_label(self, l2_lambda=0.00001):
        with tf.name_scope('loss'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.input_y,
                logits=self.logits
            )
            losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        return loss + l2_losses

    def train(self):
        learning_rate = tf.train.exponential_decay(
            self.learning_rate,
            self.global_step,
            self.decay_steps,
            self.decay_rate,
            staircase=True
        )
        train_op = tf.contrib.layers.optimize_loss(
            self.loss_val,
            global_step=self.global_step,
            learning_rate=learning_rate,
            optimizer='Adam',
            clip_gradients=self.clip_gradients
        )
        return train_op


def test():
    learning_rate = 0.01
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    dropout_keep_prob = 1  # 0.5
    filter_sizes = [3, 4, 5]
    num_filters = 128
    num_classes = 1

    textcnn = TextCNN(filter_sizes, num_filters, num_classes,
                      learning_rate, batch_size, decay_steps, decay_rate,
                      sequence_length, vocab_size, embed_size)
    with tf.Session() as sess:
        input_x = np.zeros((batch_size, sequence_length))
        input_y = np.zeros((batch_size, num_classes), dtype=np.int32)
        feed_dict = {
            textcnn.input_x: input_x,
            textcnn.input_y: input_y,
            textcnn.dropout_keep_prob: dropout_keep_prob,
        }
        sess.run(tf.global_variables_initializer())
        loss, logits = sess.run([textcnn.loss_val, textcnn.train_op],
                                feed_dict=feed_dict)
    return loss, logits


if __name__ == '__main__':
    print(test())
