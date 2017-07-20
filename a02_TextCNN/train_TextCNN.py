import tensorflow as tf
import numpy as np
import os
import word2vec

from model_TextCNN import TextCNN
from tflearn.data_utils import pad_sequences
from evaluate import evaluate
from tqdm import tqdm
from data_util_zhihu import load_data_multilabel_new, create_voabulary, \
    create_voabulary_label, transform_multilabel_as_multihot

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("num_classes", 1999, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
# tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.65, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 100, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 100, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 17, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
tf.app.flags.DEFINE_string("training_data_path", "train-zhihu6-title-desc.txt",
                           "path of training data.")
tf.app.flags.DEFINE_integer("num_filters", 256, "number of filters")
tf.app.flags.DEFINE_string("word2vec_model_path", "zhihu-word2vec-title-desc.bin-100",
                           "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_boolean("multi_label_flag", True, "use multi label or single label.")
filter_sizes = [1, 2, 3, 4, 5, 6, 7]  # [1,2,3,4,5,6,7]


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    """
    num_batches = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, num_batches)]


def batch_iter(data_x, data_y, batch_size, shuffle=True):
    assert len(data_x) == len(data_y)

    size = len(data_x)
    index_array = np.arange(size)
    if shuffle:
        np.random.shuffle(index_array)

    batch = make_batches(size, batch_size)
    for start, end in batch:
        batch_ids = index_array[start: end]
        yield data_x[batch_ids], data_y[batch_ids]


class WordIndexTransform:
    def __init__(self):
        self.word2index, self.index2word = create_voabulary(
            word2vec_model_path=FLAGS.word2vec_model_path,
            name_scope='cnn2'
        )
        self.vocab_size = len(self.word2index)
        self.word2index_label, self.index2word_label = create_voabulary_label(
            name_scope='cnn2'
        )


def load_data(word2index, word2index_label):
    train, test, _ = load_data_multilabel_new(
        word2index, word2index_label,
        multi_label_flag=True,
        traning_data_path=FLAGS.training_data_path,
        transform=False
    )

    trainX, trainY = train
    testX, testY = test

    trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len)
    testX = pad_sequences(testX, maxlen=FLAGS.sentence_len)
    return trainX, trainY, testX, testY


def format_word_embedding(index2word, vocab_size, word2vec_model_path=None):
    print('using pre-trained word embedding.')
    print('word2vec_model_path: ', word2vec_model_path)
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word_embedding_list = [[]] * vocab_size
    word_embedding_list[0] = np.zeros(FLAGS.embed_size)

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)
    count_exist = 0
    count_non_exist = 0

    for i in range(1, vocab_size):
        word = index2word[i]
        if word in word2vec_model:
            word_embedding_list[i] = word2vec_model[word]
            count_exist += 1
        else:
            word_embedding_list[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_non_exist += 1
    print("word exists embedding:", count_exist, " ;word not exist embedding:", count_non_exist)
    return np.array(word_embedding_list)


def get_label_using_logits(logits, index2word_label, top_number=5):
    index_list = np.argsort(logits)[-top_number:]  # print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list = index_list[::-1]
    label_list = []
    for index in index_list:
        label = index2word_label[index]
        label_list.append(label)
    return label_list


def do_eval(sess, textcnn, x, y, index2word_label):
    y = [transform_multilabel_as_multihot(i) for i in y]
    feed_dict = {
        textcnn.input_x: x,
        textcnn.input_y: y,
        textcnn.dropout_keep_prob: 1
    }
    logits = sess.run([textcnn.logits], feed_dict)
    result = [get_label_using_logits(l, index2word_label) for l in logits]
    y = [[index2word_label[i] for i in item] for item in y]
    print(evaluate(zip(result, y)))
    with open('evaluate.txt', 'a') as f:
        f.write(str(evaluate(zip(result, y))) + '\n')


def main(_):
    transform = WordIndexTransform()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print('Load data')
    trainX, trainY, testX, testY = load_data(transform.word2index, transform.word2index_label)

    with tf.Session(config=config) as sess:
        # Instantiate Model
        print('Instantiate Model')
        textcnn = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.num_classes,
                          FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                          FLAGS.decay_rate, FLAGS.sentence_len, transform.vocab_size,
                          FLAGS.embed_size, FLAGS.is_training)
        # Initialize Saver
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + 'checkpoint'):
            print('Restoring Variables from Checkpoint')
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:
                word_embedding = format_word_embedding(
                    transform.index2word,
                    transform.vocab_size,
                    FLAGS.word2vec_model_path
                )
                word_embedding = tf.constant(word_embedding, dtype=tf.float32)
                sess.run(tf.assign(textcnn.Embedding, word_embedding))
        # get current epoch
        curr_epoch = sess.run(textcnn.epoch_step)

        # Feed Data & Training
        batch_size = FLAGS.batch_size
        for epoch in tqdm(range(curr_epoch, FLAGS.num_epochs), total=len(trainY)):
            loss, counter = 0, 0

            for x, y in batch_iter(trainX, trainY, batch_size):
                y = [transform_multilabel_as_multihot(i) for i in y]
                feed_dict = {
                    textcnn.input_x: x,
                    textcnn.input_y: y,
                    textcnn.dropout_keep_prob: 0.5
                }

                curr_loss, _ = sess.run([textcnn.loss_val, textcnn.train_op], feed_dict)
                loss += curr_loss
                counter += 1
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\t" % (epoch, counter, loss / float(counter)))
        sess.run(textcnn.epoch_increment)

        # Validation
        save_path = FLAGS.ckpt_dir + 'model.ckpt'
        saver.save(sess, save_path, global_step=epoch)

        do_eval(sess, textcnn, testX, testY, transform.index2word_label)


if __name__ == "__main__":
    tf.app.run()


