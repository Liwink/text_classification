# -*- coding: utf-8 -*-
# training the model.
# process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np
from p7_TextCNN_model import TextCNN
from data_util_zhihu import load_data_multilabel_new, create_voabulary, \
    create_voabulary_label, transform_multilabel_as_multihot
from tflearn.data_utils import to_categorical, pad_sequences
import os
import word2vec
from evaluate import evaluate
import pickle

# configuration
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("test", False, "using small sample")
TEST = FLAGS.test

tf.app.flags.DEFINE_integer("num_classes", 1999, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
if not TEST:
    tf.app.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
else:
    tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 0.65, "Rate of decay for learning rate.")  # 0.65一次衰减多少
# tf.app.flags.DEFINE_integer("num_sampled",50,"number of noise sampling") #100
tf.app.flags.DEFINE_string("ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 100, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 100, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 17, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
# tf.app.flags.DEFINE_string("cache_path","text_cnn_checkpoint/data_cache.pik","checkpoint location for the model")
# train-zhihu4-only-title-all.txt
# tf.app.flags.DEFINE_string("traning_data_path", "train-zhihu4-only-title-all.txt",
#                            "path of traning data.")  # O.K.train-zhihu4-only-title-all.txt-->training-data/test-zhihu4-only-title.txt--->'training-data/train-zhihu5-only-title-multilabel.txt'
tf.app.flags.DEFINE_string("traning_data_path", "train-zhihu6-title-desc.txt",
                           "path of traning data.")
tf.app.flags.DEFINE_integer("num_filters", 256, "number of filters")  # 256--->512
tf.app.flags.DEFINE_string("word2vec_model_path", "zhihu-word2vec-title-desc.bin-100",
                           "word2vec's vocabulary and vectors")  # zhihu-word2vec.bin-100-->zhihu-word2vec-multilabel-minicount15.bin-100
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


# 1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    # 1.load data(X:list of lint,y:int).
    # if os.path.exists(FLAGS.cache_path):  # 如果文件系统中存在，那么加载故事（词汇表索引化的）
    #    with open(FLAGS.cache_path, 'r') as data_f:
    #        trainX, trainY, testX, testY, vocabulary_index2word=pickle.load(data_f)
    #        vocab_size=len(vocabulary_index2word)
    # else:
    if 1 == 1:
        trainX, trainY, testX, testY = None, None, None, None
        vocabulary_word2index, vocabulary_index2word = create_voabulary(word2vec_model_path=FLAGS.word2vec_model_path,
                                                                        name_scope="cnn2")  # simple='simple'
        vocab_size = len(vocabulary_word2index)
        print("cnn_model.vocab_size:", vocab_size)
        vocabulary_word2index_label, vocabulary_index2word_label = create_voabulary_label(name_scope="cnn2")
        # if FLAGS.multi_label_flag:
        #     FLAGS.traning_data_path = 'train-zhihu6-title-desc.txt'
        train, test, _ = load_data_multilabel_new(vocabulary_word2index, vocabulary_word2index_label,
                                                  multi_label_flag=FLAGS.multi_label_flag,
                                                  traning_data_path=FLAGS.traning_data_path,
                                                  transform=False)  # ,traning_data_path=FLAGS.traning_data_path
        trainX, trainY = train
        testX, testY = test

        if TEST:
            trainX = trainX[:10000]
            trainY = trainY[:10000]
            testX = testX[:100]
            testY = testY[:100]
        # 2.Data preprocessing.Sequence padding
        print("start padding & transform to one hot...")
        trainX = pad_sequences(trainX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
        testX = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
        # with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
        #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
        print("trainX[0]:", trainX[0])  # ;print("trainY[0]:", trainY[0])
        # Converting labels to binary vectors
        print("end padding & transform to one hot...")
    # 2.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Instantiate Model
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.num_classes,
                          FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                          FLAGS.decay_rate, FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training,
                          multi_label_flag=FLAGS.multi_label_flag)
        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word embedding
                assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN,
                                                 word2vec_model_path=FLAGS.word2vec_model_path)
        curr_epoch = sess.run(textCNN.epoch_step)

        sess.run(tf.assign(textCNN.learning_rate, tf.constant(FLAGS.learning_rate)))
        print('curr learning rate: ', sess.run(textCNN.learning_rate))
        import pdb; pdb.set_trace()

        # 3.feed data & training
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, counter = 0.0, 0

            index_array = np.arange(number_of_training_data)
            np.random.shuffle(index_array)

            batch = make_batches(number_of_training_data, batch_size)

            for start, end in batch:
                # shuffle
                batch_ids = index_array[start: end]

                # if epoch == 0 and counter == 0:
                #     print("trainX[start:end]:", trainX[start:end])  # ;print("trainY[start:end]:",trainY[start:end])
                feed_dict = {textCNN.input_x: trainX[batch_ids], textCNN.dropout_keep_prob: 0.5}

                train_y = [transform_multilabel_as_multihot(y) for y in trainY[batch_ids]]
                if not FLAGS.multi_label_flag:
                    feed_dict[textCNN.input_y] = train_y
                else:
                    feed_dict[textCNN.input_y_multilabel] = train_y

                curr_loss, _ = sess.run([textCNN.loss_val, textCNN.train_op],
                                                  feed_dict)  # curr_acc--->TextCNN.accuracy
                loss, counter = loss + curr_loss, counter + 1
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f" % (
                    epoch, counter, loss / float(counter)))

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            # 4.validation
            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)
                eval_loss = do_eval(sess, textCNN, testX, testY, batch_size, vocabulary_index2word_label)
                print("Epoch %d Validation Loss:%.3f" % (epoch, eval_loss))
                # save model to checkpoint

            print('global step: {}'.format(sess.run(textCNN.global_step)))
            print('learning rate: ', sess.run(textCNN.learning_rate))
        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss = do_eval(sess, textCNN, testX, testY, batch_size, vocabulary_index2word_label)

    pass


def assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, textCNN, word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:", word2vec_model_path)
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.Embedding,
                                   word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, textCNN, evalX, evalY, batch_size, vocabulary_index2word_label):
    number_examples = len(evalX)
    eval_loss, eval_counter = 0.0, 0
    logits_list = np.array([])
    for start, end in make_batches(number_examples, batch_size):
        feed_dict = {textCNN.input_x: evalX[start:end], textCNN.dropout_keep_prob: 1}
        eval_y = [transform_multilabel_as_multihot(y) for y in evalY[start:end]]
        if not FLAGS.multi_label_flag:
            feed_dict[textCNN.input_y] = eval_y
        else:
            feed_dict[textCNN.input_y_multilabel] = eval_y
        curr_eval_loss, logits= sess.run([textCNN.loss_val, textCNN.logits],
                                                         feed_dict)  # curr_eval_acc--->textCNN.accuracy
        if not len(logits_list):
            logits_list = logits
        else:
            logits_list = np.concatenate((logits_list, logits))
        # label_list_top5 = get_label_using_logits(logits_[0], vocabulary_index2word_label)
        # curr_eval_acc=calculate_accuracy(list(label_list_top5), evalY[start:end][0],eval_counter)
        eval_loss, eval_counter = eval_loss + curr_eval_loss, eval_counter + 1
    result = [get_label_using_logits(l, vocabulary_index2word_label) for l in logits_list]
    y = [[vocabulary_index2word_label[i] for i in item] for item in evalY]
    print(evaluate(zip(result, y)))
    with open('evaluate.txt', 'a') as f:
        f.write(str(evaluate(zip(result, y))) + '\n')
    return eval_loss / float(eval_counter)


# 从logits中取出前五 get label using logits
def get_label_using_logits(logits, vocabulary_index2word_label, top_number=5):
    index_list = np.argsort(logits)[-top_number:]  # print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list = index_list[::-1]
    label_list = []
    for index in index_list:
        label = vocabulary_index2word_label[index]
        # ('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        label_list.append(label)
    return label_list


# 统计预测的准确率
def calculate_accuracy(labels_predicted, labels, eval_counter):
    label_nozero = []
    # print("labels:",labels)
    labels = list(labels)
    for index, label in enumerate(labels):
        if label > 0:
            label_nozero.append(index)
    if eval_counter < 2:
        print("labels_predicted:", labels_predicted, " ;labels_nozero:", label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)


if __name__ == "__main__":
    tf.app.run()
