# -*- coding: utf-8 -*-
from keras import backend as K
from keras.callbacks import Callback
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf
import functools
import numpy as np


def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value

    return wrapper


class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='weighted')
        _val_precision = precision_score(val_targ, val_predict, average='weighted')
        _val_accuracy = accuracy_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_accuracy.append(_val_accuracy)
        # print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return


class MetricsWithGenerator(Callback):

    def __init__(self, validation_generator):
        self.validation_generator = validation_generator
        self.val_true = validation_generator.classes

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracy = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs={}):
        score = self.model.evaluate_generator(self.validation_generator)
        self.validation_generator.reset()
        pred = self.model.predict_generator(self.validation_generator, verbose=1)
       # predicted_class_indices = [1 if x >= 0.5 else 0 for x in pred]
        predicted_class_indices = np.argmax(pred, axis=-1)

        precisions, recall, fscore, support = metrics.precision_recall_fscore_support(self.val_true,
                                                                                      predicted_class_indices,
                                                                                      average='weighted')
        loss = score[0]
        accuracy = score[1]

        self.val_f1s.append(fscore)
        self.val_recalls.append(recall)
        self.val_precisions.append(precisions)
        self.val_accuracy.append(accuracy)
        self.val_loss.append(loss)

        return
