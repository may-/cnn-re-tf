# -*- coding: utf-8 -*-

##########################################################
#
# Attention-based Convolutional Neural Network
#   for Context-wise Learning
#
#
#   Note: this implementation is mostly based on
#   https://github.com/yuhaozhang/sentence-convnet/blob/master/eval.py
#
##########################################################

from datetime import datetime
import os
import tensorflow as tf
import numpy as np
import util


FLAGS = tf.app.flags.FLAGS
this_dir = os.path.abspath(os.path.dirname(__file__))
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'models', 'er-cnn'), 'Directory of the checkpoint files')


def evaluate(eval_data, config):
    """ Build evaluation graph and run. """

    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            if config.has_key('contextwise') and config['contextwise']:
                import cnn_context
                m = cnn_context.Model(config, is_train=False)
            else:
                import cnn
                m = cnn.Model(config, is_train=False)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            print "\nStart evaluation on test set ...\n"
            if config.has_key('contextwise') and config['contextwise']:
                left_batch, middle_batch, right_batch, y_batch, _ = zip(*eval_data)
                feed = {m.left: np.array(left_batch),
                        m.middle: np.array(middle_batch),
                        m.right: np.array(right_batch),
                        m.labels: np.array(y_batch)}
            else:
                x_batch, y_batch, _ = zip(*eval_data)
                feed = {m.inputs: np.array(x_batch), m.labels: np.array(y_batch)}
            loss, eval = sess.run([m.total_loss, m.eval_op], feed_dict=feed)
            pre, rec = zip(*eval)

            auc = util.calc_auc_pr(pre, rec)
            f1 = (2.0 * pre[5] * rec[5]) / (pre[5] + rec[5])
            print '%s: loss = %.6f, p = %.4f, r = %4.4f, f1 = %.4f, auc = %.4f' % (datetime.now(), loss,
                                                                                   pre[5], rec[5], f1, auc)
    return pre, rec


def main(argv=None):
    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir

    if restore_param.has_key('contextwise') and restore_param['contextwise']:
        source_path = os.path.join(restore_param['data_dir'], "ids")
        target_path = os.path.join(restore_param['data_dir'], "target.txt")
        _, data = util.read_data_contextwise(source_path, target_path, restore_param['sent_len'],
                                             train_size=restore_param['train_size'])
    else:
        source_path = os.path.join(restore_param['data_dir'], "ids.txt")
        target_path = os.path.join(restore_param['data_dir'], "target.txt")
        _, data = util.read_data(source_path, target_path, restore_param['sent_len'],
                                 train_size=restore_param['train_size'])

    pre, rec = evaluate(data, restore_param)
    util.dump_to_file(os.path.join(FLAGS.train_dir, 'results.cPickle'), {'precision': pre, 'recall': rec})


if __name__ == '__main__':
    tf.app.run()
