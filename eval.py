# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from datetime import datetime

import cnn
import util

FLAGS = tf.app.flags.FLAGS

# train parameters
this_dir = os.path.abspath(os.path.dirname(__file__))
tf.app.flags.DEFINE_string('data_dir', os.path.join(this_dir, 'data'), 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'model'), 'Directory of the saved checkpoint files')
tf.app.flags.DEFINE_boolean('savefig', True, 'Whether save PR-curve image or not')

def evaluate(eval_data, config):
    """ Build evaluation graph and run. """


    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            m = cnn.Model(config, is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            losses = []
            precision = []
            recall = []
            batches = util.batch_iter(eval_data, batch_size=config['batch_size'], num_epochs=1, shuffle=False)
            for batch in batches:
                x_batch, y_batch, _ = zip(*batch)
                feed = {m.inputs: np.array(x_batch), m.labels: np.array(y_batch)}
                loss, auc = sess.run([m.total_loss, m.auc_op], feed_dict=feed)
                losses.append(loss)
                pre, rec = zip(*auc)
                precision.append(pre)
                recall.append(rec)

            avg_precision = np.mean(np.array(precision), axis=0)
            avg_recall = np.mean(np.array(recall), axis=0)
            auc = np.trapz(avg_precision, x=avg_recall, dx=5)
            f1 = (2.0 * avg_precision[5] * avg_recall[5]) / (avg_precision[5] + avg_recall[5])
            print '%s: loss = %.6f, f1 = %.4f, auc = %.4f' % (datetime.now(), np.mean(losses), f1, auc)

    return avg_precision, avg_recall



def main(argv=None):
    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir
    source_path = os.path.join(FLAGS.data_dir, "ids.txt")
    target_path = os.path.join(FLAGS.data_dir, "clean.label")
    _, data = util.read_data(source_path, target_path, restore_param['sent_len'])

    pre, rec = evaluate(data, restore_param)

    if FLAGS.savefig:
        import matplotlib.pyplot as plt
        plt.plot(rec, pre)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(FLAGS.train_dir, 'pr_curve.png'))

if __name__ == '__main__':
    tf.app.run()

