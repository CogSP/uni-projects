import tensorflow as tf
import numpy as np

class RotatEModel(object):
    def __init__(self, num_entities, num_relations, embedding_dim=16, margin=6.0):
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.num_entities = num_entities
        self.num_relations = num_relations

        self._build_graph()

    def _build_graph(self):
        self.heads = tf.placeholder(tf.int32, [None])
        self.relations = tf.placeholder(tf.int32, [None])
        self.tails = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.float32, [None])  # 1 for pos, -1 for neg

        initializer = tf.contrib.layers.xavier_initializer()

        self.ent_real = tf.get_variable("ent_real", [self.num_entities, self.embedding_dim], initializer=initializer)
        self.ent_imag = tf.get_variable("ent_imag", [self.num_entities, self.embedding_dim], initializer=initializer)

        self.rel_phase = tf.get_variable("rel_phase", [self.num_relations, self.embedding_dim], initializer=initializer)

        # lookup embeddings
        h_re = tf.nn.embedding_lookup(self.ent_real, self.heads)
        h_im = tf.nn.embedding_lookup(self.ent_imag, self.heads)
        t_re = tf.nn.embedding_lookup(self.ent_real, self.tails)
        t_im = tf.nn.embedding_lookup(self.ent_imag, self.tails)

        phase = tf.nn.embedding_lookup(self.rel_phase, self.relations)
        phase = phase / (np.pi)

        r_re = tf.cos(phase)
        r_im = tf.sin(phase)

        # apply rotation: (h * r) - t
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        re_diff = hr_re - t_re
        im_diff = hr_im - t_im

        self.score = self.margin - tf.reduce_sum(tf.abs(re_diff) + tf.abs(im_diff), axis=1)

        self.loss = tf.reduce_mean(tf.nn.softplus(-self.labels * self.score))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def get_score_op(self, sess, h_idx, r_idx, candidate_tails):
        h_idx_batch = [h_idx] * len(candidate_tails)
        r_idx_batch = [r_idx] * len(candidate_tails)
        feed_dict = {
            self.heads: h_idx_batch,
            self.relations: r_idx_batch,
            self.tails: candidate_tails,
            self.labels: [1.0] * len(candidate_tails)
        }
        return sess.run(self.score, feed_dict=feed_dict)
    
    def get_saver(self):
        return tf.compat.v1.train.Saver()

