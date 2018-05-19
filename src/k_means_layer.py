import numpy
import tensorflow as tf


class KMeansLayer:
    def __init__(self, latent_space, cluster_amount, initial_clusters, batch_size=1000):
        self.batch_size = batch_size
        self.cluster_amount = cluster_amount
        self.latent_space = tf.layers.flatten(latent_space)
        centroid_tensor = tf.constant(initial_clusters, dtype=tf.float32)
        self.centroids = tf.get_variable('centroid_weight', initializer=centroid_tensor)

    def get_soft_assignments(self):
        z_expanded = tf.reshape(self.latent_space, [self.batch_size, 1] + self.latent_space.get_shape().as_list()[1:])
        z_expanded = tf.tile(z_expanded, (1, self.cluster_amount, 1))
        u_expanded = tf.reshape(self.centroids, [1, self.cluster_amount] + self.centroids.get_shape().as_list()[1:])
        u_expanded = tf.tile(u_expanded, (self.batch_size, 1, 1))

        distances_from_cluster_centers = tf.norm(z_expanded - u_expanded, 2, axis=2)
        qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
        qij_numerator = 1 / qij_numerator
        normalizer_q = tf.reshape(tf.reduce_sum(qij_numerator, axis=1), [self.batch_size, 1])
        return qij_numerator / normalizer_q
