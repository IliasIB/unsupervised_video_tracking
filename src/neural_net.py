import os
import numpy
import tensorflow
import tensorflow as tf
import pickle

from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot
from k_means_layer import KMeansLayer


class NeuralNetwork:
    def __init__(self):
        # Training Parameters
        self.learning_rate = 0.0001
        self.momentum = 0.9
        self.batch_size = 100
        self.k_means_batch_size = 1000
        self.num_clusters = 10

    def decoder(self, encoding):
        # Decoding
        # Deconvolutional Layer #3
        decoding_deconvolution_3 = tf.layers.conv2d_transpose(
            inputs=encoding,
            filters=120,
            kernel_size=[4, 4],
            padding="same",
            activation=None
        )
        # Upsampling Layer #2
        decoding_upsampling_2 = tf.image.resize_bilinear(
            images=decoding_deconvolution_3,
            size=[16, 16]
        )
        # Deconvolutional Layer #2
        decoding_deconvolution_2 = tf.layers.conv2d_transpose(
            inputs=decoding_upsampling_2,
            filters=50,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )
        # Upsampling Layer #1
        decoding_upsampling_1 = tf.image.resize_bilinear(
            images=decoding_deconvolution_2,
            size=[32, 32]
        )
        # Deconvolutional Layer #1
        decoding_deconvolution_1 = tf.layers.conv2d_transpose(
            inputs=decoding_upsampling_1,
            filters=50,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )
        # Logits
        logits = tf.layers.conv2d(
            inputs=decoding_deconvolution_1,
            filters=3,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="cnn_decoder"
        )
        return logits

    def encoder(self, image):
        # Input Layer
        input_layer = tf.reshape(image, [-1, 32, 32, 3])

        # Convolutional Layer #1
        encoding_convolution_1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=50,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        encoding_pool_1 = tf.layers.max_pooling2d(inputs=encoding_convolution_1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        encoding_convolution_2 = tf.layers.conv2d(
            inputs=encoding_pool_1,
            filters=50,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #2
        encoding_pool_2 = tf.layers.max_pooling2d(inputs=encoding_convolution_2, pool_size=[2, 2], strides=2)

        # Convolutional Layer #3
        encoding_convolution_3 = tf.layers.conv2d(
            inputs=encoding_pool_2,
            filters=120,
            kernel_size=[4, 4],
            padding="same",
            activation=None,
            name="cnn_encoder")
        return encoding_convolution_3

    def train_autoencoder(self, images, test_images, test_decoder=False):
        image = tf.placeholder('float32', [None, 32, 32, 3], name="image_input")
        encoder = self.encoder(image)
        decoder = self.decoder(encoder)

        loss = tf.reduce_mean(tf.pow(image - decoder, 2))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss)

        # Iterator to get next batch
        image_dataset = tf.data.Dataset.from_tensor_slices(images).batch(self.batch_size)
        iterator_data = image_dataset.make_initializable_iterator()
        image_iterator = iterator_data.get_next()

        # Iterator to get next test batch
        test_image_dataset = tf.data.Dataset.from_tensor_slices(test_images).batch(self.batch_size)
        iterator_test_data = test_image_dataset.make_one_shot_iterator()
        test_image_iterator = iterator_test_data.get_next()

        # Saver to save model
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # Amount of cycles to perform
            epoch_amount = 10

            for epoch in range(1, epoch_amount + 1):
                epoch_loss = 0
                session.run(iterator_data.initializer)

                for iteration in range(1, int(images.shape[0] / self.batch_size) + 1):
                    epoch_x = session.run(image_iterator)
                    _, c = session.run([optimizer, loss], feed_dict={image: epoch_x})
                    epoch_loss += c
                    print('Current run loss:', c)
                    print('Current epoch loss:', epoch_loss)
                    print('Epoch:', epoch, 'iteration:', iteration)
                print('Epoch', epoch, 'completed out of', epoch_amount, 'loss:', epoch_loss)

                # Save pre-training model
                saver.save(session, '../models/autoencoder_model/autoencoder_model')

            if test_decoder:
                self.test_decoder(decoder, image, session, test_image_iterator)

    def load_pre_training(self, images):
        with tf.Session() as session:
            saver = tf.train.import_meta_graph('../models/autoencoder_model/autoencoder_model.meta')
            saver.restore(session, tf.train.latest_checkpoint('../models/autoencoder_model'))

            graph = tf.get_default_graph()
            image_placeholder = graph.get_tensor_by_name("image_input:0")
            decoder = graph.get_tensor_by_name("cnn_decoder/Conv2D:0")

            encoded_images = session.run(decoder, feed_dict={image_placeholder: images})

            print("test")

    def test_decoder(self, decoder, image, session, test_image_iterator):
        test_batch = session.run(test_image_iterator)
        reconstruction_batch = session.run(decoder, feed_dict={image: test_batch})
        figure = pyplot.figure()
        subplot = figure.add_subplot(1, 2, 1)
        pyplot.imshow(test_batch[5])
        subplot.set_title("Original")
        subplot = figure.add_subplot(1, 2, 2)
        pyplot.imshow((reconstruction_batch[5]).astype(int))
        subplot.set_title("Reconstructed")
        pyplot.show()

    def train_k_means_and_autoencoder(self, images):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Extract encoder and decoder
            saver = tf.train.import_meta_graph('../models/autoencoder_model/autoencoder_model.meta')
            saver.restore(session, tf.train.latest_checkpoint('../models/autoencoder_model'))
            graph = tf.get_default_graph()
            image_placeholder = graph.get_tensor_by_name("image_input:0")
            encoder = graph.get_tensor_by_name("cnn_encoder/Conv2D:0")
            decoder = graph.get_tensor_by_name("cnn_decoder/Conv2D:0")

            # Get soft assignments
            initial_centroids = self.__get_initial_clusters(images)
            kmeans_layer = KMeansLayer(encoder, self.num_clusters, initial_centroids)
            soft_assignments = kmeans_layer.get_soft_assignments()

            # Calculate loss
            weight_k_means = 0.1
            weight_reconstruction = 1
            reconstruction_loss = tf.reduce_mean(tf.pow(image_placeholder - decoder, 2))
            k_means_loss = self.get_k_means_loss(kmeans_layer.centroids, soft_assignments, encoder)
            loss = weight_k_means * k_means_loss + weight_reconstruction * reconstruction_loss

            optimizer = tf.train.RMSPropOptimizer(self.learning_rate, name="full/conv2d/kernel/RMSProp").minimize(loss)

            # Amount of cycles to perform
            epoch_amount = 10

            # Iterator to get next batch
            image_dataset = tf.data.Dataset.from_tensor_slices(images).batch(self.batch_size)
            iterator_data = image_dataset.make_initializable_iterator()
            image_iterator = iterator_data.get_next()

            # Initialize values
            session.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('../models/autoencoder_model/autoencoder_model.meta')
            saver.restore(session, tf.train.latest_checkpoint('../models/autoencoder_model'))

            for epoch in range(1, epoch_amount + 1):
                epoch_loss = 0

                session.run(iterator_data.initializer)

                for iteration in range(1, int(images.shape[0] / self.batch_size + 1)):
                    epoch_x = session.run(image_iterator)
                    _, c = session.run([optimizer, loss], feed_dict={image_placeholder: epoch_x})
                    epoch_loss += c
                    print('Current run loss:', c)
                    print('Current epoch loss:', epoch_loss)
                    print('Epoch:', epoch, 'iteration:', iteration)
                print('Epoch', epoch, 'completed out of', epoch_amount, 'loss:', epoch_loss)

                saver.save(session, '../models/full_model/full_model')

    def load_full_training(self):
        # Saver to save model
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver.restore(session, '../models/full_model/full_model')

    def get_k_means_loss(self, cluster_centers, soft_assignments, encoder, soft_loss=False):
        encoder_size = numpy.prod(encoder.get_shape().as_list()[1:])
        z = tensorflow.reshape(encoder, [self.batch_size, 1, encoder_size])
        z = tensorflow.tile(z, [1, self.num_clusters, 1])
        u = tensorflow.reshape(cluster_centers, [1, self.num_clusters, encoder_size])
        u = tensorflow.tile(u, [self.batch_size, 1, 1])
        distances = tensorflow.reshape(tensorflow.norm(z - u, axis=2), [self.batch_size, self.num_clusters])
        if soft_loss:
            weighted_distances = distances * soft_assignments
            loss = tensorflow.reduce_mean(tensorflow.reduce_sum(weighted_distances, axis=1))
        else:
            loss = tensorflow.reduce_mean(tensorflow.reduce_min(distances, axis=1))
        return loss

    def test_kmeans(self, training_data, training_labels, test_data, test_labels):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Extract encoder and decoder
            saver = tf.train.import_meta_graph('../models/full_model/full_model.meta')
            saver.restore(session, tf.train.latest_checkpoint('../models/full_model'))
            graph = tf.get_default_graph()
            image_placeholder = graph.get_tensor_by_name("image_input:0")
            encoder = graph.get_tensor_by_name("cnn_encoder/Conv2D:0")

            flat_encoding = tf.contrib.layers.flatten(encoder)

            # K-Means Parameters
            filename = '../models/k_means_model/k_means_model.sav'
            with open(filename, "rb") as input_file:
                kmeans = pickle.load(input_file)

            idx = numpy.array([])
            # Training
            for i in range(0, len(training_data), self.batch_size):
                batch = session.run(flat_encoding, feed_dict={image_placeholder: training_data[i:i + self.batch_size]})
                kmeans.partial_fit(batch)
                id_cluster = kmeans.predict(batch)
                idx = numpy.concatenate((idx, id_cluster), axis=0)

            # Assign a label to each centroid
            # Count total number of labels per centroid, using the label of each training
            # sample to their closest centroid (given by 'idx')
            counts = numpy.zeros(shape=(self.num_clusters, 10))
            for i in range(len(idx)):
                counts[int(idx[i])] += training_labels[i]
            # Assign the most frequent label to the centroid
            labels_map = [numpy.argmax(c) for c in counts]
            labels_map = numpy.array(labels_map)
            # labels_map = tf.convert_to_tensor(labels_map)

            test_predictions = numpy.array([])
            for i in range(0, len(test_data), self.batch_size):
                batch = session.run(flat_encoding, feed_dict={image_placeholder: test_data[i:i + self.batch_size]})
                kmeans.partial_fit(batch)
                id_cluster = kmeans.predict(batch)
                test_predictions = numpy.concatenate((test_predictions, id_cluster), axis=0)

            # Evaluation ops
            # Labels (for assigning a label to a centroid and testing)
            y = tf.placeholder(tf.float32, shape=[None, 10])
            # Lookup: centroid_id -> label
            # predictions_test = numpy.array(test_predictions)
            cluster_label = labels_map[test_predictions.astype(int)]
            # Compute accuracy
            correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(y, 1), tf.int64))
            accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # Test Model
            print("Test Accuracy:", session.run(accuracy_op, feed_dict={y: test_labels}))

    def train_kmeans(self, images):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Extract encoder and decoder
            saver = tf.train.import_meta_graph('../models/full_model/full_model.meta')
            saver.restore(session, tf.train.latest_checkpoint('../models/full_model'))
            graph = tf.get_default_graph()
            image_placeholder = graph.get_tensor_by_name("image_input:0")
            encoder = graph.get_tensor_by_name("cnn_encoder/Conv2D:0")

            flat_encoding = tf.contrib.layers.flatten(encoder)

            kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, batch_size=self.batch_size)

            # Training
            for i in range(0, len(images), self.batch_size):
                batch = session.run(flat_encoding, feed_dict={image_placeholder: images[i:i + self.batch_size]})
                kmeans.partial_fit(batch)

            filename = '../models/k_means_model/k_means_model.sav'
            with open(filename, "wb") as output_file:
                pickle.dump(kmeans, output_file)

    def __get_initial_clusters(self, images):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Extract encoder and decoder
            saver = tf.train.import_meta_graph('../models/autoencoder_model/autoencoder_model.meta')
            saver.restore(session, tf.train.latest_checkpoint('../models/autoencoder_model'))
            graph = tf.get_default_graph()
            image_placeholder = graph.get_tensor_by_name("image_input:0")
            encoder = graph.get_tensor_by_name("cnn_encoder/Conv2D:0")

            flat_encoding = tf.contrib.layers.flatten(encoder)

            kmeans = MiniBatchKMeans(n_clusters=self.num_clusters, batch_size=self.batch_size)

            # Training
            for i in range(0, len(images), self.batch_size):
                batch = session.run(flat_encoding, feed_dict={image_placeholder: images[i:i + self.batch_size]})
                kmeans.partial_fit(batch)
            return kmeans.cluster_centers_

    def __encode_and_flatten(self, images):
        with tf.Session() as session:
            # Extract encoder and decoder
            saver = tf.train.import_meta_graph('../models/full_model/full_model.meta')
            saver.restore(session, tf.train.latest_checkpoint('../models/full_model'))
            graph = tf.get_default_graph()
            image_placeholder = graph.get_tensor_by_name("image_input:0")
            encoder = graph.get_tensor_by_name("cnn_encoder/Conv2D:0")

            # Flatten and encode images
            flat_encoding = tf.contrib.layers.flatten(encoder)
            flat_images = session.run(flat_encoding, feed_dict={image_placeholder: images})
        return flat_images
