import cv2
import pickle
import numpy
import tensorflow as tf
from cifar10_input import load_cifar10
from neural_net import NeuralNetwork


class UnsupervisedTracker:
    def __init__(self):
        self.neural_network = NeuralNetwork()
        self.roi_cluster_id = None

    def video_to_frames(self, video_location):
        images = []
        video_capture = cv2.VideoCapture(video_location)
        success = True
        while success:
            success, image = video_capture.read()
            if not success:
                return images
            else:
                images.append(image)

    def frames_to_video(self, frames, output_location):
        # Determine width and height of video
        frame = frames[0]
        height, width, channels = frame.shape

        # Define the codec and create VideoWriter object
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
        output = cv2.VideoWriter(output_location, four_cc, 20.0, (width, height))

        for frame in frames:
            output.write(frame)

        output.release()

    def get_region_of_interest(self, image, height=32, width=32):
        region_of_interest = cv2.selectROI("Region of interest", image, fromCenter=False)

        region_image = image[int(region_of_interest[1]):int(region_of_interest[1] + region_of_interest[3]),
                             int(region_of_interest[0]):int(region_of_interest[0] + region_of_interest[2])]
        region_image = cv2.resize(region_image, (width, height))
        cv2.destroyAllWindows()

        return region_image

    def track_object_in_video(self, video_location, training=False):
        if training:
            training_data, training_labels, test_data, test_labels = load_cifar10()
            self.neural_network.train_autoencoder(training_data, test_data)
            self.neural_network.train_k_means_and_autoencoder(training_data)
            self.neural_network.train_kmeans(training_data)
            self.neural_network.test_kmeans(training_data, training_labels, test_data, test_labels)

        if video_location is not None:
            frames = self.video_to_frames(video_location)
            region_of_interest = self.get_region_of_interest(frames[0])

            self.roi_cluster_id = self.__get_cluster_id(region_of_interest)

            self.__paint_bounding_boxes(frames)
            self.frames_to_video(frames, "../data/output_video/output.mp4")

    def __paint_bounding_boxes(self, frames):
        window_size = [80, 60]
        step_size = [8, 6]
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Restore encoder
            saver = tf.train.import_meta_graph('../models/full_model/full_model.meta')
            saver.restore(session, tf.train.latest_checkpoint('../models/full_model'))
            graph = tf.get_default_graph()
            image_placeholder = graph.get_tensor_by_name("image_input:0")
            encoder = graph.get_tensor_by_name("cnn_encoder/Conv2D:0")
            flat_encoder = tf.contrib.layers.flatten(encoder)

            # K-Means Parameters
            filename = '../models/k_means_model/k_means_model.sav'
            with open(filename, "rb") as input_file:
                kmeans = pickle.load(input_file)

            for i, frame in enumerate(frames):
                print("frame:", i)
                bounding_box_dim = None
                for y in range(0, frame.shape[0], step_size[1]):
                    for x in range(0, frame.shape[1], step_size[0]):
                        current_window = frame[y:y + window_size[1], x:x + window_size[0]]
                        resized_window = cv2.resize(current_window, (32, 32))
                        batch_window = numpy.resize(resized_window, [1, 32, 32, 3])

                        flat_encoded_image = session.run(flat_encoder, feed_dict={image_placeholder: batch_window})
                        estimation = kmeans.predict(flat_encoded_image)

                        if estimation[0] == self.roi_cluster_id:
                            if bounding_box_dim is None:
                                bounding_box_dim = [x, y, x + window_size[0], y + window_size[1]]
                            else:
                                bounding_box_dim = [min(bounding_box_dim[0], x),
                                                    min(bounding_box_dim[1], y),
                                                    max(bounding_box_dim[2], x + window_size[0]),
                                                    max(bounding_box_dim[3], y + window_size[1])]
                if bounding_box_dim is not None:
                    cv2.rectangle(frame,
                                  (bounding_box_dim[0], bounding_box_dim[1]),
                                  (bounding_box_dim[2], bounding_box_dim[3]),
                                  (0, 255, 0),
                                  2)

    def __get_cluster_id(self, image):
        image = numpy.resize(image, [1, 32, 32, 3])
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

            flat_enc_image = session.run(flat_encoding, feed_dict={image_placeholder: image})
            prediction = kmeans.predict(flat_enc_image)

            return prediction[0]

    def __get_centroids(self):
        # K-Means Parameters
        filename = '../models/k_means_model/k_means_model.sav'
        with open(filename, "rb") as input_file:
            kmeans = pickle.load(input_file)
        return kmeans.cluster_centers_

    def get_cluster_estimation(self, encoded_image, centroids):
        z_expanded = tf.reshape(encoded_image, [1, 1] + encoded_image.get_shape().as_list()[1:])
        z_expanded = tf.tile(z_expanded, (1, self.neural_network.num_clusters, 1))
        u_expanded = tf.reshape(centroids, [1] + list(centroids.shape))
        u_expanded = tf.tile(u_expanded, (1, 1, 1))

        distances_from_cluster_centers = tf.norm(z_expanded - u_expanded, 2, axis=2)
        qij_numerator = 1 + distances_from_cluster_centers * distances_from_cluster_centers
        qij_numerator = 1 / qij_numerator
        normalizer_q = tf.reshape(tf.reduce_sum(qij_numerator, axis=1), [1, 1])
        return qij_numerator / normalizer_q
