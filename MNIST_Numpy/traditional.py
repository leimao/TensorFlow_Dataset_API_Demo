'''
TensorFlow Dataset API Example

Feedable Iterator

Lei Mao

Department of Computer Science
University of Chicago

dukeleimao@gmail.com
'''

import tensorflow as tf
import numpy as np
import time


class Preprocessor(object):

    def __init__(self, num_classes):

        self.num_classes = num_classes

    def preprocess(self, images, labels):
        '''
        Data preprocess procedures
        images: tensor format, dtype = tf.uint8
        labels: tensor format, dtype = tf.uint8
        '''
        # Change dtype from uint to float
        images = images.astype(np.float32)
        # Scale images
        images = images / 255

        return images, labels

def conv_net(x, num_classes, dropout, reuse, is_training):

    with tf.variable_scope('ConvNet', reuse = reuse):

        # Tensor input become 4-D: [batch Size, height, width, channel]
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(inputs = x, filters = 32, kernel_size = 5, activation = tf.nn.relu)
        # Max pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = 2, strides = 2)

        # Convolution layer with 32 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(inputs = conv1, filters = 64, kernel_size = 3, activation=tf.nn.relu)
        # Max pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = 2, strides = 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.layers.flatten(inputs = conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(inputs = fc1, units = 1024)
        # Apply dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(inputs = fc1, rate = dropout, training = is_training)

        # Output layer, class prediction
        output = tf.layers.dense(inputs = fc1, units = num_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        output = tf.nn.softmax(output) if not is_training else output

    return output

class CNN(object):

    def __init__(self, image_shapes, num_classes = 10, batch_size = 16, dropout = 0.5, learning_rate = 0.001):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate

        self.input_shape = [None, image_shapes[0], image_shapes[1]]
        self.images = tf.placeholder(tf.float32, shape = self.input_shape)
        self.labels = tf.placeholder(tf.uint8, shape = [None])
        self.labels_onehot = tf.one_hot(indices = self.labels, depth = self.num_classes)
        self.labels_onehot = tf.cast(self.labels_onehot, tf.float32)

        self.model_initializer()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def model_initializer(self):

        self.outputs_train = conv_net(x = self.images, num_classes = self.num_classes, dropout = self.dropout, reuse = False, is_training = True)
        self.outputs_test = conv_net(x = self.images, num_classes = self.num_classes, dropout = 0, reuse = True, is_training = False)

        correct_pred = tf.equal(tf.argmax(self.outputs_test, 1), tf.argmax(self.labels_onehot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def optimizer_initializer(self):

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.outputs_train, labels = self.labels_onehot))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

    def train(self, train_dataset, num_iterations):

        train_images, train_labels = train_dataset

        train_accuracies = []
        for i in range(num_iterations):
            images = train_images[i*self.batch_size:(i+1)*self.batch_size]
            labels = train_labels[i*self.batch_size:(i+1)*self.batch_size]
            _, train_accuracy = self.sess.run([self.optimizer, self.accuracy], feed_dict = {self.images: images, self.labels: labels})
            train_accuracies.append(train_accuracy)

        train_accuracy_mean = np.mean(train_accuracies)
        
        return train_accuracy_mean

    def test(self, test_dataset, num_iterations):

        test_images, test_labels = test_dataset

        test_accuracies = []
        for i in range(num_iterations):
            images = test_images[i*self.batch_size:(i+1)*self.batch_size]
            labels = test_labels[i*self.batch_size:(i+1)*self.batch_size]
            test_accuracy = self.sess.run(self.accuracy, feed_dict = {self.images: images, self.labels: labels})
            test_accuracies.append(test_accuracy)

        test_accuracy_mean = np.mean(test_accuracies)
        
        return test_accuracy_mean

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        
        return os.path.join(directory, filename)

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


def dataset_preprocessing(images, labels, preprocess, shuffle = False):

    if shuffle:
        shuffled_idx = np.arange(len(images))
        np.random.shuffle(np.arange(len(images)))

        images = images[shuffled_idx]
        labels = labels[shuffled_idx]

    dataset = preprocess(images, labels)

    return dataset


def main():

    batch_size = 16
    num_classes = 10
    dropout = 0.5
    random_seed = 0

    learning_rate = 0.001
    num_epochs = 20
    num_iterations = 20

    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    preprocessor = Preprocessor(num_classes = num_classes)

    model = CNN(image_shapes = train_images.shape[1:], num_classes = num_classes, batch_size = batch_size, dropout = dropout, learning_rate = learning_rate)

    for i in range(num_epochs):

        train_dataset = dataset_preprocessing(images = train_images, labels = train_labels, preprocess = preprocessor.preprocess, shuffle = True)
        test_dataset = dataset_preprocessing(images = test_images, labels = test_labels, preprocess = preprocessor.preprocess, shuffle = False)

        train_accuracy_mean = model.train(train_dataset = train_dataset, num_iterations = num_iterations)
        test_accuracy_mean = model.test(test_dataset = test_dataset, num_iterations = num_iterations)
        print('Epoch: {:03d} | Train Accuracy: {:.2f} | Test Accuracy: {:.2f}'.format(i, train_accuracy_mean, test_accuracy_mean))

if __name__ == '__main__':

    time_start = time.time()

    main()
    
    time_end = time.time()
    time_elapsed = time_end - time_start

    print("Time Elapsed: %02d:%02d:%02d" % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))








