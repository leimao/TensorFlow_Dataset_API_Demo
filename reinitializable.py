'''
TensorFlow Dataset API Example

Reinitializable Iterator

Lei Mao

Department of Computer Science
University of Chicago

dukeleimao@gmail.com
'''

import tensorflow as tf
import numpy as np


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
        images = tf.cast(images, tf.float32)
        # Scale images
        images = images / 255
        # One-hot encoding
        labels = tf.one_hot(indices = labels, depth = self.num_classes)
        # Change dtype from uint to float
        labels = tf.cast(labels, tf.float32)

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

    def __init__(self, dataset_output_types, dataset_output_shapes, num_classes = 10, batch_size = 16, dropout = 0.5, learning_rate = 0.001):

        self.num_classes = 10
        self.batch_size = 16
        self.dropout = 0.5
        self.learning_rate = 0.001


        self.iterator = tf.data.Iterator.from_structure(output_types = dataset_output_types, output_shapes = dataset_output_shapes)
        self.images, self.labels = self.iterator.get_next()

        self.model_initializer()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def model_initializer(self):

        self.outputs_train = conv_net(x = self.images, num_classes = self.num_classes, dropout = self.dropout, reuse = False, is_training = True)
        self.outputs_test = conv_net(x = self.images, num_classes = self.num_classes, dropout = 0, reuse = True, is_training = False)

        correct_pred = tf.equal(tf.argmax(self.outputs_test, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def optimizer_initializer(self):

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.outputs_train, labels = self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

    def train(self, train_dataset, num_iterations):

        train_init_operator = self.iterator.make_initializer(train_dataset)
        self.sess.run(train_init_operator)

        train_accuracies = []
        for i in range(num_iterations):
            _, train_accuracy = self.sess.run([self.optimizer, self.accuracy])
            train_accuracies.append(train_accuracy)

        train_accuracy_mean = np.mean(train_accuracies)
        
        return train_accuracy_mean

    def test(self, test_dataset, num_iterations):

        test_init_operator = self.iterator.make_initializer(test_dataset)
        self.sess.run(test_init_operator)

        test_accuracies = []
        for i in range(num_iterations):
            test_accuracy = self.sess.run(self.accuracy)
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


def dataset_generation(images, labels, preprocess, batch_size = 16, repeat = False, shuffle = False):
    '''
    Generate tensorflow dataset object
    images: numpy array format
    labels: numpy array format
    '''
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess)
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size = 100000)
    dataset = dataset.batch(batch_size)

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

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    preprocessor = Preprocessor(num_classes = num_classes)

    train_dataset = dataset_generation(images = train_images, labels = train_labels, preprocess = preprocessor.preprocess, batch_size = 16, repeat = True, shuffle = True)
    test_dataset = dataset_generation(images = test_images, labels = test_labels, preprocess = preprocessor.preprocess, batch_size = 16, repeat = False, shuffle = False)

    model = CNN(dataset_output_types = train_dataset.output_types, dataset_output_shapes = train_dataset.output_shapes, num_classes = num_classes, batch_size = batch_size, dropout = dropout, learning_rate = learning_rate)

    for i in range(num_epochs):

        train_accuracy_mean = model.train(train_dataset = train_dataset, num_iterations = num_iterations)
        test_accuracy_mean = model.test(test_dataset = test_dataset, num_iterations = num_iterations)
        print('Epoch: {:03d} | Train Accuracy: {:.2f} | Test Accuracy: {:.2f}'.format(i, train_accuracy_mean, test_accuracy_mean))

if __name__ == '__main__':
    
    main()







