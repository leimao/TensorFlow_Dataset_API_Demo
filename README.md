# TensorFlow Dataset API Demo

Lei Mao

University of Chicago

## Introduction

I have been using TensorFlow since its first release (version 0.1) in 2015. So some of my coding habits for the earlier versions have been kept until now. One of the habits is that I always use ``placeholder`` and ``feed_dict`` to feed data, including training data, validation data, and test data, into TensorFlow graph for computation. ``placeholder`` and ``feed_dict`` make your program super flexible. When you have newly collected data, feeding data using ``feed_dict`` will never damage the pre-built TensorFlow graph. When you want to change the dropout rate to 0 during test time, or adjust learning rate during training time, ``feed_dict`` is always the best choice to make your program flexible.

Because the datasets of some of my research projects become larger, I started to be concerned about the data loading and preprocessing efficiencies, as it may become the "bottleneck" of training process if not being handled properly. For large datasets, it is not feasible to read all the data into memory and load the data to the program from memory (You may do it on your supercomputer, but others will probably not be able to run your program on their own desktops). So what people usually do is to read data from disk and do all the data preprocessing on the fly. So to make this portion of code looking neat, I have written ``Dataset`` and ``Iterator`` classes for some of my projects. However, they are not generalizable, meaning that if you want to use the ``Dataset`` and ``Iterator`` from one project for another project, it usually has to be modified significantly. In addition, my ``Dataset`` and ``Iterator`` was not written in multi-thread fashion, making me concerned about the loading and preprocessing efficiencies.

As TensorFlow updates, I started to be aware that there are official ``Dataset`` and ``Iterator`` classes in TensorFlow, which allows users to make use of their internal optimization for loading and preprocessing data. According to the [TensorFlow](https://www.tensorflow.org/performance/datasets_performance) official documentation, using their ``Iterator`` should be asymptotically faster than an ordinary single-thread ``Iterator``. However, in my preliminary tests, I found the TensorFlow ``Iterator`` was significanly slower than a manual single-thread ``Iterator`` in some cases, probably due to its heavy overhead running time.


## TensorFlow Dataset API Usages

Frankly, I think it is not easy to learn the TensorFlow Dataset APIs, because there are many different ways to use the APIs and those ways have slightly different effects, which means that they have to be used for different purposes. The official [guide](https://www.tensorflow.org/guide/datasets) provided some toy examples of how to use TensorFlow Dataset APIs in different ways. But it is really hard for users to understand why they have to code in that way for each step, not even mention how to choose appropriate ways to code for different purposes.


For most of the ways of using TensorFlow Dataset APIs, they will work well for most of your research projects because usually the dataset of your research project is fixed. You can create ``Dataset`` and ``Iterator`` instances for your fixed training, validation, and test dataset independently. Your TensorFlow program does not have to be written in a fashion that allows new data neither. However, in practice, we always trained our model for testing new data, and our TensorFlow program has to allow new data streams if it is going to be a real application. In this case, we need to carefully design our program to allow the new data stream using TensorFlow official ``Iterator``.



The toy dataset on TensorFlow official [guide](https://www.tensorflow.org/guide/datasets) for TensorFlow Dataset API usages is trivial. Here I will use MNIST dataset as a concrete example. TensorFlow ``Dataset`` and ``Iterator`` instances are the two compulsary components of the API.


### Dataset Instance

```python
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

def dataset_generation(images, labels, preprocess, buffer_size = 100000, batch_size = 16, repeat = False, shuffle = False):
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
        dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.batch(batch_size)

    return dataset

# Create TensorFlow dataset preprocessing unit
preprocessor = Preprocessor(num_classes = num_classes)
# Create TensorFlow Dataset instance
train_dataset = dataset_generation(images = train_images, labels = train_labels, preprocess = preprocessor.preprocess, batch_size = 16, repeat = True, shuffle = True)
test_dataset = dataset_generation(images = test_images, labels = test_labels, preprocess = preprocessor.preprocess, batch_size = 16, repeat = False, shuffle = False)
```

To generate TensorFlow dataset instance, usually four things have to be set.

**Iterable Numpy Array Format Dataset**

For small dataset, it could be just the numpy array of the whole dataset. For large dataset, it could be the filenames or filepath of the data stored in the hard drive.

**Data Preprocess**

The dataset preprocessing was done using ``map``. This is where the preprocessing and loading efficiencies happen, since the ``map`` function allows the procedures to run in parallel.

**Shuffle**

Usually we could just shuffle the dataset beforehand without using the built-in shuffle function. If you use shuffle for dataset, it will shuffle the dataset every time you start a new ``Iterator`` instance and it is usually slow in practice.

**Batch Size**

Designate the batch size for your dataset.


### Iterator Instance

According to the TensorFlow official [guide](https://www.tensorflow.org/guide/datasets), there are four types of ``Iterator``: one-shot, initializable, reinitializable, and feedable. Personally I think reinitializable and feedable ``Iterator``s are the most useful in practice. I also integrate the ``Iterator`` instances into training class because I usually prefer to write TensorFlow code in a object-oriented fashion.

A typical object-oriented TensorFlow code with TensorFlow official ``Iterator`` reinitializable instance looks like this.

```python
class CNN(object):

    def __init__(self, dataset_output_types, dataset_output_shapes, num_classes = 10, batch_size = 16, dropout = 0.5, learning_rate = 0.001):

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate

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

```

We use ``tf.data.Iterator`` to create ``Iterator`` instance, and ``iterator.get_next()`` as the inputs for the TensorFlow graph. Each time we want to switch dataset, we have to reinitialize the ``Iterator`` using the following way.

```python
train_init_operator = self.iterator.make_initializer(train_dataset)
self.sess.run(train_init_operator)
```

Therefore, even if there are new data stream coming, we just have to create a ``Dataset`` instance outside the main TensorFlow graph, and pass the ``Dataset`` instance into the main TensorFlow graph for test.

## TensorFlow Dataset API Drawbacks

I tested training MNIST digit classifier on a NVIDIA TitanX GPU using Numpy format MNIST dataset with manual single-thread data loadinig and preprocessing, TensorFlow reinitializable ``Iterator``, and TensorFlow feedable ``Iterator``. I found TensorFlow reinitializable ``Iterator`` and TensorFlow feedable ``Iterator`` are comparable, but they are not significantly faster than manual single-thread data loadinig and preprocessing. 


### Manual Instance
```
Epoch: 000 | Train Accuracy: 0.51 | Test Accuracy: 0.70
Epoch: 001 | Train Accuracy: 0.87 | Test Accuracy: 0.82
Epoch: 002 | Train Accuracy: 0.91 | Test Accuracy: 0.88
Epoch: 003 | Train Accuracy: 0.94 | Test Accuracy: 0.89
Epoch: 004 | Train Accuracy: 0.97 | Test Accuracy: 0.91
Epoch: 005 | Train Accuracy: 0.99 | Test Accuracy: 0.91
Epoch: 006 | Train Accuracy: 1.00 | Test Accuracy: 0.90
Epoch: 007 | Train Accuracy: 1.00 | Test Accuracy: 0.90
Epoch: 008 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 009 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 010 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 011 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 012 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 013 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 014 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 015 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 016 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 017 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 018 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 019 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Time Elapsed: 00:00:05
```

### TensorFlow Reinitializable ``Iterator`` Instance

```
Epoch: 000 | Train Accuracy: 0.49 | Test Accuracy: 0.62
Epoch: 001 | Train Accuracy: 0.88 | Test Accuracy: 0.83
Epoch: 002 | Train Accuracy: 0.94 | Test Accuracy: 0.86
Epoch: 003 | Train Accuracy: 0.95 | Test Accuracy: 0.93
Epoch: 004 | Train Accuracy: 0.99 | Test Accuracy: 0.89
Epoch: 005 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 006 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 007 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 008 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 009 | Train Accuracy: 1.00 | Test Accuracy: 0.90
Epoch: 010 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 011 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 012 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 013 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 014 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 015 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 016 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 017 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 018 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 019 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Time Elapsed: 00:00:05
```

### TensorFlow Feedable ``Iterator`` Instance

```
Epoch: 000 | Train Accuracy: 0.49 | Test Accuracy: 0.70
Epoch: 001 | Train Accuracy: 0.87 | Test Accuracy: 0.80
Epoch: 002 | Train Accuracy: 0.92 | Test Accuracy: 0.85
Epoch: 003 | Train Accuracy: 0.95 | Test Accuracy: 0.90
Epoch: 004 | Train Accuracy: 0.97 | Test Accuracy: 0.89
Epoch: 005 | Train Accuracy: 0.99 | Test Accuracy: 0.89
Epoch: 006 | Train Accuracy: 0.99 | Test Accuracy: 0.91
Epoch: 007 | Train Accuracy: 1.00 | Test Accuracy: 0.88
Epoch: 008 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 009 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 010 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 011 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 012 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 013 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 014 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 015 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 016 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 017 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 018 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Epoch: 019 | Train Accuracy: 1.00 | Test Accuracy: 0.92
Time Elapsed: 00:00:05
```

I later tried to increase the complexity of preprocessing function but found the there is only very slight improvement.

<br />


Because for some datasets, we would not load the whole dataset into memory as a Numpy array but read batches from hard drive using their filepaths. I tested loading MNIST dataset from hard drive during training. The test result is surprising, the TensorFlow ``Iterator`` intances are actually much slower than single-thread manual data loading.

### Manual Instance

```
Epoch: 000 | Train Accuracy: 0.40 | Test Accuracy: 0.65
Epoch: 001 | Train Accuracy: 0.77 | Test Accuracy: 0.82
Epoch: 002 | Train Accuracy: 0.90 | Test Accuracy: 0.91
Epoch: 003 | Train Accuracy: 0.94 | Test Accuracy: 0.91
Epoch: 004 | Train Accuracy: 0.96 | Test Accuracy: 0.95
Epoch: 005 | Train Accuracy: 1.00 | Test Accuracy: 0.95
Epoch: 006 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 007 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 008 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 009 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 010 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 011 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 012 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 013 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 014 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 015 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 016 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 017 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 018 | Train Accuracy: 1.00 | Test Accuracy: 0.95
Epoch: 019 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Time Elapsed: 00:00:05
```

### TensorFlow Reinitializable ``Iterator`` Instance

```
Epoch: 000 | Train Accuracy: 0.42 | Test Accuracy: 0.68
Epoch: 001 | Train Accuracy: 0.78 | Test Accuracy: 0.79
Epoch: 002 | Train Accuracy: 0.89 | Test Accuracy: 0.90
Epoch: 003 | Train Accuracy: 0.93 | Test Accuracy: 0.89
Epoch: 004 | Train Accuracy: 0.98 | Test Accuracy: 0.94
Epoch: 005 | Train Accuracy: 0.98 | Test Accuracy: 0.92
Epoch: 006 | Train Accuracy: 0.99 | Test Accuracy: 0.94
Epoch: 007 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 008 | Train Accuracy: 0.99 | Test Accuracy: 0.92
Epoch: 009 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 010 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 011 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 012 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 013 | Train Accuracy: 1.00 | Test Accuracy: 0.91
Epoch: 014 | Train Accuracy: 1.00 | Test Accuracy: 0.95
Epoch: 015 | Train Accuracy: 1.00 | Test Accuracy: 0.95
Epoch: 016 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 017 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 018 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 019 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Time Elapsed: 00:00:12
```

### TensorFlow Feedable ``Iterator`` Instance

```
Epoch: 000 | Train Accuracy: 0.38 | Test Accuracy: 0.62
Epoch: 001 | Train Accuracy: 0.78 | Test Accuracy: 0.77
Epoch: 002 | Train Accuracy: 0.88 | Test Accuracy: 0.91
Epoch: 003 | Train Accuracy: 0.92 | Test Accuracy: 0.91
Epoch: 004 | Train Accuracy: 0.97 | Test Accuracy: 0.93
Epoch: 005 | Train Accuracy: 0.99 | Test Accuracy: 0.93
Epoch: 006 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 007 | Train Accuracy: 0.98 | Test Accuracy: 0.93
Epoch: 008 | Train Accuracy: 0.98 | Test Accuracy: 0.92
Epoch: 009 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 010 | Train Accuracy: 0.99 | Test Accuracy: 0.93
Epoch: 011 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 012 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 013 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 014 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 015 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 016 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 017 | Train Accuracy: 1.00 | Test Accuracy: 0.93
Epoch: 018 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Epoch: 019 | Train Accuracy: 1.00 | Test Accuracy: 0.94
Time Elapsed: 00:00:12
```

## Conclusions

Using TensorFlow ``Iterator`` is not necessary faster than the self-implemented ``Iterator`` because of its heavy overhead running time. Based on these test results, I would not favor TensorFlow ``Iterator`` over my self-implemented ``Iterator`` in my daily TensorFlow usages. Maybe I would keep using ``feed_dict`` frequently simply because it looks more natural.


## Final Remarks

TensorFlow is evolving fast. It always tries to keep up with the state-of-art deep learning research, which other deep learning frameworks usually don't. Keep yourself updated on TensorFlow is not easy, and of course there are many caveats when you are using the new features.


## GitHub

All the testing TensorFlow codes have been open sourced on my [GitHub](https://github.com/leimao/TensorFlow_Dataset_API_Demo).


### Numpy Format MNIST Dataset

* [Manual Instance](https://github.com/leimao/TensorFlow_Dataset_API_Demo/blob/master/MNIST_Numpy/traditional.py)
* [TensorFlow Reinitializable ``Iterator`` Instance](https://github.com/leimao/TensorFlow_Dataset_API_Demo/blob/master/MNIST_Numpy/reinitializable.py)
* [TensorFlow Feedable ``Iterator`` Instance](https://github.com/leimao/TensorFlow_Dataset_API_Demo/blob/master/MNIST_Numpy/feedable.py)

### PNG Format MNIST Dataset

* [Manual Instance](https://github.com/leimao/TensorFlow_Dataset_API_Demo/blob/master/MNIST_PNG/traditional.py)
* [TensorFlow Reinitializable ``Iterator`` Instance](https://github.com/leimao/TensorFlow_Dataset_API_Demo/blob/master/MNIST_PNG/reinitializable.py)
* [TensorFlow Feedable ``Iterator`` Instance](https://github.com/leimao/TensorFlow_Dataset_API_Demo/blob/master/MNIST_PNG/feedable.py)