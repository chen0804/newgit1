from __future__ import division, print_function, unicode_literals
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
import time
time_start=time.time()

ops.reset_default_graph()
sess = tf.Session()



#x = np.load('E:/py/DEAP_capsnet/x_data_10_63.npy')
#x = x[0:24000]
#归一化

x = np.load('E:/py/DEAP_capsnet/x_data_36.npy')
y = np.load('E:/py/DEAP_capsnet/label_bin_arousal.npy')
x_data = np.zeros((24000,18,18))
x_data = np.zeros((24000,18,18))
x_data[:,0:9,0:9] = x[:,0:9,0:9]
x_data[:,0:9,9:18] = x[:,0:9,9:18]
x_data[:,9:18,0:9] = x[:,0:9,18:27]
x_data[:,9:18,9:18] = x[:,9:18,0:9]

x = x_data
x = x[21600:24000, :, :]
print(x.shape)
# for i in range(12000):
#   x[i, :, :] = (x[i,:,:] - np.mean(x[i,:,:]))/np.std(x[i,:,:]):

# y = np.load('E:/py/DEAP_capsnet/label_y60.npy')

y = y[21600:24000]


#train_xdata, test_xdata, train_labels, test_labels = train_test_split(x, y, test_size=0.1, random_state=0)

#train_xdata = x[0:605]
train_xdata = x[360:960:,:]
test_xdata = x[960:1020,:,:]
train_labels = y[360:960]
test_labels = y[960:1020]






X = tf.placeholder(shape=[None, 18, 18, 1], dtype=tf.float32, name="X")
caps1_n_maps = 4
caps1_n_caps = caps1_n_maps * 7* 7 # 1152 primary capsules
caps1_n_dims = 8
conv1_params = {
    "filters": 32,
    "kernel_size": 3,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}

conv2_params = {
    "filters":  32, #caps1_n_maps * caps1_n_dims, # 256 convolutional filters
    "kernel_size": 3,
    "strides": 2,
    "padding": "valid",
    "activation": tf.nn.relu
}
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims],
                       name="caps1_raw")
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector
caps1_output = squash(caps1_raw, name="caps1_output")
caps2_n_caps = 10
caps2_n_dims = 16
init_sigma = 0.1

W_init = tf.random_normal(
    shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
    stddev=init_sigma, dtype=tf.float32, name="W_init")
W = tf.Variable(W_init, name="W")
batch_size = tf.shape(X)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                       name="caps1_output_expanded")
caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                   name="caps1_output_tile")
caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                             name="caps1_output_tiled")
W_tiled
caps1_output_tiled
caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                            name="caps2_predicted")
caps2_predicted
raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                       dtype=np.float32, name="raw_weights")
routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                   name="weighted_predictions")
weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                             name="weighted_sum")
caps2_output_round_1 = squash(weighted_sum, axis=-2,
                              name="caps2_output_round_1")
caps2_output_round_1
caps2_predicted
caps2_output_round_1
caps2_output_round_1_tiled = tf.tile(
    caps2_output_round_1, [1, caps1_n_caps, 1, 1, 1],
    name="caps2_output_round_1_tiled")
agreement = tf.matmul(caps2_predicted, caps2_output_round_1_tiled,
                      transpose_a=True, name="agreement")
raw_weights_round_2 = tf.add(raw_weights, agreement,
                             name="raw_weights_round_2")
routing_weights_round_2 = tf.nn.softmax(raw_weights_round_2,
                                        dim=2,
                                        name="routing_weights_round_2")
weighted_predictions_round_2 = tf.multiply(routing_weights_round_2,
                                           caps2_predicted,
                                           name="weighted_predictions_round_2")
weighted_sum_round_2 = tf.reduce_sum(weighted_predictions_round_2,
                                     axis=1, keep_dims=True,
                                     name="weighted_sum_round_2")
caps2_output_round_2 = squash(weighted_sum_round_2,
                              axis=-2,
                              name="caps2_output_round_2")
caps2_output = caps2_output_round_2


def condition(input, counter):
    return tf.less(counter, 100)


def loop_body(input, counter):
    output = tf.add(input, tf.square(counter))
    return output, tf.add(counter, 1)


with tf.name_scope("compute_sum_of_squares"):
    counter = tf.constant(1)
    sum_of_squares = tf.constant(0)

    result = tf.while_loop(condition, loop_body, [sum_of_squares, counter])

with tf.Session() as sess:
    print(sess.run(result))
sum([i**2 for i in range(1, 100 + 1)])
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)
y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
y_proba_argmax
y_pred = tf.squeeze(y_proba_argmax, axis=[1,2], name="y_pred")
y_pred
Y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
T = tf.one_hot(Y, depth=caps2_n_caps, name="T")
with tf.Session():
    print(T.eval(feed_dict={Y: np.array([0, 1])}))
caps2_output
caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                              name="caps2_output_norm")
present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                              name="present_error_raw")
present_error = tf.reshape(present_error_raw, shape=(-1, 10),
                           name="present_error")
absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                             name="absent_error_raw")
absent_error = tf.reshape(absent_error_raw, shape=(-1, 10),
                          name="absent_error")
L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
           name="L")
reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss") + reg
correct = tf.equal(Y, y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(margin_loss, name="training_op")





init = tf.global_variables_initializer()


n_epochs = 1
batch_size = 15

n_iterations_per_epoch = 400

eval_every = 2
loss_tests = []
acc_tests = []
train_loss = []
train_acc =[]


with tf.Session() as sess:
    sess.run(init)
    for iteration in range(0, n_iterations_per_epoch):
        rand_index = np.random.choice(len(train_xdata), size=batch_size)
        rand_x = train_xdata[rand_index]
        rand_x = np.expand_dims(rand_x, 3)      # 加了一个通道的维度
        rand_x = rand_x.reshape([-1, 18, 18, 1])
        rand_y = train_labels[rand_index]
        train_dict = {X: rand_x, Y: rand_y}

        sess.run(training_op, feed_dict=train_dict)
        temp_train_loss = sess.run(margin_loss, feed_dict=train_dict)
        temp_train_acc = sess.run(accuracy, feed_dict=train_dict)

        if (iteration + 1) % eval_every == 0:
            eval_index = np.random.choice(len(test_xdata), size=15)
            eval_x = test_xdata[eval_index]
            eval_x = np.expand_dims(eval_x, 3)
            eval_y = test_labels[eval_index]
            test_dict = {X: eval_x, Y: eval_y}
            loss_test, acc_test = sess.run(
            [margin_loss, accuracy],
            feed_dict={X: eval_x.reshape([-1, 18, 18, 1]),
                           Y: eval_y})
            loss_tests.append(loss_test)

            # Record and print results
            train_loss.append(temp_train_loss)
            train_acc.append(temp_train_acc)
            acc_tests.append(acc_test)
            acc_and_loss = [(iteration + 1), temp_train_loss, temp_train_acc, acc_test]
            acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
            print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

    print("训练集精度矩阵")
    print(train_acc[-11:-1])
    print("平均训练精度"+str(np.mean(train_acc[-11:-1])))
    print("测试集精度矩阵")
    print(acc_tests[-11:-1])
    print("平均测试精度"+str(np.mean(acc_tests[-11:-1])))


    # Matlotlib code to plot the loss and accuracies
    eval_indices = range(0, n_iterations_per_epoch ,eval_every )
    # Plot loss over time
    plt.plot(eval_indices, train_loss, 'k-')
    plt.title('Loss per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.show()



    # Plot train and test accuracy
    plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
    plt.plot(eval_indices, acc_tests, 'r--', label='Test Set Accuracy')
    plt.title('Train and Test Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    time_end = time.time()
    print('totally cost', time_end - time_start)