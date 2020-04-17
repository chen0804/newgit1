import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()

iris = datasets.load_iris()
x_vals = iris.data
y = iris.target

labels = tf.expand_dims(y, 1)
indices = tf.expand_dims(tf.range(0, 150, 1 ), 1)
concated =tf.concat([indices, labels],1)
y_vals = tf.sparse_to_dense(concated, tf.stack([150, 3]),1.0, 0.0)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
y_vals = y_vals.eval(session=sess)


sess = tf.Session()
seed = 2
tf.set_random_seed(seed)
np.random.seed(seed)

# Initialize placeholders
x_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)

#Split data into train/test = 80%/20%
train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]


# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_vals_train = np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test = np.nan_to_num(normalize_cols(x_vals_test))

# Declare batch size
batch_size = 50


# Create variables for both NN layers
hidden_layer_nodes = 1
A1 = tf.Variable(tf.random_normal(shape=[4, hidden_layer_nodes]))  # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))  # one biases for each hidden node
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes, 3]))  # hidden inputs -> 1 output
b2 = tf.Variable(tf.random_normal(shape=[3]))  # 1 bias for the output



# Declare model operations
hidden_output = tf.add(tf.matmul(x_data, A1), b1)
final_output = tf.add(tf.matmul(hidden_output, A2), b2)

final_output_1 = tf.nn.softmax(final_output)
prediction = tf.cast(tf.argmax(final_output_1,1),tf.float32)
prediction_correct = tf.cast(tf.equal(tf.argmax(final_output_1,1), tf.argmax(y_target,1)), tf.float32)
accuracy = tf.reduce_mean(prediction_correct)

# Declare loss function (MSE)

loss = tf.reduce_mean(tf.square(y_target -final_output_1))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(loss)



# Initialize variables


init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []
train_acc =[]
test_acc = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = y_vals_train[rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    test_temp_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    test_loss.append(np.sqrt(test_temp_loss))



    temp_acc_train = sess.run(accuracy, feed_dict={x_data:rand_x, y_target:rand_y})
    train_acc.append(temp_acc_train)
    temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
    test_acc.append(temp_acc_test)
    if (i + 1) % 50 == 0:
        print('Generation: ' + str(i + 1) + '. loss = ' + str(temp_loss))
        print('. train_acc = ' + str(temp_acc_train))
        print('. test_acc = ' + str(temp_acc_test))
        #print(test_acc)



# Plot loss (MSE) over time
plt.plot(loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.legend(loc='upper right')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

plt.plot(train_acc, 'k-', label='Train acc')
plt.plot(test_acc, 'r--', label='Test acc')
plt.title('train and test acc')
plt.legend(loc='lower right')
plt.xlabel('Generation')
plt.ylabel('accuracy')
plt.show()
