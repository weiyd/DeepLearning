# encoding=utf-8
# TF实现简单的CNN

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# 读取数据
mnist = input_data.read_data_sets("MINIST_data/", one_hot=True)
# 建立会话
sess = tf.InteractiveSession()

# 复用权值初始化函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 复用偏置初始化函数，偏置0.1避免死亡节点
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 复用2D卷积函数
def con2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 复用2x2最大池化函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# 网络输入
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# -1是代表任意，具体根据实际计算 1代表通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积 输入28x28x1 输出14x14x32
w_con1 = weight_variable([5, 5, 1, 32])
b_con1 = bias_variable([32])
h_con1 = tf.nn.relu(con2d(x_image, w_con1) + b_con1)
h_pool1 = max_pool_2x2(h_con1)

# 第二层卷积 输入14x14x32 输出7x7x64
w_con2 = weight_variable([5, 5, 32, 64])
b_con2 = bias_variable([64])
h_con2 = tf.nn.relu(con2d(h_pool1, w_con2) + b_con2)
h_pool2 = max_pool_2x2(h_con2)

# 第三层全连接(带有dropout) 输入7*7*64 输出1024x1
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# 保留节点量
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层全连接 输入1024 输出10
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# 采用Adam对交叉熵优化
cross_entroy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entroy)

# softmax输出最大的对应的Onehot即为预测数字
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 10 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d,training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1}))
