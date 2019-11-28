import tensorflow as tf
import pylab
import numpy as np
import argparse
from tensorflow.contrib.layers.python.layers import batch_norm

parser = argparse.ArgumentParser()

parser.add_argument('--train_from_begin', action='store_true', #初始化为bool,有就是true
                        help='Whether to clean up the model directory if present.')
args = parser.parse_args()
rebegin = args.train_from_begin
train_epochs = 200

#BATCH_SIZE = 32
is_train = False
BATCH_SIZE = 1
DISPLAY_STEP = 1
STORE_STEP = 20

#数据信息
train_im_num = 59000
val_im_num = 999
test_im_num = 10000

def weight_variable(shape):#得到shape的正态分布随机矩阵
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def relu_conv2d(h ,W, b):
  return tf.nn.relu(tf.nn.conv2d(h, W, strides=[1,1,1,1], padding='SAME')+b)

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def avg_pool_7x7(x):
  return tf.nn.avg_pool(x, ksize=[1,7,7,1], strides=[1,7,7,1], padding='SAME')

def batch_norm_layer(x):
  return batch_norm(x, decay = 0.9, updates_collections=None, is_training = is_train)

def relu_bn_conv2d(h,W,b):
  return tf.nn.relu(batch_norm_layer(tf.nn.conv2d(h,W, strides=[1,1,1,1], padding='SAME')+b))
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None, 28,28, 1])
y = tf.placeholder(tf.float32, [None, 10])


x_image = tf.reshape(x, [-1, 28,28, 1]) #可以直接把npz中的数据拿过来辽
W_conv1_3x3 = weight_variable([3,3,1,64]) 
b_conv1_3x3 = bias_variable([64]) #定义了第一个卷积核参数

h_conv1 = relu_bn_conv2d(x_image, W_conv1_3x3, b_conv1_3x3) #采用一次卷积，得到64维输出
h_conv1 = max_pool_2x2(h_conv1) #步长和ksize一样可避免重复池化[batch, 14, 14, 64]
################   first layer get[-1, 28, 28, 64]

#W_conv21_5x1 = weight_variable([5,1,64,64])
#b_conv21_5x1 = bias_variable([64])
#W_conv22_1x5 = weight_variable([1,5,64,64])
#b_conv22_1x5 = bias_variable([64])
W_conv2_5x5 = weight_variable([5,5,64,64])
b_conv2_5x5 = bias_variable([64])

W_conv2_7x7 = weight_variable([7,7,64,64])
b_conv2_7x7 = bias_variable([64])
#W_conv22_1x7 = weight_variable([1,7,64,64])
#b_conv22_1x7 = bias_variable([64])

W_conv2_3x3 = weight_variable([3,3,64,64])
b_conv2_3x3 = bias_variable([64])
#W_conv22_1x3 = weight_variable([1,3,64,64])
#b_conv22_1x3 = bias_variable([64])

#h_conv2_5x1 = relu_conv2d(h_conv1, W_conv21_5x1, b_conv21_5x1)
#h_conv2_5x5 = relu_conv2d(h_conv2_5x1, W_conv22_1x5, b_conv22_1x5)
h_conv2_5x5 = relu_conv2d(h_conv1, W_conv2_5x5, b_conv2_5x5)
#h_conv2_7x1 = relu_conv2d(h_conv1, W_conv21_7x1, b_conv21_7x1)
#h_conv2_7x7 = relu_conv2d(h_conv2_7x1, W_conv22_1x7, b_conv22_1x7)
h_conv2_7x7 = relu_conv2d(h_conv1, W_conv2_7x7, b_conv2_7x7)
#h_conv2_3x1 = relu_conv2d(h_conv1, W_conv21_3x1, b_conv21_3x1)
#h_conv2_3x3 = relu_conv2d(h_conv2_3x1, W_conv22_1x3, b_conv22_1x3)
h_conv2_3x3 = relu_conv2d(h_conv1, W_conv2_3x3, b_conv2_3x3)
h_conv2 = tf.concat([h_conv2_5x5, h_conv2_7x7, h_conv2_3x3], 3)
h_pool2 = max_pool_2x2(h_conv2)
################    second layer get[-1,28,28,64*3]

W_conv3 = weight_variable([5,5,192,10])
b_conv3 = bias_variable([10])

h_conv3 = relu_conv2d(h_pool2, W_conv3, b_conv3)
nt_hpool = avg_pool_7x7(h_conv3)
nt_hpool_flat = tf.reshape(nt_hpool, [-1,10])

pred = tf.nn.softmax(nt_hpool_flat)
#w = tf.Variable(tf.random_normal([784,10]))
#b = tf.Variable(tf.zeros([10]))
#正向传播


#pre_mat = tf.matmul(x, w)+b  #more layers?
#pred = tf.nn.softmax(pre_mat)

cost_l = tf.nn.softmax_cross_entropy_with_logits(logits = nt_hpool_flat, labels = y) #计算pred与y的交叉熵
cost = tf.reduce_mean(cost_l)
'''global_step = tf.Variable(0, trainable = False)
decay_learning_rate = tf.train.exponential_decay(0.04, global_step, 1000, 0.9) #global_step每100步，学习率变为原来的0.94
train_step = tf.train.AdamOptimizer(decay_learning_rate).minimize(cost, global_step = global_step)
'''
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

learning_rate = 0.0001
optimizer = tf.train.RMSPropOptimizer(learning_rate = learning_rate,
        decay = 0.9, #对于梯度的折扣效果相当于学习率下降
        momentum = 0.9,
        epsilon = 1e-10,
        use_locking=False,
        centered = False,
        name = 'RMSProp'
        ).minimize(cost)


dataX = np.load("./ori_data/X_kannada_MNIST_train.npz")['arr_0']
dataX = dataX.reshape([-1, 28,28,1])
datay = np.load("./ori_data/y_kannada_MNIST_train.npz")['arr_0']
datay = np.eye(10,dtype=float)[datay] #one-hot
train_x = dataX[0:train_im_num]
train_y = datay[0:train_im_num]

val_x = dataX[train_im_num:-1]
val_y = datay[train_im_num:-1]
test_x = np.load("./ori_data/X_kannada_MNIST_test.npz")['arr_0']
test_x = test_x.reshape([-1, 28, 28,1])

dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.shuffle(50).batch(BATCH_SIZE).repeat()
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(BATCH_SIZE).repeat()

test_dataset = tf.data.Dataset.from_tensor_slices(test_x)
test_dataset = test_dataset.batch(BATCH_SIZE).repeat()
test_batch = test_dataset.make_one_shot_iterator()
test_xs = test_batch.get_next()
batch_iterator = dataset.make_initializable_iterator()
batch_xs, batch_ys = batch_iterator.get_next()
val_batch_iterator = val_dataset.make_initializable_iterator()
val_batch_xs, val_batch_ys = val_batch_iterator.get_next()
#input_queue = tf.train.slice_input_producer([input_x, input_y], shuffle=False)#切片 这个方法又慢又不好用
#batch_xs, batch_ys = tf.train.batch(input_queue, batch_size=BATCH_SIZE)

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
#saver
saver = tf.train.Saver()
model_dir = "./model/"


def train_buf():
  with tf.Session(config = tfconfig) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(batch_iterator.initializer)
    tf.train.start_queue_runners(sess=sess)
    for i in range(20000):
      x_, y_ = sess.run([batch_xs, batch_ys])
      optimizer.run(feed_dict = {x:x_, y:y_}, session=sess)
      if i%200 == 0:
        acc = accuracy.eval(feed_dict={x:x_, y:y_}, session=sess)
        print("step: %d, training acc: %g"%(i, acc))

def train():
#启动session
  with tf.Session(config = tfconfig) as sess:
      sess.run(tf.global_variables_initializer())
      #开启多线程
      #coord = tf.train.Coordinator()
      #threads = tf.train.start_queue_runners(sess, coord)
      sess.run(batch_iterator.initializer)
      sess.run(val_batch_iterator.initializer)
      if not rebegin:
        ckpt = tf.train.get_checkpoint_state(model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("restore from ",ckpt.model_checkpoint_path)
      print("into sess")
      for epoch in range(train_epochs):
          avg_cost = 0.
          total_batch = int(train_im_num/BATCH_SIZE)
          for i in range(total_batch):
              #运行优化器
              x_, y_ = sess.run([batch_xs, batch_ys])
              #train_step.run(feed_dict = {x:x_, y:y_}, session = sess)
              _, acc = sess.run([optimizer, accuracy], feed_dict = {x: x_, y: y_})
              if i%200 == 0:
                print("epoch: %d, step: %d, acc; %g"%(epoch, i, acc))
              if i%1000 == 0:
                val_x, val_y = sess.run([val_batch_xs, val_batch_ys])
                acc_v = sess.run(accuracy, feed_dict={x:val_x, y:val_y})
                print("epoch : %d, step: %d, acc: %g"%(epoch, i, acc_v))
                save_path = saver.save(sess, model_dir+str(epoch)+"_"+str(i)+"model.ckpt")
                print("Model saved in file: %s" %save_path)

      print("Finished!")

def evaluate():
  print("start evaluate")
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(val_batch_iterator.initializer)
    ckpt = tf.train.get_checkpoint_state(model_dir)
      
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("into sess")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ",accuracy.eval({x: val_x , y: val_y} )) #总体准确率
    for i in range(val_im_num):
      output = tf.argmax(pred, 1)
      x_, y_ = sess.run([val_batch_xs, val_batch_ys])     #最好一次只测一幅图片
      outputval, predv = sess.run([output, pred], feed_dict = {x: x_, y: y_})
      if outputval[0] != np.argmax(y_):
        print("in pic ", i, "pred: ",outputval,"actual: ",np.argmax(y_) )
      #if (predv == y_).all():
        #print("True, ", np.argmax(y_))
      #else:
        #print("False, label is ",np.argmax(y_), "but predv, ", predv )
      #im = x_[0]
      #im = im.reshape(-1,28)
      #pylab.imshow(im)
      #pylab.show()

def test():
  print("id,label")
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(val_batch_iterator.initializer)
    ckpt = tf.train.get_checkpoint_state(model_dir)
      
    saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(test_im_num):
      output = tf.argmax(pred, 1)
      x_ = sess.run(test_xs)
      predlabel = sess.run(output, feed_dict = {x: x_})
      print(str(i+1) + ","+ str(predlabel[0]))

def test_buf():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(val_batch_iterator.initializer)
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)
    for i in range(8976, 10000):
      output = tf.argmax(pred, 1)
      x_ = []
      x_.append(test_x[i])
      predlabel = sess.run(output, feed_dict = {x: x_})
      print(str(i) + ","+ str(predlabel[0]))


if __name__== '__main__':
  if is_train:
    #train_buf()
    train()
  else :
    #evaluate()
    test()
