# to train a model to look at images and predict the corresponding digits, in #VMware
# beginners guide

from tensorflow.examples.tutorials.mnist import input_data as input
data= input.read_data_sets("MNIST_data/", one_hot=True)

# import tensorflow
import tensorflow as tf
x = tf.placeholder(dtype=tf.float32,shape= [None, 784], name="input_it")
W = tf.Variable(tf.zeros([784, 10]), name="weight")
b = tf.Variable(tf.zeros([10]), name="bias")
mul= tf.matmul(x,W)
soft_reg= tf.nn.softmax(mul+b)
# new placeholder for holding the correct answer  
y_= tf.placeholder("float",shape= (None, 10), name="new_input")
#find the cross entropy- cost function that is to be minimized
cross_entro= -tf.reduce_sum(y_*tf.log(soft_reg))

#Using Adagrad optimizer to minimize entropy
train_step = tf.train.AdagradOptimizer(learning_rate=0.01, name="Adagrad").minimize(cross_entro)

#initialize the variables created
init = tf.initialize_all_variables()

#launch the model in the session
sess= tf.Session()
sess.run(init)
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)...     
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})... 

#evaluating the model
correct_prediction = tf.equal(tf.argmax(soft_reg,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#print the accuracy of the model
 print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
