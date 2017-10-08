#This code has been authored by Ankur Kumar (@dopetard on Medium) 
#A big shoutout and thanks to the man for writing such an amazing tutorial (Link in Description)
#I have merely followed the steps and made a few changes of mine

import tensorflow as tf #importing the tensorflow library
T, F = 1.0, -1.0 #True has the +1.0 value and has F -1.0, it's important to
#note that you can assign any value to them

bias = 1.0

training_input = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],

]

training_output = [
    [T],
    [T],
    [T],
    [F],
]
tf.set_random_seed(1) #HELPS MAJORLY in Debugging 
w = tf.Variable(tf.random_normal([3,1]), dtype = tf.float32) #Random Normal Distribution 

# step(x) = {1 if x > 0; -1 otherwise } or SIGNUM function

def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater) # 1 if True and 0 if False
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1) #if it is greater, returns 1 or else returns -1

output = step(tf.matmul(training_input, w)) #Multiplying the entire training_input by the weight matrix
error = tf.subtract(training_output, output)
mse = tf.reduce_mean(tf.square(error)) #Computes the mean across all elements along all axes if the axis isn't specified

delta = tf.matmul(training_input, error, transpose_a = True) #delta is the desired adjustment 
train = tf.assign(w, tf.add(w, delta)) #adjusts the weights 

sess = tf.Session()
sess.run(tf.global_variables_initializer())

err, target = 1, 0
epoch, max_epochs = 0, 10 #epochs are iterations 

while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])


#evaluating mse to track progress
#evaluatiing train to make adjustments

print('epoch:', epoch, 'mse:', err)

print(sess.run(w))
