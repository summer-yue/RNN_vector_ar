import tensorflow as tf

#Define the properties of encoder and decoder
batch_size = 2
y_length = 1
x_length = 3
N = 5
M = 3

#[Batch Size, Sequence Length, Input Dimension]
data = tf.placeholder(tf.float32, [batch_size, N, x_length + y_length])
#[Batch Size, Sequence Length, output Dimension]
target = tf.placeholder(tf.float32, [batch_size, M, y_length])

num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)

#val = tf.transpose(val, [1, 0, 2])

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(prediction))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

#execution
init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 2
no_of_batches = 10
epoch = 500
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
    	#Encoder: array of length N, each element being a tuple array of two arrays indicating the 
		#input and output for this batch
		#decoder_input: array of length M, each element being an array indicating X values in the batch
		#Target:  array of length M, each element being an array indicating target outputs in the batch
		encoder_input, decoder_input, target = generate_new_batch(batch_size ,y_length, x_length, N, M) 
        sess.run(minimize,{data: decoder_input, target: target})

#testing
_, test_decoder_input, test_target = generate_new_batch(batch_size ,y_length, x_length, N, M) 
incorrect = sess.run(error,{data: test_decoder_input, target: test_target})
print(incorrect)
sess.close()


