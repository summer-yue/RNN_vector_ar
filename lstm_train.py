import tensorflow as tf

from data_generator import generate_new_batch

#Define the properties of encoder and decoder
batch_size = 2
y_length = 1
x_length = 3
N = 6
M = 3
num_hidden = 256
test_batch_num = 10

encoder_inputs = []
decoder_inputs = []
targets = []

def create_feed_dict(encoder_inputs, encoder_input_data, decoder_inputs, decoder_input_data, decoder_targets, decoder_target_data):
    # Lists are not hashable so we need to use this trick to get the feed dict to work
        feed_dict = {}
        for placeholder, data in zip(encoder_inputs, encoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(decoder_inputs, decoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(decoder_targets, decoder_target_data):
            feed_dict[placeholder] = data

        return feed_dict

#[Batch Size, Sequence Length, Input Dimension]
for _ in xrange(N):
    encoder_inputs.append(tf.placeholder(tf.float32, [batch_size, x_length + y_length]))

for _ in xrange(M):
    decoder_inputs.append(tf.placeholder(tf.float32, [batch_size, x_length]))
    targets.append(tf.placeholder(tf.float32, [batch_size, y_length]))
    
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)

#encoder
_, final_state = tf.nn.rnn(cell, encoder_inputs, dtype=tf.float32)

#decoder
decoder_outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs, final_state, cell, scope='decoder')

# Create output weight matrix
W_o = tf.Variable(tf.random_normal([num_hidden, y_length], stddev = 0.35), name='W_o')# weight matrix [y_length , num_hidden]
b_o =  tf.Variable(tf.zeros([y_length, 1]), name='b_o') # bias vector [y_length]

decoder_outputs = list(map(lambda h: tf.matmul(h, W_o) + b_o, decoder_outputs))


loss = 0.0

for output, target in zip(decoder_outputs, targets):
    loss += tf.nn.l2_loss(output - target)
 
loss = loss/M

saver = tf.train.Saver()

optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(loss)


#something to see how bad it is. Maybe percentage that's more than 0.3 away? or something?

#execution
init_op = tf.initialize_all_variables()
sess = tf.Session()
tf.scalar_summary("loss", loss)
writer = tf.train.SummaryWriter("./logs/nn_logs", sess.graph)
merged = tf.merge_all_summaries()
sess.run(init_op)

batch_size = 2  
num_steps = 100000 #num_of_batches
num_step_for_checkpoint = 100
epoch = 2
checkpoint_setting = 1


for i in range(epoch):
    for j in range(num_steps):
        #Encoder: array of length N, each element being a tuple array of two arrays indicating the 
        #input and output for this batch
        #decoder_input: array of length M, each element being an array indicating X values in the batch
        #Target:  array of length M, each element being an array indicating target outputs in the batch
        e_input, d_input, tar = generate_new_batch(batch_size ,y_length, x_length, N, M) 
        feed_dict = create_feed_dict(encoder_inputs, e_input, decoder_inputs, d_input, targets, tar)
        summary, _, train_loss = sess.run([merged, minimize, loss],feed_dict =  feed_dict)
        if (j%num_step_for_checkpoint == 0):
        	#testing
            incorrect = 0.0

            #you should move this up so you can add it to the summaries
            for k in range(test_batch_num):
                e_input, d_input, tar = generate_new_batch(batch_size ,y_length, x_length, N, M) 
                feed_dict = create_feed_dict(encoder_inputs, e_input, decoder_inputs, d_input, targets, tar)
                incorrect += sess.run(loss,feed_dict = feed_dict)
            incorrect = incorrect/test_batch_num 
            print("Epoch %d Step %d Train Loss %f Test Loss %f" %(i, j, train_loss, incorrect))

        writer.add_summary(summary, j)
    
    #Set a checkpoint for every num_steps*checkpoint_setting of batches
    checkpoint_setting -=1 
    if (checkpoint_setting == 0):
    	checkpoint_setting = 1
    else:
    	save_path = saver.save(sess, "/tmp/model.ckpt")
    	print("Model saved in file: %s" % save_path)


sess.close()
