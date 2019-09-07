import tensorflow as tf

x = tf.Variable(1, name="first_variable")
y = tf.constant(2)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(x))
print(sess.run(y))

