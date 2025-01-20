import tensorflow as tf

mb_rewards_human = tf.constant([5.0, 10.0, 15.0, 20.0], dtype=tf.float32)
mb_rewards = tf.constant([3.0, 8.0, 12.0, 18.0], dtype=tf.float32)

# Compute the result
result = tf.divide(tf.maximum(0, mb_rewards_human - mb_rewards), tf.reduce_max(mb_rewards_human))

# Print the results using tf.print
tf.print("tf.reduce_max(mb_rewards_human):", tf.reduce_max(mb_rewards_human))
tf.print("Result:", result)

# Compute dot product
dot_product = tf.tensordot(mb_rewards_human, mb_rewards, axes=1)
tf.print("Dot product result:", dot_product)