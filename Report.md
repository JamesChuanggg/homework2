# Homework2 - Policy Gradient 

## Simple Policy Network
#### Policy Network
Here we use 2-layer neural network to represent the policy. Make sure you add softmax layer to represent probability distribution.
```python
self.f1 = tf.contrib.layers.fully_connected(inputs=self._observations, num_outputs=self.hidden_dim, activation_fn=tf.tanh)
self.f2 = tf.contrib.layers.fully_connected(inputs=self.f1, num_outputs=self.out_dim, activation_fn=None)
probs = tf.nn.softmax(self.f2)
```

#### Compute the surrogate loss
Since the optimizer in Tensorflow only support minimizing loss (gradient descent), so we simply add a minus sign to represent **gradient ascent**.
```python
surr_loss = -tf.reduce_mean(tf.mul(log_prob, self._advantages))	# minus means "maximize"
```

#### Compute the accumulated discounted rewards at each timestep
Construct a simple for-loop to calculate the accumulated discounted from the end of the game to the start.
```python
def discount_cumsum(x, discount_rate):
    discount_x = np.zeros(len(x))
    for i in range(len(x)):
	    if i > 0:
	      discount_x[len(x)-1-i] = x[len(x)-1-i] + discount_x[len(x)-i]*discount_rate
	    else:
	      discount_x[len(x)-1] = x[len(x)-1]
    return discount_x
```

#### Use baseline to reduce the variance
We simply calculate the advantage function by substracting the baseline from orginal reward.
```python
a = r - b
```

## Baseline or not?

## Why we need to normalize the advantages?
In practice it is important to normalize advantage function. One good idea is to “standardize” these returns (e.g. subtract mean, divide by standard deviation) before we plug them into backprop. 
This way we’re always encouraging and discouraging roughly half of the performed actions. Mathematically you can also interpret these tricks as a way of controlling the variance of the policy gradient estimator.
