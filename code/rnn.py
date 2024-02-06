# coding: utf-8
from rnnmath import *
from model import Model, is_param, is_delta

class RNN(Model):
	'''
	This class implements Recurrent Neural Networks.
	
	You should implement code in the following functions:
		predict				->	predict an output sequence for a given input sequence
		acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
		acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
		acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
		acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions

	Do NOT modify any other methods!
	Do NOT change any method signatures!
	'''
	
	def __init__(self, vocab_size, hidden_dims, out_vocab_size):
		'''
		initialize the RNN with random weight matrices.
		
		DO NOT CHANGE THIS - The order of the parameters is important and must stay the same.
		
		vocab_size		size of vocabulary that is being used
		hidden_dims		number of hidden units
		out_vocab_size	size of the output vocabulary
		'''

		super().__init__(vocab_size, hidden_dims, out_vocab_size)

		# matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
		with is_param():
			self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
			self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
			self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)

		# matrices to accumulate weight updates
		with is_delta():
			self.deltaU = np.zeros_like(self.U)
			self.deltaV = np.zeros_like(self.V)
			self.deltaW = np.zeros_like(self.W)

	def predict(self, x):
		'''
		predict an output sequence y for a given input sequence x
		
		x	list of words, as indices, e.g.: [0, 4, 2]
		
		returns	y,s
		y	matrix of probability vectors for each input word
		s	matrix of hidden layers for each input word
		
		'''
		
		# matrix s for hidden states, y for output states, given input x.
		# rows correspond to times t, i.e., input words
		# s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
		s = np.zeros((len(x) + 1, self.hidden_dims))
		y = np.zeros((len(x), self.out_vocab_size))

		for t in range(len(x)):
			# On the first time step, the hidden state is calculated without including information from previos state, as such does not exist.
			# It is simply calculated via taking the sigmoid (equation 1 in the instructions) of the Vx_t function
			if t == 0:
				s[t] = sigmoid(np.dot(self.V, make_onehot(x[t], self.vocab_size)))
			else:
				# Else an information from the previous states is includes (via equations 2 and 1 from the introductions)
				s[t] = sigmoid(np.dot(self.V, make_onehot(x[t], self.vocab_size)) + np.dot(self.U, s[t-1]))
			# In the end we compute the outputs (via equation 3 and 4 from the instructions)
			y[t] = softmax(np.dot(self.W, s[t]))


		return y, s
	
	def acc_deltas(self, x, d, y, s):
		'''
		accumulate updates for V, W, U
		standard back propagation
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		
		x	list of words, as indices, e.g.: [0, 4, 2]
		d	list of words, as indices, e.g.: [4, 2, 3]
		y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
			should be part of the return value of predict(x)
		s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
			should be part of the return value of predict(x)
		
		no return values
		'''

		for t in reversed(range(len(x))):
			# Updating the W matrix (equations 8 and 9)
			temp_deltaW = make_onehot(d[t], self.vocab_size) - y[t]
			self.deltaW += np.outer(temp_deltaW,s[t])

			# Updating the V matrix (equations 10 and 11)
			temp_deltaV = np.multiply(np.dot(self.W.T,temp_deltaW), grad(s[t]))
			self.deltaV += np.outer(temp_deltaV, make_onehot(x[t], self.vocab_size))
			
			# Updating the U matrix (equation 14)
			self.deltaU += np.outer(temp_deltaV, s[t - 1])		

	def acc_deltas_np(self, x, d, y, s):
		'''
		accumulate updates for V, W, U
		standard back propagation
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		for number prediction task, we do binary prediction, 0 or 1

		x	list of words, as indices, e.g.: [0, 4, 2]
		d	array with one element, as indices, e.g.: [0] or [1]
		y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
			should be part of the return value of predict(x)
		s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
			should be part of the return value of predict(x)
		
		no return values
		'''

		# Updating the W matrix just for the last time step (equations 8 and 9)
		temp_deltaW = make_onehot(d[0], self.vocab_size) - y[len(x) - 1]
		self.deltaW += np.outer(temp_deltaW,s[len(x) - 1])

		# Updating the V matrix just for the last time step (equations 10 and 11)
		temp_deltaV = np.multiply(np.dot(self.W.T,temp_deltaW), grad(s[len(x) - 1]))
		self.deltaV += np.outer(temp_deltaV, make_onehot(x[len(x) - 1], self.vocab_size))
		
		# Updating the U matrix just for the last time step (equation 14)
		self.deltaU += np.outer(temp_deltaV, s[len(x) - 2])
		
	def acc_deltas_bptt(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)
		
		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		
		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]
		y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
				should be part of the return value of predict(x)
		s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
				should be part of the return value of predict(x)
		steps	number of time steps to go back in BPTT
		
		no return values
		'''
			
		for t in reversed(range(len(x))):
			# Updating the W matrix (equations 8 and 9)
			temp_deltaW = make_onehot(d[t], self.vocab_size) - y[t]
			self.deltaW += np.outer(temp_deltaW,s[t])

			# Updating the V matrix (equations 10 and 11)
			temp_deltaV = np.multiply(np.dot(self.W.T,temp_deltaW), grad(s[t]))
			self.deltaV += np.outer(temp_deltaV, make_onehot(x[t], self.vocab_size))
			
			# Updating the U matrix (equation 14)
			self.deltaU += np.outer(temp_deltaV, s[t - 1])

			# The V and U matrices are additionally updated r (step) times
			for r in range(steps):
				if r + 1 > t:
					break
				temp_deltaV = np.multiply(np.dot(self.U.T, temp_deltaV), grad(s[t - r - 1]))
				self.deltaV += np.outer(temp_deltaV, make_onehot(x[t - r - 1], self.vocab_size))
				self.deltaU += np.outer(temp_deltaV, s[t - r - 2])

	def acc_deltas_bptt_np(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)

		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
		for number prediction task, we do binary prediction, 0 or 1

		x	list of words, as indices, e.g.: [0, 4, 2]
		d	array with one element, as indices, e.g.: [0] or [1]
		y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
				should be part of the return value of predict(x)
		s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
				should be part of the return value of predict(x)
		steps	number of time steps to go back in BPTT

		no return values
		'''
		
		# Updating the W matrix just for the last time step (equations 8 and 9)
		temp_deltaW = make_onehot(d[0], self.vocab_size) - y[len(x) - 1]
		self.deltaW += np.outer(temp_deltaW,s[len(x) - 1])

		# Updating the V matrix just for the last time step (equations 10 and 11)
		temp_deltaV = np.multiply(np.dot(self.W.T,temp_deltaW), grad(s[len(x) - 1]))
		self.deltaV += np.outer(temp_deltaV, make_onehot(x[len(x) - 1], self.vocab_size))
		
		# Updating the U matrix just for the last time step(equation 14)
		self.deltaU += np.outer(temp_deltaV, s[len(x) - 2])

		# The V and U matrices are additionally updated r (step)
		for r in range(steps):
			if r + 1 > len(x) - 1:
				break
			temp_deltaV = np.multiply(np.dot(self.U.T, temp_deltaV), grad(s[len(x) - 1 - r - 1]))
			self.deltaV += np.outer(temp_deltaV, make_onehot(x[len(x) - 1 - r - 1], self.vocab_size))
			self.deltaU += np.outer(temp_deltaV, s[len(x) - 1 - r - 2])