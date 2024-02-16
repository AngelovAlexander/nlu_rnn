# coding: utf-8

from rnnmath import *
from gru_abstract import GRUAbstract


class GRU(GRUAbstract):
    '''
    This class implements Gated Recurrent Unit (GRU).

    You should implement code in the following functions:
        forward				->	forward pass for a single GRU cell
        acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
        acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''
    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
         DO NOT CHANGE THIS

        The GRU parameters given below are initialized in GRUAbstract class

        self.Ur, self.deltaUr
        self.Vr, self.deltaVr
        self.Uz, self.deltaUz
        self.Vz, self.deltaVz
        self.Uh, self.deltaUh
        self.Vh, self.deltaVh
        self.W,  self.deltaW

        vocab_size		size of vocabulary that is being used
        hidden_dims		number of hidden units
        out_vocab_size	size of the output vocabulary
        '''
        super().__init__(vocab_size, hidden_dims, out_vocab_size)

    def forward(self, x, s_previous):
        '''
        Unlike the RNN this is just the forward step for a single GRU cell. The input is a single
        index x rather than the entire sequence

        x	index of word at current time step
        s_previous   vector of the previous hidden layer

        returns	y,s
        y	probability vector for the input word x
        s	hidden layer for current time step

        '''

        # one hot encode word x
        x_in = make_onehot(x, self.vocab_size)
        # reset gate (equation 18)
        r = sigmoid(self.Vr @ x_in + self.Ur @ s_previous)
        # update gate (equation 19)
        z = sigmoid(self.Vz @ x_in + self.Uz @ s_previous)
        # candidate hidden state (equation 20)
        h = np.tanh(self.Vh @ x_in + self.Uh @ (r * s_previous))
        # hidden state
        s = z * s_previous + (1 - z) * h
        # final output
        y = softmax(self.W @ s)
        return y, s, h, z, r


    def acc_deltas_np(self, x, d, y, s):
        '''
        accumulate updates for Vr, Ur, Vh, Uh, Vz, Uz, W
        standard back propagation

        this should not update Vr, Ur, Vh, Uh, Vz, Uz, W directly. instead, calculate delta output (the gradient with
        repsect to the output) and pass it to self.backward() method of GRUAbstract

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)

        no return values
        '''

        # Calculating the delta output for the last time steps (equation 9)
        delta_output = make_onehot(d[0], self.vocab_size) - y[len(x) - 1]
        # backpropogation
        self.backward(x, len(x) - 1, s, delta_output)

    def acc_deltas_bptt_np(self, x, d, y, s, steps):
        '''
        accumulate updates for Vr, Ur, Vh, Uh, Vz, Uz, W
        back propagation through time BPTT

        this should not update Vr, Ur, Vh, Uh, Vz, Uz, W directly. instead, calculate delta output (the gradient with
        respect to the output) and pass it to self.backward() method of GRUAbstract. This method has a setps
        parameter for the num ber of backward steps.

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)

        no return valuess
        '''
        # Calculating the delta output for the last time steps (equation 9)
        delta_output = make_onehot(d[0], self.vocab_size) - y[len(y) - 1]
        # backpropogation through time
        self.backward(x, len(x) - 1, s, delta_output, steps)