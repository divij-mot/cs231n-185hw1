from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_train = X.shape[0]
    for i in range(num_train):
      scores = X[i].dot(W)

      # # compute the probabilities in numerically stable way
      # scores -= np.max(scores)
      # p = np.exp(scores)
      # p /= p.sum()  # normalize
      # logp = np.log(p)

      # loss -= logp[y[i]]  # negative log probability is the loss
      exp_vector = np.exp(scores - scores.max())
      softmax = exp_vector/exp_vector.sum()
      loss -= np.log(softmax[y[i]])
      softmax[y[i]] -= 1
      dW += np.outer(X[i], softmax)

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_train = X.shape[0]
    scores = X @ W
    scores -= np.max(scores, axis = 1, keepdims = True)
    exp_scores = np.exp(scores)
    softmax = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    probability_correct = softmax[np.arange(num_train), y]
    loss = -np.sum(np.log(probability_correct))
    loss = loss/num_train
    loss += reg  * np.sum(W * W)
    copy = softmax.copy()
    copy[np.arange(num_train), y] -= 1
    dW = (X.T @ copy) / num_train
    dW += 2 * reg * W
    



    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################


    return loss, dW
