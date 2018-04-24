
import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
from imagernn.utils import initw

class GRUGenerator:
    def init(input_size, hidden_size, output_size):
      model = {}
      # Recurrent weights: take x_t, h_{t-1}, and bias unit
      # and produce the 3 gates and the input to cell signal
      model['ones'] = np.ones(1,3*hidden_size)
      model['U'] = initw(input_size , 3 * hidden_size)
      model['W'] = initw(hidden_size , 3 * hidden_size)
      model['GUR'] = np.row_stack([model['ones'],model['U'],model['W']])
      # Decoder weights (e.g. mapping to vocabulary)
      model['Wd'] = initw(hidden_size, output_size)  # decoder
      model['bd'] = np.zeros((1, output_size))
      print "=============initialize lstm generator=========="

      # Theano: Created shared variables
      model['Wd'] = theano.shared(name='Wd', value=model['Wd'].astype(theano.config.floatX))
      model['bd'] = theano.shared(name='bd', value=model['bd'].astype(theano.config.floatX))
      model['GRU'] = theano.shared(name='GRU', value=model['GRU'].astype(theano.config.floatX))

      update = ['GUR', 'Wd', 'bd']
      regularize = ['GUR', 'Wd']
      return {'model': model, 'update': update, 'regularize': regularize}

    def forward(Xi, Xs, model, params, **kwargs):
      """
      Xi is 1-d array of size D (containing the image representation)
      Xs is N x D (N time steps, rows are data containng word representations), and
      it is assumed that the first row is already filled in as the start token. So a
      sentence with 10 words will be of size 11xD in Xs.
      """
      predict_mode = kwargs.get('predict_mode', False)

      # Google paper concatenates the image to the word vectors as the first word vector
      X = np.row_stack([Xi, Xs])

      # options
      # use the version of LSTM with tanh? Otherwise dont use tanh (Google style)
      # following http://arxiv.org/abs/1409.3215
      tanhC_version = params.get('tanhC_version', 0)
      drop_prob_encoder = params.get('drop_prob_encoder', 0.0)
      drop_prob_decoder = params.get('drop_prob_decoder', 0.0)

      if drop_prob_encoder > 0:  # if we want dropout on the encoder
        # inverted version of dropout here. Suppose the drop_prob is 0.5, then during training
        # we are going to drop half of the units. In this inverted version we also boost the activations
        # of the remaining 50% by 2.0 (scale). The nice property of this is that during prediction time
        # we don't have to do any scailing, since all 100% of units will be active, but at their base
        # firing rate, giving 100% of the "energy". So the neurons later in the pipeline dont't change
        # their expected firing rate magnitudes
        if not predict_mode:  # and we are in training mode
          scale = 1.0 / (1.0 - drop_prob_encoder)
          U = (np.random.rand(*(X.shape)) < (1 - drop_prob_encoder)) * scale  # generate scaled mask
          X *= U  # drop!

      # follows http://arxiv.org/pdf/1409.2329.pdf

      U = model['U']
      W = model['W']
      ones = model['ones']
      n = X.shape[0]
      d = model['Wd'].shape[0]  # size of hidden layer
      output_size = model['W'].shape[1]
      z,r,c,s = np.zeros((n,d))

      s[-1] = np.zeros(d)
      for t in xrange(n):
        # set input

        z[t] = T.nnet.hard_sigmoid(U[t,:d].dot(X[t]) + W[t,:d].dot(s[t-1]))

        r[t] = T.nnet.hard_sigmoid(U[t,d:2*d].dot(X[t]) + W[t,d:2*d].dot(s[t-1]))
        c[t] = T.tanh(U[t,2*d:].dot(X[t]) + W[t,2*d:].dot(s[t-1] * r[t]))
        s[t] = (T.ones_like(z[t]) - z[t]) * c[t] + z[t] * s[t-1]

      if drop_prob_decoder > 0:  # if we want dropout on the decoder
        if not predict_mode:  # and we are in training mode
          scale2 = 1.0 / (1.0 - drop_prob_decoder)
          U2 = (np.random.rand(*(s.shape)) < (1 - drop_prob_decoder)) * scale2  # generate scaled mask
          s *= U2  # drop!

      # decoder at the end
      Wd = model['Wd']
      bd = model['bd']
      # NOTE1: we are leaving out the first prediction, which was made for the image
      # and is meaningless.
      Y = s[1:, :].dot(Wd) + bd

      cache = {}
      if not predict_mode:
        # we can expect to do a backward pass
        cache['U']= U
        cache['W'] = W
        cache['GRU'] =  np.row_stack([model['ones'],model['U'],model['W']])
        cache['s'] = s
        cache['Wd'] = Wd
        cache['bd'] = bd
        cache['X'] = X
        cache['ones'] = ones
        cache['tanhC_version'] = tanhC_version
        cache['drop_prob_encoder'] = drop_prob_encoder
        cache['drop_prob_decoder'] = drop_prob_decoder
        cache['Y'] = Y
        if drop_prob_encoder > 0: cache['U'] = U  # keep the dropout masks around for backprop
        if drop_prob_decoder > 0: cache['U2'] = U2

      return Y, cache

    def backward(dY, cache):

      Wd = cache['Wd']
      bd = cache['bd']
      s = cache['s']
      ones = cache['ones']
      GRU = cache['GRU']
      U = cache['U']
      W = cache['W']
      X = cache['X']
      Y = cache['Y']
      # tanhC_version = cache['tanhC_version']
      drop_prob_encoder = cache['drop_prob_encoder']
      drop_prob_decoder = cache['drop_prob_decoder']
      n, d = s.shape

      prediction = T.argmax(Y, axis=1)
      o_error = T.sum(T.nnet.categorical_crossentropy(Y, y))
      cost = o_error
      dones = T.grad(cost,ones)
      dU = T.grad(cost, U)
      dW = T.grad(cost, W)
      dWb = T.grad(cost, Wd)
      dbd = T.grad(cost, bd)
      dX = T.grad(cost, X)
      dGRU = np.row_stack([dones,dU,dW])
      return {'GRU': dGRU, 'Wd': dWd, 'bd': dbd, 'dXi': dX[0, :], 'dXs': dX[1:, :]}



    def predict(Xi, model, Ws, params, **kwargs):
      """
      Run in prediction mode with beam search. The input is the vector Xi, which
      should be a 1-D array that contains the encoded image vector. We go from there.
      Ws should be NxD array where N is size of vocabulary + 1. So there should be exactly
      as many rows in Ws as there are outputs in the decoder Y. We are passing in Ws like
      this because we may not want it to be exactly model['Ws']. For example it could be
      fixed word vectors from somewhere else.
      """
      tanhC_version = params['tanhC_version']
      beam_size = kwargs.get('beam_size', 1)

      GRU = model['GRU']
      d = model['Wd'].shape[0]  # size of hidden layer
      Wd = model['Wd']
      bd = model['bd']
      U = model['U']
      W = model['W']

      # lets define a helper function that does a single LSTM tick
      def GRUtick(X, h_prev):
        t = 0
        z, r, c, s = np.zeros((1, d))

        z = T.nnet.hard_sigmoid(U[t, :d].dot(X) + W[t, :d].dot(h_prev))

        r = T.nnet.hard_sigmoid(U[t, d:2 * d].dot(X[t]) + W[t, d:2 * d].dot(h_prev))
        c = T.tanh(U[t, 2 * d:].dot(X) + W[t, 2 * d:].dot(h_prev * r[t]))
        s = (T.ones_like(z[t]) - z[t]) * c[t] + z[t] * s[t - 1]

        Y = s.dot(Wd) + bd
        return (Y, s)  # return output, new hidden, new cell

      # forward prop the image
      (y0, h) = GRUtick(Xi, np.zeros(d))

      # perform BEAM search. NOTE: I am not very confident in this implementation since I don't have
      # a lot of experience with these models. This implements my current understanding but I'm not
      # sure how to handle beams that predict END tokens. TODO: research this more.
      if beam_size > 1:
        # log probability, indices of words predicted in this beam so far, and the hidden and cell states
        beams = [(0.0, [], h)]
        nsteps = 0
        while True:
          beam_candidates = []
          for b in beams:
            ixprev = b[1][-1] if b[1] else 0  # start off with the word where this beam left off
            if ixprev == 0 and b[1]:
              # this beam predicted end token. Keep in the candidates but don't expand it out any more
              beam_candidates.append(b)
            (y1, h1) = GRUtick(Ws[ixprev], b[2])
            y1 = y1.ravel()  # make into 1D vector
            maxy1 = np.amax(y1)
            e1 = np.exp(y1 - maxy1)  # for numerical stability shift into good numerical range
            p1 = e1 / np.sum(e1)
            y1 = np.log(1e-20 + p1)  # and back to log domain
            top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
            for i in xrange(beam_size):
              wordix = top_indices[i]
              beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1))
          beam_candidates.sort(reverse=True)  # decreasing order
          beams = beam_candidates[:beam_size]  # truncate to get new beams
          nsteps += 1
          if nsteps >= 20:  # bad things are probably happening, break out
            break
        # strip the intermediates
        predictions = [(b[0], b[1]) for b in beams]
      else:
        # greedy inference. lets write it up independently, should be bit faster and simpler
        ixprev = 0
        nsteps = 0
        predix = []
        predlogprob = 0.0
        while True:
          (y1, h) = GRUtick(Ws[ixprev], h)
          ixprev, ixlogprob = ymax(y1)
          predix.append(ixprev)
          predlogprob += ixlogprob
          nsteps += 1
          if ixprev == 0 or nsteps >= 20:
            break
        predictions = [(predlogprob, predix)]

      return predictions


def ymax(y):
  """ simple helper function here that takes unnormalized logprobs """
  y1 = y.ravel()  # make sure 1d
  maxy1 = np.amax(y1)
  e1 = np.exp(y1 - maxy1)  # for numerical stability shift into good numerical range
  p1 = e1 / np.sum(e1)
  y1 = np.log(1e-20 + p1)  # guard against zero probabilities just in case
  ix = np.argmax(y1)
  return (ix, y1[ix])
