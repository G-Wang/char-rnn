import torch
from torch import nn
import numpy as np
import unidecode
import string
import random
from torch import distributions

from utils import get_batch, run_words
from models import RNN

train_data = unidecode.unidecode(open('sherlock.txt').read()) # load the text file, reading it
vocab = string.printable # use all printable string characters as vocabulary
vocab_length = len(vocab) # vocabulary length
data_len = len(train_data) # get length of training data
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def no_test_forward():
    loss_func = torch.nn.functional.nll_loss
    net = RNN(100, 100, 100)
    net.to(device) # add cuda to device
    optim = torch.optim.Adam(net.parameters(),lr=1e-4)
    # step 2: create a training batch of data, size 101, format this data and convert it to pytorch long tensors
    dat = get_batch(train_data,100)
    dat = torch.LongTensor([vocab.find(item) for item in dat])
    # step 3: convert our dat into input/output
    x_t = dat[:-1]
    y_t = dat[1:]
    ho = net.init_hidden()
    # remember to load all variables used by the model to the device, this means the i/o as well as the hidden state
    x_t, y_t, ho = x_t.to(device), y_t.to(device), ho.to(device)
    # test forward pass
    log_prob, hidden = net.forward(x_t, ho)
    # let's see if the forward pass of the next hidden state is already cuda
    #log_prob2, hidden2 = net.forward(x_t, hidden)
    loss = loss_func(log_prob, y_t)
    optim.zero_grad()
    loss.backward()
    optim.step()



def test_running():
    epochs = 100000
    seq_batch_size = 100
    print_yes = 100
    loss_func = torch.nn.functional.nll_loss
    # create network and optimizer
    net = RNN(100, 120, 150, 2)
    net.to(device) # add cuda to device
    optim = torch.optim.Adam(net.parameters(),lr=3e-5)
    # main training loop:
    for epoch in range(epochs):
        dat = get_batch(train_data,seq_batch_size)
        dat = torch.LongTensor([vocab.find(item) for item in dat])
        # pull x and y
        x_t = dat[:-1]
        y_t = dat[1:]
        hidden = net.init_hidden()
        # turn all into cuda
        x_t, y_t, hidden = x_t.to(device), y_t.to(device), hidden.to(device)
        # initialize hidden state and forward pass
        logprob, hidden = net.forward(x_t, hidden)
        loss = loss_func(logprob, y_t)
        # update
        optim.zero_grad()
        loss.backward()
        optim.step()
        # print the loss for every kth iteration
        if epoch % print_yes == 0:
            print('*'*100)
            print('\n epoch {}, loss:{} \n'.format(epoch, loss))
            # make sure to pass True flag for running on cuda
            print('sample speech:\n', run_words(net, vocab, 500, True))
