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

epochs = 100000
seq_batch_size = 100
print_yes = 100
loss_func = torch.nn.functional.nll_loss

def main():
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

if __name__=="__main__":
    main()
