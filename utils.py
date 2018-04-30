import torch
import random
from torch import distributions

def run_words(net, vocab, setence_len=5, cuda=False):
    """given a network and valid vocabulary, let the network generate a setence
    """
    # create hidden state
    ho = net.init_hidden()
    # create random word index
    x_in = torch.LongTensor([random.randint(0,len(vocab)-1)])
    if cuda:
        x_in, ho = x_in.cuda(), ho.cuda()
    # create output index
    output = [int(x_in)]
    # now we iterate through our setence, pasing x_in to get y_out, and setting y_out as x_in for the next time step
    for i in range(setence_len):
        y_out, ho = net.forward(x_in, ho)
        dist = distributions.Categorical(probs=y_out.exp())
        # get max val and index
        sample = dist.sample()
        output.append(int(sample))
        x_in = sample

    # now we print our word
    print_out = ''
    for item in output:
        print_out += vocab[item]
    return print_out


def get_batch(text_corpus, batch_size=100):
    start = random.randint(0, len(text_corpus)-batch_size)
    end = start + batch_size + 1
    return text_corpus[start:end]
