from __future__ import division

import sys
from functions import fill_batch, make_dict, take_len
import numpy as np
import collections

import torchwordemb
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

import pickle
import generators as gens
import random
import time
import warnings

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

train_txt = "fce-data/train.kaneko.txt"
dev_txt = "fce-data/dev.kaneko.txt"
test_txt = "fce-data/test.kaneko.txt"
load_model  = "ptmodel/ptBLSTM.model0"

vocab_size = take_len(train_txt)
batch_size = 64
embed_size = 300
output_size = 2
hidden_size = 200
extra_hidden_size = 50
epoch = 20
gpu = True


random.seed(0)

def precision_recall_f(pres, tags, cons, use_cuda=False):
    c_p = 0  # actual
    correct_p = 0  # predicted
    c_r = 0
    correct_r = 0
    _tags = np.array(tags, dtype=np.int64)
    tags = Variable(torch.from_numpy(_tags).t())
    if use_cuda:
        tags = tags.cuda()

    # pres is the post-soft max probabilities
    # num is index, a is (pres, tag)
    for num, a in enumerate(zip(pres, tags)):
        pre_l = [a[0].data[k].max(0)[1].cpu().numpy()[0] for k in range(len(a[0])) if cons[num][k] == True]
        tag_l = [int(a[1].data[n]) for n in range(len(a[1].data)) if cons[num][n] == True]
        for a, b in zip(tag_l, pre_l):
            if a == 1:
                c_r += 1
                if b == a:
                    correct_r += 1
            if b == 1:
                c_p += 1
                if b == a:
                    correct_p += 1
    return c_p, correct_p, c_r, correct_r


def evaluate(model, word2id):
    c_p = 0
    correct_p = 0
    c_r = 0
    correct_r = 0

    gen1 = gens.word_list(dev_txt)
    gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
    batchs = [b for b in gen2]
    for batch in batchs:
        tag0 = batch[:]
        tags = [a[:-1]  for a in tag0]
        batch = [b[1:] for b in batch]
        batch = fill_batch([b[-1].split() for b in batch])
        tags = fill_batch(tags, token=-1)
        pres, cons = forward(batch, tags, model, word2id, mode=False, use_cuda=gpu)
        a, b, c, d =  precision_recall_f(pres, tags, cons)
        c_p += a
        correct_p += b
        c_r += c
        correct_r += d
    try:
        precision = correct_p/c_p
        recall = correct_r/c_r
        f_measure = (1 + 0.5**2)*precision*recall/(0.5**2*precision + recall)
        print('Precision:\t{}'.format(precision))
        print('Recall:\t{}'.format(recall))
        print('F-value\t{}'.format(f_measure))
    except ZeroDivisionError:
        precision = 0
        recall = 0
        f_measure = 0
        print('Precision:\tnothing')
        print('Recall:\tnothing')
        print('F-value\tnothing')
    return precision, recall, f_measure


class BiLSTMw2v(nn.Module):

    def __init__(self, _vocab_size, embed_size, hidden_size, output_size, extra_hidden_size, use_cuda=False):
        super(BiLSTMw2v, self).__init__()
        self.vocab_size = _vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_cuda = use_cuda

        self.x2e = nn.Embedding(_vocab_size, embed_size)
        self.e2h_for = nn.LSTM(embed_size, hidden_size)
        self.e2h_back = nn.LSTM(embed_size, hidden_size)
        self.h2s = nn.Linear(hidden_size*2, extra_hidden_size)
        self.s2o = nn.Linear(extra_hidden_size, output_size)

        self.blstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True)
        self.relu = nn.ReLU()

        self.e2h_for_hidden = self._init_hidden()
        self.e2h_back_hidden = self._init_hidden()
        self.blstm_hidden = self._init_hidden(2)

    def _init_hidden(self, layers=1):
        h = Variable(torch.zeros(layers, 1, self.hidden_size))
        cell = Variable(torch.zeros(layers, 1, self.hidden_size))
        if self.use_cuda:
            h = h.cuda()
            cell = cell.cuda()
        return (h, cell)

    def _reset_state(self):
        self.zero_grad()
        self.e2h_for_hidden = self._init_hidden()
        self.e2h_back_hidden = self._init_hidden()
        self.blstm_hidden = self._init_hidden(2)


    def initialize_embed(self, word2vec_model, word2id):
        w2v_vocab, w2v_vectors = torchwordemb.load_word2vec_text(word2vec_model)
        for word, i in word2id.items():
            # ignore the unicode conversion/comparison warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if word in w2v_vocab.keys():
                    self.x2e.weight.data[i].copy_(w2v_vectors[w2v_vocab[word]])


    def forward(self, x):
        self._reset_state()
        e_states = []
        h_back_states = []
        h_states = []
        o_states = []

        # list of tensors of size batch_size x embed_size
        e_states = [self.relu(self.x2e(w)) for w in x]
        for e in e_states:
            inp = e.view(len(e), 1, -1)
            out, self.blstm_hidden = self.blstm(inp, self.blstm_hidden)
            out = out.view(len(e), -1)
            h_states.append(out)

        for h in h_states:
            o_states.append(self.s2o(self.relu(self.h2s(h))).view(-1, self.output_size))

#       print(o_states)
        return o_states


def forward(batchs, tags, model, word2id, mode, use_cuda=False):
    softmax = nn.Softmax()
    cross_entropy_loss = nn.CrossEntropyLoss()

    if mode:  # are we calculating the loss??
        accum_loss = Variable(torch.zeros(1))  # initialize the loss count to zero
        x = Variable(torch.LongTensor([[word2id[word] if word in word2id else word2id['<unk>'] for word in sen] for sen in batchs])).t()
        _tags = np.array(tags, dtype=np.int64)
        tags = Variable(torch.from_numpy(_tags)).t()

        if use_cuda:
            accum_loss = accum_loss.cuda()
            x = x.cuda()
            tags = tags.cuda()

        pres = model(x)

        for tag, pre in zip(tags, pres):
            accum_loss += cross_entropy_loss(pre, tag)

        sortmax_pres = [softmax(pre) for pre in pres]
        condition = x.data != -1
        return accum_loss, sortmax_pres, condition

    else:
        x = Variable(torch.LongTensor([[word2id[word] if word in word2id else word2id['<unk>'] for word in sen] for sen in batchs])).t()

        if use_cuda:
            x = x.cuda()

        pres = model(x)
        sortmax_pres = [softmax(pre) for pre in pres]
        condition = x.data != -1
        return sortmax_pres, condition


def train():
    id2word = {}
    word2id = {}
    word_freq = collections.defaultdict(lambda: 0)
    id2word[0] = "<unk>"
    word2id["<unk>"] = 0
    id2word[1] = "<s>"
    word2id["<s>"] = 1
    id2word[2] = "</s>"
    word2id["</s>"] = 2

    word2id, id2word, word_list, word_freq = make_dict(train_txt, word2id, id2word, word_freq)
    model = BiLSTMw2v(vocab_size, embed_size, hidden_size, output_size, extra_hidden_size, use_cuda=gpu)
    model.initialize_embed('embedding.txt', word2id)
    if gpu:
        model.cuda()
    opt = optim.Adam(model.parameters(), lr=0.001)

    best_epoch = -1
    best_fscore = -1

    for i in range(1, epoch + 1):
        print("\nepoch {}".format(i))
        start = time.time()
        total_loss = 0
        gen1 = gens.word_list(train_txt)
        gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
        batchs = [b for b in gen2]
        bl = list(range(len(batchs)))
        random.shuffle(bl)
        for n, j in enumerate(bl):
            opt.zero_grad()
            tag0 = batchs[j][:]
            tags = [[int(c) for c in a[:-1]] for a in tag0]
            batch = fill_batch([b[-1].split() for b in batchs[j]])
            tags = fill_batch(tags, token=0)
            accum_loss, pres, cons = forward(batch, tags, model, word2id, mode=True, use_cuda=gpu)
            accum_loss.backward()
            opt.step()
            total_loss += accum_loss.data[0]
        print("total_loss {}".format(total_loss))

        torch.save(model.state_dict(), "{}{}".format(load_model, i))

        _, _, fscore = evaluate(model, word2id)
        if fscore > best_fscore:
            best_fscore = fscore
            best_epoch = i
        print('best epoch {} with fscore {:.6f}'.format(best_epoch, best_fscore))

        print("time: {}".format(time.time() - start))

    torch.save(model.state_dict(), load_model)


def main():
    if len(sys.argv) != 2:
        print('No arguments!!')
        exit()
    if sys.argv[1] == 'train':
        train()
    else:
        print('Miss type!!')

if __name__ == '__main__':
    main()
