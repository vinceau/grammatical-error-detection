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

def precision_recall_f(pres, tags, cons):
    c_p = 0  # actual
    correct_p = 0  # predicted
    c_r = 0
    correct_r = 0
    _tags = np.array(tags, dtype=np.int64)
    tags = Variable(torch.from_numpy(_tags))
    if torch.cuda.is_available():
        tags = tags.cuda()

    # pres is the post-soft max probabilities
    # num is index, a is (pres, tag)
    pre_l = [pres[k].data.max(0)[1].cpu().numpy()[0] for k in range(len(pres)) if cons[k] == True]
    tag_l = [int(tags.data[n]) for n in range(len(tags.data)) if cons[n] == True]
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

    ev_sents = []
    ev_tags = []

    for ba in batchs:
        # for each batch
        tags = [[int(c) for c in a[:-1]] for a in ba]
        batch = fill_batch([b[-1].split() for b in ba])
        tags = fill_batch(tags, token=0)
        ev_sents.extend(batch)
        ev_tags.extend(tags)


    for sent, tags in zip(all_sents, all_tags):
        x = Variable(torch.LongTensor([word2id[word] if word in word2id else word2id['<unk>'] for word in sent]))
        if torch.cuda.is_available():
            x = x.cuda()

        cons = x.data != -1
        pres = model(x)
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

    def __init__(self, vocab_size, embed_size, hidden_size, output_size, extra_hidden_size):
        super(BiLSTMw2v, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.x2e = nn.Embedding(vocab_size, embed_size)
        self.h2s = nn.Linear(hidden_size*2, extra_hidden_size)
        self.s2o = nn.Linear(extra_hidden_size, output_size)

        self.blstm = nn.LSTM(embed_size, hidden_size, num_layers=1, bidirectional=True)
        self.relu = nn.ReLU()

        self.blstm_hidden = self._init_hidden(2)


    def _init_hidden(self, layers=1):
        h = Variable(torch.zeros(layers, 1, self.hidden_size))
        cell = Variable(torch.zeros(layers, 1, self.hidden_size))
        if torch.cuda.is_available():
            h = h.cuda()
            cell = cell.cuda()
        return (h, cell)

    def _reset_state(self):
        self.blstm_hidden = self._init_hidden(2)
        self.zero_grad()


    # do it per sentence for now
    def forward(self, x):
        self._reset_state()
        h_states = []
        o_states = []

        sent = self.x2e(x)
        for word in sent:
            inp = word.view(1, -1, self.embed_size)
            out, self.blstm_hidden = self.blstm(inp, self.blstm_hidden)
            out = out.view(-1)
            h_states.append(out)

        s = self.h2s(torch.stack(h_states))
        return self.s2o(self.relu(s))


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

train_txt = "fce-data/train.kaneko.txt"
dev_txt = "fce-data/dev.kaneko.txt"
test_txt = "fce-data/test.kaneko.txt"
vocab_dict = "model/ptBLSTMVocab.pkl"
load_model = "model/ptBLSTM.model0"
state_model = "model/ptBLSTM.sta"

vocab_size = take_len(train_txt)
batch_size = 64
embed_size = 300
output_size = 2
hidden_size = 200
extra_hidden_size = 50
epoch = 20
write_embeddings_to_file = False


random.seed(0)


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

best_epoch = -1
best_fscore = -1

softmax = nn.Softmax()
cross_entropy_loss = nn.CrossEntropyLoss()

total_loss = 0
gen1 = gens.word_list(train_txt)
gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
batchs = [b for b in gen2]
bl = list(range(len(batchs)))

all_sents = []
all_tags = []

for ba in batchs:
    # for each batch
    tags = [[int(c) for c in a[:-1]] for a in ba]
    batch = fill_batch([b[-1].split() for b in ba])
    tags = fill_batch(tags, token=0)
    all_sents.extend(batch)
    all_tags.extend(tags)


model = BiLSTMw2v(vocab_size, embed_size, hidden_size, output_size, extra_hidden_size)

if torch.cuda.is_available():
    model = model.cuda()


opt = optim.Adam(model.parameters(), lr=0.001)

for i in range(1, epoch+1):
    print('\nepoch no. {}'.format(i))
    random.shuffle(bl)
    epoch_total_loss = 0

    for sent, tags in zip(all_sents, all_tags):
        # for each batch
        model.zero_grad()
        model._reset_state()

        x = Variable(torch.LongTensor([word2id[word] if word in word2id else word2id['<unk>'] for word in sent]))

        if torch.cuda.is_available():
            x = x.cuda()

        pres = model(x)

        _tags = np.array(tags, dtype=np.int64)
        tags = Variable(torch.from_numpy(_tags))

        if torch.cuda.is_available():
            tags = tags.cuda()

        loss = cross_entropy_loss(pres, tags)
        loss.backward()

        opt.step()
        epoch_total_loss += loss.data[0]

    print('total loss for epoch {} is {}'.format(i, epoch_total_loss))

    _, _, fscore = evaluate(model, word2id)
    if fscore > best_fscore:
        best_fscore = fscore
        best_epoch = i

    print('best epoch {} with fscore {:.6f}'.format(best_epoch, best_fscore))


