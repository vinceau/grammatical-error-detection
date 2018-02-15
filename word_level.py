from __future__ import division

import string

import torch
from torch.autograd import Variable
from torch import nn
from torch import optim

import argparse
import logging
import numpy as np
import nltk



def index2sparse(index, size):
    i = torch.LongTensor([[0],[index]])
    v = torch.FloatTensor([1])
    return torch.sparse.FloatTensor(i, v, torch.Size([1, size]))


"""
Word level features:

* begins with capital letter or not (good for captilisation errors)
* ends with s (plural related errors)
* starts with a vowel (an, a)
* contains numbers
* contains symbols
* is only symbol (no alphanum)
"""

class WordSparse(object):

    @staticmethod
    def is_capital(word):
        if word[0].isupper():
            return 1
        return 0

    @staticmethod
    def trailing_s(word):
        if word[-1].lower() == 's':
            return 1
        return 0

    @staticmethod
    def starting_vowel(word):
        if word[0].lower() in 'aeiou':
            return 1
        return 0

    @staticmethod
    def contains_num(word):
        for l in word:
            if l.isdigit():
                return 1
        return 0

    @staticmethod
    def contains_sym(word):
        for l in word:
            if l in string.punctuation:
                return 1
        return 0

    @staticmethod
    def only_sym(word):
        for l in word:
            if l not in string.punctuation:
                return 0
        return 1

    @staticmethod
    def gen_sparse(r, c, vals):
        rows = []
        cols = []
        for i in range(r):
            rows.extend([i] * c)
            cols.extend(range(c))

        index = torch.LongTensor([rows, cols])
        values = torch.FloatTensor(vals)
        return torch.sparse.FloatTensor(index, values, torch.Size([r, c]))


    @staticmethod
    def word2feat(word):
        return [
            WordSparse.is_capital(word),
            WordSparse.trailing_s(word),
            WordSparse.starting_vowel(word),
            WordSparse.contains_num(word),
            WordSparse.contains_sym(word),
            WordSparse.only_sym(word)
        ]


class POSSparse(object):

    def __init__(self, window=2):
        # pad ends with zero
        self.pos2id = {
            'PAD': 0
        }
        self.posgram2id = {
            'UNKPOS': 0
        }
        self.window = window

    def sent2pos(self, sentence):
        """Converts a sentence to POS id
        """
        ret = []
        pos = nltk.pos_tag(sentence)
        for _, p in pos:
            if p not in self.pos2id:
                self.pos2id[p] = len(self.pos2id)
            ret.append(self.pos2id[p])
        return ret

    def sent2posgram(self, sentence, addunk=False):
        """Converts a sentence into POSgram id
        """
        ret = []
        ids = self.sent2pos(sentence)
        for i in range(len(sentence)):
            block = []
            for j in range(i - self.window, i + 1 + self.window):
                if j < 0 or j >= len(sentence):
                    block.append(0)
                else:
                    block.append(ids[j])
            pg = str(block)
            if pg not in self.posgram2id:
                if addunk:
                    self.posgram2id[pg] = len(self.posgram2id)
                else:
                    # we haven't seen this sequence of POS tags before so
                    # return the id of UNKPOS
                    pg = 'UNKPOS'
            ret.append(self.posgram2id[pg])

        return ret


class WordFeatures(object):

    def __init__(self):
        self.ps = POSSparse(1)
        self.char2id = {
            'UNKCHAR': 0
        }

    def posgram_length(self):
        return len(self.ps.posgram2id)

    def preprocess(self, all_sentences):
        for s in all_sentences:
            self.ps.sent2posgram(s, addunk=True)
            for w in s:
                for char in w:
                    if char not in self.char2id:
                        self.char2id[char] = len(self.char2id)

    def index2sparse(self, index):
        i = [[0], [index]]
        v = [1]
        return i, v

    def merge_sparse(self, size, sparse1, sparse2):
        i1, v1 = sparse1
        i2, v2 = sparse2
        new_i = [i1[0] + i2[0], i1[1] + [q + size for q in i2[1]]]
        return new_i, (v1 + v2)

    def sent2sparse_posgram(self, sentence):
        pg = self.ps.sent2posgram(sentence)
        #hcf = [WordSparse.word2feat(w) for w in sentence]
        total_posgram = len(self.ps.posgram2id)
        return [index2sparse(i, total_posgram) for i in pg]


    def batch2sparse_posgram(self, batch):
        res = []
        # batch pos gram
        bpg = [self.ps.sent2posgram(s) for s in batch]
        total_posgram = len(self.ps.posgram2id)

        batch_size = len(batch)
        for i in range(len(bpg[0])):  # sentence length
            indices = []
            for j, s in enumerate(bpg):  # batch length
                indices.append([j, s[i]])
            values = [1] * batch_size
            ii = torch.LongTensor(indices).t()
            vv = torch.FloatTensor(values)
            r = torch.sparse.FloatTensor(ii, vv, torch.Size([batch_size, total_posgram]))
            if torch.cuda.is_available():
                r = r.cuda()
            res.append(r)
        return res

"""
import json
with open('test.json', 'r') as f:
    sents = json.load(f)['data']

inp = sents[0]['input_sentence']
out = sents[0]['corrected_sentence']
ps = POSSparse()
pg = ps.sent2posgram(inp)
ws = WordSparse()
#for w in inp:
#    print(w)
#    print(ws.word2feat(w))
#print(pg)
wf = WordFeatures()

embed_size = 300
batch_size = 64
from functions import fill_batch
import generators as gens
train_txt = "fce-data/train.kaneko.txt"
gen1 = gens.word_list(train_txt)
gen2 = gens.batch(gens.sorted_parallel(gen1, embed_size*batch_size), batch_size)
batchs = [b for b in gen2]
bl = list(range(len(batchs)))

# preprocess the word features
all_sents = []
for n, j in enumerate(bl):
    batch = fill_batch([b[-1].split() for b in batchs[j]])
    all_sents.extend(batch)

wf.preprocess(all_sents)
res = wf.batch2sparse_posgram(batch)
"""
