import chainer.links as L
from chainer import functions as F
from chainer import optimizers as O
from chainer import Chain, Variable , cuda, serializers

class BLSTM_wf_lstm(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, extra_hidden_size, feat_length):
        super(BLSTM_wf_lstm, self).__init__(
            x2e = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            e2h_for = L.LSTM(embed_size + feat_length, hidden_size),
            e2h_back = L.LSTM(embed_size + feat_length, hidden_size),
            h2s = L.Linear(hidden_size*2, extra_hidden_size),
            s2o = L.Linear(extra_hidden_size, output_size)
        )

    def __call__(self, x, feats):
        self._reset_state()
        e_states = []
        h_back_states = []
        h_states = []
        o_states = []

        e_states = [F.relu(self.x2e(w)) for w in x]
        e_states = [F.concat(j) for j in zip(e_states, feats)]

        for e in e_states[::-1]:
            h_back = self.e2h_back(e)
            h_back_states.insert(0, h_back)
        for e, h_back in zip(e_states, h_back_states):
            h_for = self.e2h_for(e)
            h_states.append(F.concat((h_for, h_back)))
        o_states = [self.s2o(F.relu(self.h2s(h))) for h in h_states]
        return o_states

    def _reset_state(self):
        self.zerograds()
        self.e2h_for.reset_state()
        self.e2h_back.reset_state()

    def initialize_embed(self, word2vec_model, word_list, word2id, id2word):
        for i in range(len(word_list)):
            word = word_list[i]
            if word in word2vec_model:
                self.x2e.W.data[i+2] = word2vec_model[word]

