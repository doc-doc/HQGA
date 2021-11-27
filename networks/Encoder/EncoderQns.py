import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch

class EncoderQns(nn.Module):
    def __init__(self, dim_embed, dim_hidden, vocab_size, glove_embed, use_bert=True,
                 input_dropout_p=0.2, rnn_dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        :param dim_embed:
        :param dim_hidden:
        :param vocab_size:
        :param input_dropout_p:
        :param rnn_dropout_p:
        :param n_layers:
        :param bidirectional:
        :param rnn_cell:
        """
        super(EncoderQns, self).__init__()
        self.dim_hidden = dim_hidden
        self.vocab_size = vocab_size
        self.glove_embed = glove_embed
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.use_bert = use_bert
        if self.use_bert:
            input_dim = 768
            self.embedding = nn.Linear(input_dim, dim_embed)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_embed)
            word_mat = torch.FloatTensor(np.load(self.glove_embed))
            self.embedding = nn.Embedding.from_pretrained(word_mat, freeze=False)

        self.rnn = self.rnn_cell(dim_embed, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)


    def forward(self, qns, qns_lengths, hidden=None):
        """
         encode question
        :param qns:
        :param qns_lengths:
        :return:
        """

        qns_embed = self.embedding(qns)
        qns_embed = self.input_dropout(qns_embed)
        packed = pack_padded_sequence(qns_embed, qns_lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden
