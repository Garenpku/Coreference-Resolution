import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', bidirectional=False, use_concept=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, dialog_hidden=128, embedding=128):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        if use_concept:
            if use_attention:
                self.rnn = self.rnn_cell(hidden_size*2 + embedding, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
            else:
                self.rnn = self.rnn_cell(hidden_size + embedding, hidden_size, n_layers, batch_first=True,
                                         dropout=dropout_p)
        else:
            self.rnn = self.rnn_cell(hidden_size * 2, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.hidden_size = hidden_size

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.copy_c = nn.Linear(dialog_hidden, 64)
        if use_concept:
            self.copy_e = nn.Linear(embedding, 64)
        else:
            self.copy_e = nn.Linear(embedding * 2, 64)
        self.copy_o = nn.Linear(embedding, 64)
        self.copy_match = nn.Linear(128, 64)
        self.copy_h = nn.Linear(hidden_size, 64)
        self.copy_distribution = nn.Linear(64, 1)

        self.choose_gate = nn.Linear(self.hidden_size * 2, 1)

    def to_cuda(self, x):
        if torch.cuda.is_available():
            return x.cuda()
        return x

    def forward_step(self, input_var, hidden, encoder_outputs, function, mix=None, score_copy=None, use_concept=False,
                     concept_rep=None, use_copy=True):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if use_concept:
            tmp = self.to_cuda(torch.zeros_like(concept_rep))
            #decoder_input = torch.cat([embedded, tmp], dim=-1)
            decoder_input = torch.cat([mix, embedded, tmp], dim=-1)
        else:
            decoder_input = torch.cat([mix, embedded], dim=-1)
        output, hidden = self.rnn(decoder_input, hidden)


        attn = None
        if self.use_attention:
            output, attn, mix = self.attention(output, encoder_outputs)
            # 这里是为了planning的极端实验
            #mix = self.to_cuda(torch.zeros_like(mix))
        score_vocab = self.out(output.contiguous().view(-1, self.hidden_size))

        #debug
        debug = torch.sum(score_vocab, dim=-1).cpu().detach().numpy()
        #print("sum:\n", debug)

        #mean = score_vocab.mean(dim=1).unsqueeze(1)
        #std = score_vocab.std(dim=1).unsqueeze(1)

        #debug = std.cpu().detach().numpy()
        #print("std:\n", debug)

        #score_vocab = (score_vocab - mean) / std
        score_vocab = torch.softmax(score_vocab, dim=1)
        if use_copy:
            choose_rate = torch.sigmoid(self.choose_gate(torch.cat([hidden.squeeze(), embedded.squeeze()], dim=1)))
            debug = choose_rate.cpu().detach().numpy()
            #print("choose rate:\n", debug)
            final_score = score_copy * choose_rate + score_vocab * (1 - choose_rate)
            #final_score = score_vocab
        else:
            final_score = score_vocab
        # predicted_softmax = function(final_score, dim=1).view(batch_size, output_size, -1)
        #predicted_softmax = final_score / torch.sum(final_score, -1).view(-1, 1)
        predicted_softmax = final_score
        predicted_softmax = torch.log(predicted_softmax)
        if use_concept:
            return predicted_softmax, hidden, attn, mix, choose_rate
        else:
            return predicted_softmax, hidden, attn, mix

    def copy(self, context, decoder_state, dialog_state, concepts, embeddings, tgt_vocab, src_vocab, concept_rep=None):

        batch_size = len(decoder_state)
        res_c = self.copy_c(context)
        res_h = self.copy_h(decoder_state)
        #res_o = self.copy_o(concept_rep.squeeze())
        tmp = torch.cat([res_c, res_h], dim=-1)
        res_tmp = self.copy_match(tmp).unsqueeze(2)
        res_e = self.copy_e(embeddings)
        score = torch.bmm(res_e, res_tmp).reshape(batch_size, -1)

        #debug = torch.sum(score, dim=-1).cpu().detach().numpy()
        #print("sum of copy:\n", debug)

        mean = score.mean(dim=1).unsqueeze(1)
        std = score.std(dim=1).unsqueeze(1)

        #debug = std.cpu().detach().numpy()
        #print("std:\n", debug)

        score = (score - mean) / std

        copy_distribution = torch.softmax(score, dim=1)
        summary_step = torch.bmm(score.unsqueeze(1), embeddings).reshape((batch_size, -1))

        #word_distribution = [dialog_state[i] * copy_distribution[:, i].unsqueeze(1) for i in range(len(dialog_state))]
        word_distribution = [copy_distribution]

        score_copy_overall = self.to_cuda(torch.zeros((batch_size, self.output_size)))
        for k in range(len(dialog_state)):
            concepts_numpy = concepts[k].cpu().detach().numpy()
            mapped_concepts = []
            for i in range(len(concepts_numpy)):
                mapped_sent = []
                for j in range(len(concepts_numpy[i])):
                    cpt = src_vocab.itos[concepts_numpy[i][j]]
                    if cpt in tgt_vocab.stoi:
                        mapped_sent.append(tgt_vocab.stoi[cpt])
                    else:
                        mapped_sent.append(self.output_size)
                mapped_concepts.append(mapped_sent)
            mapped_concepts_tensor = torch.tensor(mapped_concepts)
            score_copy = torch.zeros((batch_size, self.output_size+1))
            if torch.cuda.is_available():
                mapped_concepts_tensor = mapped_concepts_tensor.cuda()
                score_copy = score_copy.cuda()
            score_copy = score_copy.scatter(1, mapped_concepts_tensor, word_distribution[k])
            score_copy = score_copy[:, :-1]
            if torch.cuda.is_available():
                score_copy = score_copy.cuda()
            score_copy_overall += score_copy
        return score_copy_overall, summary_step

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax,
                teacher_forcing_ratio=0, batch_state=None, batch_concepts=None, batch_embeddings=None, context=None,
                src_vocab=None, tgt_vocab=None, use_concept=False, concept_rep=None):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        if teacher_forcing_ratio == 0:
            max_length = max(50, max_length)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        all_choose_rates = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        if use_teacher_forcing:
            # decoder_input = inputs[:, :-1]
            for di in range(max_length):
                decoder_input = inputs[:, di].unsqueeze(1)
                if di == 0:
                    mix = torch.zeros(batch_size, 1, self.hidden_size)
                    if torch.cuda.is_available():
                        mix = mix.cuda()
                if use_concept:
                    score_copy, summary_step = self.copy(context, decoder_hidden.squeeze(), batch_state, batch_concepts,
                                           batch_embeddings, tgt_vocab, src_vocab, concept_rep)
                    #score_copy = torch.zeros([len(context), self.output_size])
                    if torch.cuda.is_available():
                        score_copy = score_copy.cuda()
                    decoder_output, decoder_hidden, step_attn, mix, choose_rate = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs, mix=mix,
                                                                                       function=function,
                                                                                       score_copy=score_copy,
                                                                                       use_concept=use_concept,
                                                                                       concept_rep=summary_step.unsqueeze(1))
                    all_choose_rates.append(choose_rate)
                else:
                    score_copy, _ = self.copy(context, decoder_hidden.squeeze(), batch_state, batch_concepts, batch_embeddings,
                                              tgt_vocab, src_vocab)
                    decoder_output, decoder_hidden, step_attn, mix = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs,
                                                                                       mix=mix,
                                                                                       function=function,
                                                                                       score_copy=score_copy)
                step_output = decoder_output.squeeze(1)
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                if di == 0:
                    mix = torch.zeros(batch_size, 1, self.hidden_size)
                if torch.cuda.is_available():
                    mix = mix.cuda()
                if use_concept:
                    score_copy, summary_step = self.copy(context, decoder_hidden.squeeze(), batch_state, batch_concepts,
                                           batch_embeddings, tgt_vocab, src_vocab, concept_rep)
                    #score_copy = torch.zeros([len(context), self.output_size])
                    if torch.cuda.is_available():
                        score_copy = score_copy.cuda()
                    decoder_output, decoder_hidden, step_attn, mix, choose_rate = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs,
                                                                                       function=function,
                                                                                       score_copy=score_copy,
                                                                                       use_concept=use_concept,
                                                                                       mix=mix,
                                                                                       concept_rep=summary_step.unsqueeze(1))
                    all_choose_rates.append(choose_rate)
                else:
                    score_copy, _ = self.copy(context, decoder_hidden.squeeze(), batch_state, batch_concepts, batch_embeddings,
                                              tgt_vocab, src_vocab)
                    decoder_output, decoder_hidden, step_attn, mix = self.forward_step(decoder_input, decoder_hidden,
                                                                                       encoder_outputs,
                                                                                       function=function,
                                                                                       mix=mix,
                                                                                       score_copy=score_copy)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols


        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        if use_concept:
            return (decoder_outputs, decoder_hidden, ret_dict) , all_choose_rates
        else:
            return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            if len(h) == 2:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            else:
                h = h
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
