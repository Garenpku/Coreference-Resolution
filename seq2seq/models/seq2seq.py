import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import copy
import seq2seq
from seq2seq.util.print_state import print_state
from seq2seq.util.conceptnet_util import ConceptNet
from seq2seq.util.dataset_util import *


class MLP(nn.Module):
    def __init__(self, num_layers, input_size, output_size, mid_size=128, activation=torch.tanh):
        super(MLP, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        self.activation = activation
        for i in range(num_layers):
            if num_layers == 1:
                self.layers.append(self.to_cuda(torch.nn.Linear(input_size, output_size)))
                break
            if i == 0:
                self.layers.append(self.to_cuda(torch.nn.Linear(input_size, mid_size)))
            elif i == num_layers - 1:
                self.layers.append(self.to_cuda(torch.nn.Linear(mid_size, output_size)))
            else:
                self.layers.append(self.to_cuda(torch.nn.Linear(mid_size, mid_size)))

    def to_cuda(self, x):
        if torch.cuda.is_available():
            return x.cuda()
        else:
            return x

    def forward(self, input_tensor):
        tmp = input_tensor
        for i in range(self.num_layers):
            tmp = self.activation(self.layers[i](tmp))
        return tmp


class Planner(nn.Module):
    def __init__(self, num_layers, num_topics, input_size, output_size):
        super(Planner, self).__init__()
        self.MLP_list = []
        self.num_topics = num_topics
        for i in range(num_topics):
            tmp = MLP(num_layers, input_size, output_size)
            if torch.cuda.is_available():
                tmp = tmp.cuda()
            self.MLP_list.append(tmp)
            for param in tmp.parameters():
                param.data.uniform_(-0.08, 0.08)

    def forward(self, input_tensor):
        output = []
        for i in range(self.num_topics):
            output.append(self.MLP_list[i](input_tensor))
        return output


class Seq2seq(nn.Module):
    """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.

    Args:
        encoder (EncoderRNN): object of EncoderRNN
        decoder (DecoderRNN): object of DecoderRNN
        decode_function (func, optional): function to generate symbols from output hidden states (default: F.log_softmax)

    Inputs: input_variable, input_lengths, target_variable, teacher_forcing_ratio
        - **input_variable** (list, option): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the encoder.
        - **input_lengths** (list of int, optional): A list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
        - **target_variable** (list, optional): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs. This information is forwarded to the decoder.
        - **teacher_forcing_ratio** (int, optional): The probability that teacher forcing will be used. A random number
          is drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0)

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (batch): batch-length list of tensors with size (max_length, hidden_size) containing the
          outputs of the decoder.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs, *KEY_INPUT* : target outputs if provided for decoding, *KEY_ATTN_SCORE* : list of
          sequences, where each list is of attention weights }.

    """

    def __init__(self, encoder, decoder, dialog_encoder=None, speaker_encoder=None, decode_function=F.log_softmax,
                 cpt_vocab=None, filter_vocab=None, dep_corpus=None, stopword=None,
                 hidden_size=128, concept_level='simple', mid_size=64, dialog_hidden=128, conceptnet_file=None,
                 num_topics=1):
        super(Seq2seq, self).__init__()
        self.concept_level = concept_level
        # if conceptnet_file:
        #    self.concept_net = ConceptNet(conceptnet_file)
        self.encoder = encoder
        self.decoder = decoder
        self.dialog_encoder = dialog_encoder
        self.speaker_encoder = speaker_encoder
        self.decode_function = decode_function
        self.cpt_vocab = cpt_vocab
        if self.cpt_vocab:
            self.cpt_embedding = nn.Embedding(len(cpt_vocab.itos), hidden_size)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.decoder_input_MLP = torch.nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.hidden = hidden_size
        self.state_state_match = torch.nn.Linear(hidden_size * 4, hidden_size * 2, bias=False)
        self.state_context_match = torch.nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.context_state_match = torch.nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.planner = Planner(num_layers=2, num_topics=num_topics, input_size=hidden_size * 3,
                               output_size=1)
        for param in self.planner.parameters():
            param.data.uniform_(-0.08, 0.08)
        self.planner = self.to_cuda(self.planner)
        self.summary_transformer = torch.nn.Linear(hidden_size * 4, hidden_size)
        self.topic_to_description = torch.nn.Linear(hidden_size * 3, hidden_size * 3)
        self.reconstruct = torch.nn.Linear(hidden_size * 3, self.encoder.vocab_size)
        self.state_choose = torch.nn.Linear(hidden_size * 9, 3)
        self.num_topics = num_topics
        if filter_vocab:
            self.filter_vocab = filter_vocab
        if conceptnet_file:
            self.cn = ConceptNet(conceptnet_file, self.filter_vocab)
        self.stopword = stopword
        self.test = Variable(torch.ones(1, 3), requires_grad=True)

    def to_cuda(self, v):
        if torch.cuda.is_available():
            return v.cuda()
        return v

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def extract_per_utt(self, input_variable, encoder_outputs, eou_index):
        input_index = input_variable.numpy() if not torch.cuda.is_available() else input_variable.cpu().numpy()
        eou_pos = [np.where(line == eou_index)[0] for line in input_index]
        utt_hidden = [torch.cat([encoder_outputs[j][i].unsqueeze(0) for i in eou_pos[j]], 0) for j in
                      range(input_variable.shape[0])]
        max_num_utt = max([len(line) for line in utt_hidden])
        for i in range(input_variable.shape[0]):
            if torch.cuda.is_available():
                utt_hidden[i] = torch.cat(
                    [utt_hidden[i], torch.zeros([max_num_utt - len(utt_hidden[i]), len(utt_hidden[0][0])]).cuda()])
            else:
                utt_hidden[i] = torch.cat(
                    [utt_hidden[i], torch.zeros([max_num_utt - len(utt_hidden[i]), len(utt_hidden[0][0])])])
        utt_hidden = [line.unsqueeze(0) for line in utt_hidden]
        return torch.cat(utt_hidden, 0), eou_pos

    def utterance_extract(self, input_variable, eou_index, pad_index, src_vocab):
        input_index = input_variable.numpy() if not torch.cuda.is_available() else input_variable.cpu().numpy()
        eou_pos = [np.where(line == eou_index)[0] for line in input_index]
        input_index = [[int(num) for num in line] for line in input_index]
        batch_size = len(eou_pos)
        utterance_batch = []
        for i in range(batch_size):
            utterances_sample = []
            last_pos = 0
            for index in list(eou_pos[i]):
                utterances_sample.append(input_index[i][last_pos:index])
                last_pos = index + 1
            utterance_batch.append(utterances_sample)

        utterance_surface = [[[src_vocab.itos[word] for word in line] for line in sample] for sample in utterance_batch]

        utterance_one_speaker = []
        utterance_another_speaker = []
        for i in range(batch_size):
            if len(utterance_one_speaker) < i + 1:
                utterance_one_speaker.append([])
            if len(utterance_another_speaker) < i + 1:
                utterance_another_speaker.append([])
            if len(utterance_batch[i]) % 2 == 0:
                for j in range(0, len(utterance_batch[i]), 2):
                    utterance_one_speaker[i].append(copy.deepcopy(utterance_batch[i][j]))
                for j in range(1, len(utterance_batch[i]), 2):
                    utterance_another_speaker[i].append(copy.deepcopy(utterance_batch[i][j]))
            else:
                for j in range(1, len(utterance_batch[i]), 2):
                    utterance_one_speaker[i].append(copy.deepcopy(utterance_batch[i][j]))
                for j in range(0, len(utterance_batch[i]), 2):
                    utterance_another_speaker[i].append(copy.deepcopy(utterance_batch[i][j]))

        valid_length = [[len(line) for line in sample] for sample in utterance_batch]
        max_sentence = max([len(line) for line in utterance_batch])
        for i in range(batch_size):
            if len(utterance_batch[i]) < max_sentence:
                for k in range(max_sentence - len(utterance_batch[i])):
                    utterance_batch[i].append([pad_index])
        for i in range(max_sentence):
            max_token = 0
            for j in range(batch_size):
                if len(utterance_batch[j][i]) > max_token:
                    max_token = len(utterance_batch[j][i])
            for j in range(batch_size):
                if len(utterance_batch[j][i]) < max_token:
                    utterance_batch[j][i].extend((max_token - len(utterance_batch[j][i])) * [pad_index])
        input_by_utterance = []
        for i in range(max_sentence):
            input_utterance = []
            for j in range(batch_size):
                input_utterance.append(utterance_batch[j][i])
            input_by_utterance.append(torch.tensor(input_utterance))
        if torch.cuda.is_available():
            input_by_utterance = [tensor.cuda() for tensor in input_by_utterance]
        return input_by_utterance, valid_length, utterance_one_speaker, utterance_another_speaker, utterance_surface

    # input: a list of tensors with shape [*, embedding_size]
    def embedding_padding(self, embedding):
        res = []
        max_len = max([len(line) for line in embedding])
        vector_length = embedding[0].shape[-1]
        zero = torch.zeros((1, vector_length))
        if torch.cuda.is_available():
            zero = zero.cuda()
        for i in range(len(embedding)):
            if len(embedding[i]) < max_len:
                res.append(torch.cat([embedding[i], torch.cat((max_len - len(embedding[i])) * [zero])]).unsqueeze(0))
            else:
                res.append(embedding[i].unsqueeze(0))
        return torch.cat(res)

    # return size: batch * num_sentence * num_concept_per_sentence * embedding
    def concept_mapping(self, concept, response, vocab, src_vocab, related_dep=None):
        """
        batch_size = len(concept)
        pad_index = vocab.stoi['<pad>']
        eou_index = vocab.stoi['<eou>']
        expand_index = vocab.stoi['<expand>']
        np_concept = concept.numpy() if not torch.cuda.is_available() else concept.cpu().numpy()
        end_pos = []
        for line in np_concept:
            pos = np.where(line == pad_index)[0]
            if len(pos):
                end_pos.append(pos[0])
            else:
                end_pos.append(len(line))
        np_concept = [np_concept[i][:end_pos[i]] for i in range(len(np_concept))]

        expanded_batch = []
        # indexes_batch = []
        concepts_batch = []
        for line in np_concept:
            concepts_batch.append(line[:np.where(line == expand_index)[0][0]])
            expanded_batch.append(line[np.where(line == expand_index)[0][0] + 1:])

        # newly added: we use source vocabulary rather than concept vocabulary
        concepts_test = [[vocab.itos[word] for word in line] for line in concepts_batch]
        expanded_test = [[vocab.itos[word] for word in line] for line in expanded_batch]
        concepts_batch_for_embedding = [[src_vocab.stoi[word] for word in line] for line in concepts_test]
        # expanded_batch_for_embedding = [[src_vocab.stoi[word] for word in line] for line in expanded_test]

        unk = src_vocab.stoi['<unk>']
        expanded = [[src_vocab.stoi[word] for word in line if src_vocab.stoi[word] != unk] for line in expanded_test]

        concept_batch = []
        concept_raw_vocab = []
        embedding_batch = []
        embedding_expand = []
        for i in range(len(concepts_batch)):
            concept_d = []
            concept_d_for_embedding = []
            utt_pos = np.where(concepts_batch[i] == eou_index)[0]
            utt_pos = np.concatenate([[-1], utt_pos])
            for j in range(1, len(utt_pos)):
                concept_d.append(concepts_batch[i][utt_pos[j - 1] + 1:utt_pos[j]])
                concept_d_for_embedding.append(concepts_batch_for_embedding[i][utt_pos[j - 1] + 1:utt_pos[j]])
            if torch.cuda.is_available():
                embedding_expand.append(self.encoder.embedding(torch.LongTensor(expanded[i]).cuda()))
                concept_mapped = [self.encoder.embedding(torch.LongTensor(line).cuda()) for line in
                                  concept_d_for_embedding]
            else:
                embedding_expand.append(self.encoder.embedding(torch.LongTensor(expanded[i])))
                concept_mapped = [self.encoder.embedding(torch.LongTensor(line)) for line in concept_d_for_embedding]
            concept_batch.append(concept_d)
            embedding_batch.append(concept_mapped)
            concept_raw_vocab.append(concept_d_for_embedding)
        embedding_expand = self.embedding_padding(embedding_expand)
        """
        batch_size = len(concept)
        concept_raw_vocab = concept

        #dep_one_speaker = []
        concept_one_speaker = []
        concept_another_speaker = []
        for i in range(batch_size):
            if len(concept_one_speaker) < i + 1:
                concept_one_speaker.append([])
                #dep_one_speaker.append([])
            if len(concept_another_speaker) < i + 1:
                concept_another_speaker.append([])
            if len(concept_raw_vocab[i]) % 2 == 0:
                for j in range(0, len(concept_raw_vocab[i]), 2):
                    concept_one_speaker[i].append([word for word in concept_raw_vocab[i][j] if word != 0])
                    #if j < len(related_dep[i]):
                        #dep_one_speaker[i].append(related_dep[i][j])
                for j in range(1, len(concept_raw_vocab[i]), 2):
                    concept_another_speaker[i].append([word for word in concept_raw_vocab[i][j] if word != 0])
            else:
                for j in range(1, len(concept_raw_vocab[i]), 2):
                    concept_one_speaker[i].append([word for word in concept_raw_vocab[i][j] if word != 0])
                    #if j < len(related_dep[i]):
                        #dep_one_speaker[i].append(related_dep[i][j])
                for j in range(0, len(concept_raw_vocab[i]), 2):
                    concept_another_speaker[i].append([word for word in concept_raw_vocab[i][j] if word != 0])
        embedding_one_speaker = [[self.encoder.embedding(self.to_cuda(torch.LongTensor(line))) for line in sample] for
                                 sample in concept_one_speaker]
        embedding_another_speaker = [[self.encoder.embedding(self.to_cuda(torch.LongTensor(line))) for line in sample]
                                     for sample in concept_another_speaker]
        embedding_response = [self.encoder.embedding(self.to_cuda(torch.LongTensor(sample))) for sample in response]

        return concept_one_speaker, embedding_one_speaker, concept_another_speaker, embedding_another_speaker, response, embedding_response

    def state_track(self, utterance_one_speaker, concept_one_speaker, hidden_one_speaker, relevant_context,
                    context_one_speaker, embedding):
        concept_one_speaker = copy.deepcopy(concept_one_speaker)
        state = []
        all_embedding = []
        batch_size = len(utterance_one_speaker)
        max_steps = max([len(line) for line in hidden_one_speaker])

        for i in range(len(concept_one_speaker)):
            if len(concept_one_speaker[i]) < max_steps:
                concept_one_speaker[i].extend((max_steps - len(concept_one_speaker[i])) * [[]])

        zero = torch.zeros_like(hidden_one_speaker[0][0][0])
        zero_embedding = self.to_cuda(torch.zeros(self.hidden))
        if torch.cuda.is_available():
            zero = zero.cuda()
        for i in range(max_steps):
            context = relevant_context[:, i].unsqueeze(1)
            if i != 0:
                context_speaker = context_one_speaker[:, i - 1].unsqueeze(1)
            else:
                context_speaker = self.to_cuda(torch.zeros_like(context_one_speaker[:, 0]).unsqueeze(1))
            hidden = []
            hidden_concept = []
            embedding_concept = []
            concept_pos = []
            for j in range(batch_size):
                if len(hidden_one_speaker[j]) > i:
                    hidden.append(hidden_one_speaker[j][i])
                    concept_pos.append([utterance_one_speaker[j][i].index(word) for word in concept_one_speaker[j][i]])
                    if len(concept_pos[-1]):
                        hidden_concept.append(
                            torch.cat([hidden_one_speaker[j][i][pos].unsqueeze(0) for pos in concept_pos[-1]]))
                        embedding_concept.append(embedding[j][i])
                    else:
                        hidden_concept.append(zero.unsqueeze(0))
                        embedding_concept.append(zero_embedding.unsqueeze(0))
                else:
                    embedding_concept.append(zero_embedding.unsqueeze(0))
                    hidden.append(zero.unsqueeze(0))
                    hidden_concept.append(zero.unsqueeze(0))
                    concept_pos.append([])
            hidden = self.embedding_padding(hidden)
            hidden_concept = self.embedding_padding(hidden_concept)
            embedding_concept = self.embedding_padding(embedding_concept)
            context = torch.cat([torch.cat([line] * hidden.shape[1], dim=0).unsqueeze(0) for line in context], dim=0)
            context_speaker = torch.cat(
                [torch.cat([line] * hidden.shape[1], dim=0).unsqueeze(0) for line in context_speaker], dim=0)
            key = self.state_state_match(torch.cat([hidden, context, context_speaker], dim=-1))
            score = torch.bmm(hidden_concept, torch.transpose(key, 1, 2))
            score = self.softmax(score)
            value = torch.bmm(score, hidden)
            value = torch.cat([value, embedding_concept], dim=-1)
            if len(state):
                # state = torch.cat([state, value], dim=1)
                for k in range(len(value)):
                    if len(concept_one_speaker[k][i]):
                        state[k] = torch.cat([state[k], value[k][:len(concept_one_speaker[k][i])]])
                # all_embedding = torch.cat([all_embedding, embedding_concept])
            else:
                state = [line[:len(concept_one_speaker[k][i])] for k, line in enumerate(value)]
                # all_embedding = embedding_concept

        return state

    def state_to_distribution(self, state, dialog_one_speaker, dialog_another_speaker):
        mask = [len(line) * [1] for line in state]
        max_states = max([len(line) for line in mask])
        for i in range(len(mask)):
            if len(mask[i]) == 0:
                mask[i].extend(max_states * [0])
            elif len(mask[i]) < max_states:
                mask[i].extend((max_states - len(mask[i])) * [0])
        mask = self.to_cuda(torch.Tensor(mask) * 10)
        state = self.embedding_padding(state)
        key_state = self.state_context_match(state)
        transformed_context = self.context_state_match(
            torch.cat([dialog_another_speaker[:, -1], dialog_one_speaker[:, -1]], dim=-1))
        score = torch.bmm(key_state, transformed_context.unsqueeze(-1))
        score = torch.reshape(score, (score.shape[0], score.shape[1])) + mask
        distribution = self.softmax(score)
        summary = torch.bmm(distribution.unsqueeze(1), state).reshape(len(key_state), -1)
        return distribution, summary

    def language_planning(self, planning, distribution_state, context_one, context_another, src_vocab, num_expansion,
                          linearize):
        max_states = max([len(line) for line in planning])
        if max_states == 0:
            return self.to_cuda(torch.zeros(len(planning), self.encoder.vocab_size)), []
        #print("Max states in planning: ", max_states)
        #print("Shape of state: ", distribution_state.shape)
        for i in range(len(planning)):
            if len(planning[i]) < max_states:
                planning[i].extend((max_states - len(planning[i])) * [num_expansion * ['<pad>']])
            for j in range(len(planning[i])):
                if len(planning[i][j]) < num_expansion:
                    planning[i][j].extend((num_expansion - len(planning[i][j])) * ['<pad>'])
        planning = [[[src_vocab.stoi[word] for word in cpt] for cpt in line] for line in planning]
        mask = self.to_cuda(10 * torch.Tensor([[[int(word != src_vocab.stoi['<pad>']) for word in cpt] for cpt in line] for line in planning]))
        planning_linear = [linearize(line) for line in planning]
        embedding_planning = self.encoder.embedding(self.to_cuda(torch.LongTensor(planning)))
        tmp_u = torch.cat(
            [torch.cat([context_one.unsqueeze(1)] * num_expansion, dim=1).unsqueeze(1)] * max_states, dim=1)
        tmp_v = torch.cat(
            [torch.cat([context_another.unsqueeze(1)] * num_expansion, dim=1).unsqueeze(1)] * max_states, dim=1)
        embedding_planning = torch.cat([embedding_planning, tmp_u, tmp_v], dim=-1)
        #print("Shape of Embedding: ", embedding_planning.shape)
        planned_result = [topic.reshape(topic.shape[0], topic.shape[1], topic.shape[2]) + mask for topic in self.planner(embedding_planning)]
        planned_result = [self.softmax(topic) for topic in planned_result]
        planned_result = [distribution_state.unsqueeze(-1) * topic for topic in planned_result]
        planning_vocab = self.to_cuda(torch.zeros((len(planning), max_states, len(src_vocab))))
        #print("Shape of planning vocab: ", planning_vocab.shape)
        #print("Shape of planning result: ", planned_result[0].shape)
        planned_result = [planning_vocab.scatter(2, self.to_cuda(torch.tensor(planning)), topic) for topic in
                          planned_result]
        planned_result = [torch.bmm(self.to_cuda(torch.ones((len(planning), 1, max_states))), topic).squeeze() for
                          topic in planned_result]
        #for i in range(self.num_topics):
            #planned_result[i] = planned_result[i] * mask
        return planned_result[0], planning_linear

    def forward(self, input_variable, input_lengths=None, target_variable=None, dep_corpus=None,
                teacher_forcing_ratio=0, concept=None, vocabs=None, use_concept=False, track_state=False, sample_index=None):
        src_vocab = vocabs.src_vocab
        tgt_vocab = vocabs.tgt_vocab
        if use_concept:
            cpt_vocab = vocabs.cpt_vocab
            eou_index = src_vocab.stoi['<eou>']
            pad_index = src_vocab.stoi['<pad>']
            related_dep = [dep_corpus[index] for index in sample_index]

            # encode the dialog utterance by utterance
            input_by_utterance, valid_length, utterance_one_speaker, utterance_another_speaker, utterance_surface = self.utterance_extract(
                input_variable, eou_index, pad_index, src_vocab)

            cpt_per_utt, cpt_linear = get_concepts(utterance_surface, self.cn, self.stopword)
            state_corpus, state_linear = filter_relate(cpt_per_utt, self.cn)
            state_id = [[[src_vocab.stoi[word] for word in line] for line in sample] for sample in state_corpus]
            response = [[tgt_vocab.itos[word] for word in sample] for sample in target_variable]
            response = [[word for word in sample[1:sample.index('<eos>')] if word in self.cn.cpt_dict] for sample in response]
            response_id = [[src_vocab.stoi[word] for word in sample] for sample in response]

            encoder_hidden_dialog = []
            all_encoder_hiddens = []
            encoder_hidden = []
            hidden_one_speaker = []
            hidden_another_speaker = []
            last_hidden = []
            last_outputs = []
            for i in range(len(utterance_one_speaker)):
                hidden_one_speaker.append([])
                hidden_another_speaker.append([])
                last_hidden.append([])
            for k in range(len(input_by_utterance)):
                """
                if len(encoder_hidden):
                    encoder_outputs, encoder_hidden = self.encoder(input_by_utterance[k], encoder_hidden[0])
                else:
                """
                encoder_outputs, encoder_hidden = self.encoder(input_by_utterance[k])
                for l in range(len(utterance_one_speaker)):
                    if (k % 2 == len(valid_length[l]) % 2) and k < len(valid_length[l]):
                        hidden_one_speaker[l].append(encoder_outputs[l])
                    elif k < len(valid_length[l]):
                        hidden_another_speaker[l].append(encoder_outputs[l])
                    if k == len(valid_length[l]) - 1:
                        assert not last_hidden[l]
                        last_hidden[l] = torch.cat([encoder_hidden[0], encoder_hidden[1]], dim=-1)[l]
                encoder_hidden_dialog.append(torch.cat((encoder_hidden[0], encoder_hidden[1]), 1).unsqueeze(1))
                for i in range(len(valid_length)):
                    if len(valid_length[i]) < k + 1:
                        continue
                    if len(all_encoder_hiddens) < i + 1:
                        all_encoder_hiddens.append([encoder_outputs[i, :valid_length[i][k]]])
                    else:
                        all_encoder_hiddens[i].append(encoder_outputs[i, :valid_length[i][k]])
            last_hidden = torch.cat([line.unsqueeze(0) for line in last_hidden], dim=0)
            encoder_hidden_dialog = torch.cat(encoder_hidden_dialog, dim=1)
            all_encoder_hiddens = [torch.cat(sample) for sample in all_encoder_hiddens]
            zero = self.to_cuda(torch.zeros((1, all_encoder_hiddens[0].shape[-1])))
            max_len = max([len(line) for line in all_encoder_hiddens])
            for k in range(len(all_encoder_hiddens)):
                if len(all_encoder_hiddens[k]) < max_len:
                    all_encoder_hiddens[k] = torch.cat(
                        (all_encoder_hiddens[k], torch.cat((max_len - len(all_encoder_hiddens[k])) * [zero])))
            all_encoder_hiddens = [line.unsqueeze(0) for line in all_encoder_hiddens]
            all_encoder_hiddens = torch.cat(all_encoder_hiddens, 0)

            # encode the dialog in a parallel way
            """
            encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
            utt_hidden, eou_pos = self.extract_per_utt(input_variable, encoder_outputs, eou_index)
            """

            dialog_output, (context, _) = self.dialog_encoder(encoder_hidden_dialog)
            dialog_zero = self.to_cuda(torch.zeros_like(dialog_output[0][0]))
            relevant_context = []
            relevant_context_another = []
            relevant_utterance = []
            relevant_utterance_another = []
            for i in range(len(utterance_one_speaker)):
                relevant_context.append([])
                relevant_utterance.append([])
                relevant_context_another.append([])
                relevant_utterance_another.append([])
            for i in range(len(utterance_one_speaker)):
                if len(valid_length[i]) % 2 == 0:
                    relevant_context[i].append(dialog_zero.unsqueeze(0))
                for j in range((len(valid_length[i]) + 1) % 2, len(valid_length[i]) - 1, 2):
                    relevant_context[i].append(dialog_output[i][j].unsqueeze(0))
                for j in range(len(valid_length[i]) % 2, len(valid_length[i]), 2):
                    relevant_utterance[i].append(encoder_hidden_dialog[i][j].unsqueeze(0))
                relevant_context[i] = torch.cat(relevant_context[i], dim=0)
                relevant_utterance[i] = torch.cat(relevant_utterance[i], dim=0)
            relevant_utterance = self.embedding_padding(relevant_utterance)
            for i in range(len(utterance_another_speaker)):
                if len(valid_length[i]) % 2 == 1:
                    relevant_context_another[i].append(dialog_zero.unsqueeze(0))
                for j in range(len(valid_length[i]) % 2, len(valid_length[i]) - 1, 2):
                    relevant_context_another[i].append(dialog_output[i][j].unsqueeze(0))
                for j in range((len(valid_length[i]) + 1) % 2, len(valid_length[i]), 2):
                    relevant_utterance_another[i].append(encoder_hidden_dialog[i][j].unsqueeze(0))
                relevant_context_another[i] = torch.cat(relevant_context_another[i], dim=0)
                relevant_utterance_another[i] = torch.cat(relevant_utterance_another[i], dim=0)
            relevant_utterance_another = self.embedding_padding(relevant_utterance_another)
            dialog_one_speaker, (context_speaker, _) = self.speaker_encoder(relevant_utterance)
            dialog_another_speaker, (context_another_speaker, _) = self.speaker_encoder(relevant_utterance_another)

            for i in range(len(utterance_one_speaker)):
                assert len(utterance_one_speaker[i]) == len(relevant_context[i])
                assert len(utterance_one_speaker[i]) == len(hidden_one_speaker[i])
                assert len(utterance_another_speaker[i]) == len(hidden_another_speaker[i])
                assert len(utterance_another_speaker[i]) == len(relevant_context_another[i])

            relevant_context = self.embedding_padding(relevant_context)
            relevant_context_another = self.embedding_padding(relevant_context_another)

            concept_one_speaker, embedding_one_speaker, concept_another_speaker, embedding_another_speaker, expanded, expanded_embedding = self.concept_mapping(
                state_id, response_id, cpt_vocab, src_vocab, related_dep)

            for i in range(len(utterance_one_speaker)):
                assert len(utterance_one_speaker[i]) == len(concept_one_speaker[i])
                assert len(utterance_another_speaker[i]) == len(concept_another_speaker[i])

            def linearize(line):
                res = []
                for ut in line:
                    res.extend(ut)
                return res
            num_expansion = 40
            one_last = [line[-1] for line in concept_one_speaker]
            one_last_str = [[src_vocab.itos[word] for word in line] for line in one_last]
            another_last = [line[-1] for line in concept_another_speaker]
            another_last_str = [[src_vocab.itos[word] for word in line] for line in another_last]
            planning_u = [self.cn.expand_list(line, self.filter_vocab, num_expansion, self.stopword) for line in one_last_str]
            planning_v = [self.cn.expand_list(line, self.filter_vocab, num_expansion, self.stopword) for line in another_last_str]
            concept_one_linear = [linearize(line[:-1]) for line in concept_one_speaker]
            concept_one_linear_str = [[src_vocab.itos[word] for word in line] for line in concept_one_linear]
            planning = [self.cn.expand_list(line, self.filter_vocab, num_expansion, self.stopword) for line in concept_one_linear_str]
            # planning = [planning[i] + planning_another[i] for i in range(len(planning))]

            state = self.state_track(utterance_one_speaker, concept_one_speaker, hidden_one_speaker, relevant_context,
                                     dialog_one_speaker, embedding_one_speaker)
            state_another = self.state_track(utterance_another_speaker, concept_another_speaker, hidden_another_speaker,
                                             relevant_context_another, dialog_another_speaker,
                                             embedding_another_speaker)
            state_v = [state_another[k][len(state_another[k])-len(another_last[k]):] for k in range(len(state))]
            state_u = [state[k][len(state[k])-len(one_last[k]):] for k in range(len(state))]
            state_s = [state[k][:len(state[k])-len(one_last[k])] for k in range(len(state))]

            distribution_state, summary = self.state_to_distribution(state_s, dialog_one_speaker, dialog_another_speaker)
            distribution_u, summary_u = self.state_to_distribution(state_u, dialog_one_speaker, dialog_another_speaker)
            distribution_v, summary_v = self.state_to_distribution(state_v, dialog_one_speaker, dialog_another_speaker)

            #tmp = torch.cat([torch.ones_like(state_u[0][0].unsqueeze(0))] * 32, dim=0)
            tmp = torch.ones([32, self.hidden * 9])
            #p_state = self.state_choose(tmp)
            #p_state = self.state_choose(torch.cat([summary, summary_u, summary_v], dim=-1))
            #p_state = torch.ones([32, 3]) + self.test
            #p_state = torch.softmax(p_state.squeeze(), -1)
            p_state = self.test

            # reconstruction supervision
            """
            reconstruct_result = self.softmax(self.reconstruct(self.embedding_padding(state)))
            dep_by_order = []
            for i in range(len(dep_one_speaker)):
                dep_d = []
                for j in range(len(dep_one_speaker[i])):
                    dep_d.extend([dep_one_speaker[i][j][src_vocab.itos[word]] for word in concept_one_speaker[i][j]])
                dep_by_order.append(dep_d)
            """

            planned_result_s, planning_linear = self.language_planning(planning, distribution_state,
                                                                     dialog_one_speaker[:, -1],
                                                                     dialog_another_speaker[:, -1], src_vocab,
                                                                     num_expansion, linearize)
            planned_result_u, planning_linear_u = self.language_planning(planning_u, distribution_u,
                                                                         dialog_one_speaker[:, -1],
                                                                         dialog_another_speaker[:, -1], src_vocab,
                                                                         num_expansion, linearize)
            planned_result_v, planning_linear_v = self.language_planning(planning_v, distribution_v,
                                                                         dialog_one_speaker[:, -1],
                                                                         dialog_another_speaker[:, -1], src_vocab,
                                                                         num_expansion, linearize)
            #planning_linear = [planning_linear[i] + planning_linear_u[i] + planning_linear_v[i] for i in range(len(planning_linear))]
            #p_state = 1/3 * torch.ones_like(p_state)
            #planned_result = [torch.bmm(p_state.unsqueeze(1), torch.cat([planned_result_s.unsqueeze(1), planned_result_u.unsqueeze(1), planned_result_v.unsqueeze(1)], dim=1)).squeeze()]
            planned_result = [p_state[:, 0].unsqueeze(1) * planned_result_s + p_state[:, 1].unsqueeze(1) * planned_result_u + p_state[:, 2].unsqueeze(1) * planned_result_v]
            planned_result = [p_state[:, 0].unsqueeze(1) * torch.ones_like(planned_result_s) / planned_result_s.shape[1]]

            debug_prob = [topic.cpu().detach().numpy() for topic in planned_result]

            """
            intermediate = self.planner(torch.cat([summary, summary_another], dim=-1))

            planned_result = [self.decoder.out(topic) for topic in intermediate]
            planned_result = [self.softmax(topic) for topic in planned_result]

            # 以下是带linear K-expansion的代码
            max_len = max([len(batch) for batch in expanded])
            mask = [self.to_cuda(torch.zeros(len(batch))) for batch in expanded]
            for i in range(len(mask)):
                if len(mask[i]) < max_len:
                    mask[i] = torch.cat([mask[i], self.to_cuda(1000 * torch.ones(max_len - len(mask[i])))]).unsqueeze(0)
                else:
                    mask[i] = mask[i].unsqueeze(0)
            mask = torch.cat(mask, dim=0)
            tmp = torch.cat([torch.cat([summary, context.squeeze()], dim=-1).unsqueeze(1)] * expanded_embedding.shape[1], dim=1)
            intermediate = torch.cat([tmp, expanded_embedding], dim=-1)
            planned_result = self.planner(intermediate)
            planned_result = [topic.squeeze() - mask for topic in planned_result]
            # 注释到这里
            """


            top = [topic.topk(100, dim=1) for topic in planned_result]
            top_words = [topic[1].cpu().detach().numpy() for topic in top]
            top_prob = [topic[0].cpu().detach().numpy() for topic in top]

            # print(top_words_for_print)

            top_words = [self.to_cuda(torch.LongTensor(topic)) for topic in top_words]
            top_embeddings = [self.encoder.embedding(topic) for topic in top_words]
            top_prob = [topic[0] / torch.sum(topic[0], dim=-1).unsqueeze(1) for topic in top]

            expanded_for_loss = expanded
            expanded_for_loss = [
                [word for word in line if word != src_vocab.stoi['<unk>'] and word in planning_linear[i]] for i, line
                in enumerate(expanded_for_loss)]

            planned_max = torch.cat([topic.topk(1, dim=1)[1] for topic in planned_result], dim=1).tolist()
            if teacher_forcing_ratio == 1:
                for k in range(len(planned_max)):
                    if len(expanded_for_loss[k]):
                        planned_max[k] = [expanded_for_loss[k][0]]
            planned_max = self.to_cuda(torch.LongTensor(planned_max))
            emb_max = self.encoder.embedding(planned_max).squeeze()
            description = self.topic_to_description(torch.cat([self.encoder.embedding(planned_max).squeeze(), dialog_one_speaker[:,-1], dialog_another_speaker[:,-1]], dim=-1))
            auxiliary_words = self.reconstruct(description)
            top_auxiliary_words = auxiliary_words.topk(50, dim=1)[1]
            words_to_copy = torch.cat([top_auxiliary_words, planned_max], dim=1)
            embedding_to_copy = self.encoder.embedding(words_to_copy)

            # choose loss
            words_list = words_to_copy.tolist()
            target_surface = [[tgt_vocab.itos[word] for word in line[1:]] for line in
                              target_variable.cpu().detach().numpy()]
            target_index = [[src_vocab.stoi[word] for word in line] for line in target_surface]
            choose_ground = [[int(word in words_list[i] and word != src_vocab.stoi['<pad>']) for word in line] for
                             i, line in enumerate(target_index)]
            choose_ground = self.to_cuda(torch.tensor(choose_ground))

            if track_state:
                top_words_for_print = [
                    [[src_vocab.itos[word] + '_' + str(top_prob[j][i][k]) for k, word in enumerate(batch[:5])] for
                     i, batch
                     in enumerate(list(topic))] for j, topic in enumerate(top_words)]
                res_all = [[top_words_for_print[j][i] for j in range(self.num_topics)] for i in
                           range(len(state))]
                for i in range(len(expanded_for_loss)):
                    for j in range(len(expanded_for_loss[i][:self.num_topics])):
                        res_all[i][j].append(src_vocab.itos[expanded_for_loss[i][j]] + '_GOLD-' + str(
                            debug_prob[j][i][expanded_for_loss[i][j]]))

            o = self.summary_transformer(torch.cat([description, emb_max], dim=1))
            last_state = dialog_output[:, -1, :].unsqueeze(0)

            # planning极端实验的选择
            last_state = torch.cat((last_state, o.unsqueeze(0)))
            # last_state = torch.cat((o.unsqueeze(0), self.to_cuda(torch.zeros_like(last_state))))
            result, all_choose_rates = self.decoder(inputs=target_variable,
                                                    encoder_hidden=last_state,
                                                    encoder_outputs=all_encoder_hiddens,
                                                    function=self.decode_function,
                                                    teacher_forcing_ratio=teacher_forcing_ratio,
                                                    batch_state=top_prob,
                                                    batch_concepts=[words_to_copy],
                                                    batch_embeddings=embedding_to_copy,
                                                    context=context.squeeze(),
                                                    src_vocab=src_vocab,
                                                    tgt_vocab=tgt_vocab,
                                                    use_concept=use_concept,
                                                    concept_rep=o.unsqueeze(1))
            # 暂不使用reconstruction
            reconstruct_result = []
            dep_by_order = []
            if not track_state:
                return result, (planned_result, expanded_for_loss), (all_choose_rates, choose_ground), (
                    reconstruct_result, dep_by_order)
            else:
                return result, (planned_result, expanded_for_loss), (all_choose_rates, choose_ground), (
                    reconstruct_result, dep_by_order), res_all
        else:
            encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
            context_zero = self.to_cuda(torch.zeros(len(encoder_outputs), self.hidden))
            result = self.decoder(inputs=target_variable,
                                  encoder_hidden=encoder_hidden,
                                  encoder_outputs=encoder_outputs,
                                  context=context_zero,
                                  function=self.decode_function,
                                  batch_state=[1],
                                  batch_concepts=[input_variable],
                                  batch_embeddings=encoder_outputs,
                                  teacher_forcing_ratio=teacher_forcing_ratio,
                                  src_vocab=src_vocab,
                                  tgt_vocab=tgt_vocab)
            return result
