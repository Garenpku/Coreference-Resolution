import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy
from . import torch_utils


def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def list_padding(ls, dim, pad_id="<pad>", dtype="ls"):
    ls = deepcopy(ls)
    if dim == 2:
        max_len = max([len(line) for line in ls])
        for i in range(len(ls)):
            if len(ls[i]) < max_len:
                ls[i].extend((max_len - len(ls[i])) * [pad_id])
    elif dim == 3:
        max_sent_len = max([max([len(line) for line in sample]) for sample in ls])
        max_discourse_len = max([len(line) for line in ls])
        for i in range(len(ls)):
            if len(ls[i]) < max_discourse_len:
                for _ in range(max_discourse_len - len(ls[i])):
                    if dtype == "ls":
                        ls[i].append([])
                    elif dtype == "np":
                        ls[i].append(np.zeros((max_sent_len, max_sent_len)))
            for j in range(len(ls[i])):
                if len(ls[i][j]) < max_sent_len:
                    if dtype == "ls":
                        ls[i][j].extend((max_sent_len - len(ls[i][j])) * [pad_id])
                    elif dtype == "np":
                        ls[i][j] = np.concatenate((ls[i][j], np.zeros((len(ls[i][j]), max_sent_len - len(ls[i][j])))), axis=-1)
                        ls[i][j] = np.concatenate((ls[i][j], np.zeros((max_sent_len - len(ls[i][j]), max_sent_len))), axis=0)
                        ls[i][j] = np.expand_dims(ls[i][j], 0)
                elif dtype == "np":
                    ls[i][j] = np.expand_dims(ls[i][j], axis=0)
            if dtype == "np":
                ls[i] = np.expand_dims(np.concatenate(ls[i], axis=0), axis=0)
        if dtype == "np":
            ls = np.concatenate(ls, axis=0)
    return ls


def tensor_padding(tensor):
    max_len = max([len(line) for line in tensor])
    for i, line in enumerate(tensor):
        if len(line) < max_len:
            tensor[i] = torch.cat([tensor[i], torch.zeros([max_len - len(line)] + list(line.shape)[1:])]).unsqueeze(0)
        else:
            tensor[i] = tensor[i].unsqueeze(0)
    return torch.cat(tensor)

def get_linearized_pos(ls, max_len):
    new_index = [[[max_len * i + j for j, word in enumerate(sent)] for i, sent in enumerate(sample)] for sample in ls]
    linearized_batch = []
    for sample in new_index:
        linearized_sample = []
        for sent in sample:
            linearized_sample.extend(sent)
        linearized_batch.append(linearized_sample)
    return linearized_batch


class MLP(nn.Module):
    def __init__(self, num_layers, input_size, output_size, mid_size=128, activation=torch.tanh):
        super(MLP, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        self.activation = activation
        for i in range(num_layers):
            if num_layers == 1:
                self.layers.append(to_cuda(torch.nn.Linear(input_size, output_size)))
                break
            if i == 0:
                self.layers.append(to_cuda(torch.nn.Linear(input_size, mid_size)))
            elif i == num_layers - 1:
                self.layers.append(to_cuda(torch.nn.Linear(mid_size, output_size)))
            else:
                self.layers.append(to_cuda(torch.nn.Linear(mid_size, mid_size)))

    def forward(self, input_tensor):
        tmp = input_tensor
        for i in range(self.num_layers):
            tmp = self.activation(self.layers[i](tmp))
        return tmp


class Attention(nn.Module):
    def __init__(self, key_size, value_size, mid_size=128):
        super(Attention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.W1 = to_cuda(torch.nn.Linear(key_size, mid_size))
        self.W2 = to_cuda(torch.nn.Linear(value_size, mid_size))
        self.A = to_cuda(torch_utils.add_params([mid_size, mid_size], "attention-A"))

    def forward(self, key, value):
        tmp_key = self.W1(key)
        tmp_value = self.W2(value)
        result = torch.bmm(torch.matmul(tmp_key, self.A), torch.transpose(tmp_value, 1, 2))
        return result


class GNN(nn.Module):
    def __init__(self, vocab_len, hidden_size=128, num_loop=10, pretrained_embedding=None):
        super(GNN, self).__init__()
        self.node_embedder = nn.Embedding(vocab_len, hidden_size)
        self.vocab_len = vocab_len
        if pretrained_embedding is not None:
            self.node_embedder.weight = nn.Parameter(pretrained_embedding)
        self.hidden_size = hidden_size

        self.num_loop = num_loop
        self.gru_gnn = torch.nn.GRUCell(hidden_size, hidden_size)
        self.gnn_w = torch_utils.add_params([self.hidden_size, self.hidden_size], "gnn_w")
        self.gnn_b = torch_utils.add_params([self.hidden_size], "gnn_b")

        self.mention_predictor = MLP(3, self.hidden_size, 1)
        self.corefer_predictor = MLP(3, self.hidden_size * 2, 1)
        for param in self.mention_predictor.parameters():
            param.data.uniform_(-0.08, 0.08)
        for param in self.corefer_predictor.parameters():
            param.data.uniform_(-0.08, 0.08)

        self.attention = Attention(self.hidden_size, self.hidden_size)
        for param in self.attention.parameters():
            param.data.uniform_(-0.1, 0.1)

    def forward(self, batch, vocab, gold_mention=True, K=400):
        input_var_ls = batch.input_seq()
        adjacency_matrix = batch.adjacency()
        input_var = list_padding(input_var_ls, 3, vocab.stoi['<pad>'])
        input_var = torch.LongTensor(input_var)
        node_feature = self.node_embedder(input_var)
        adjacency_matrix = torch.Tensor(list_padding(adjacency_matrix, 3, dtype="np"))

        # get node representation with gnn
        for gnn_idx in range(self.num_loop):
            tmp_feature = torch.matmul(node_feature, self.gnn_w) + self.gnn_b  # [ds_size, ds_hid_dim]
            '''
            for tmp_i in range(len(self.adjacent)):
                for tmp_tgt_i in self.adjacent[tmp_i]:
                    new_node_feature[tmp_i] = new_node_feature[tmp_i] + tmp_feature[tmp_tgt_i]
                    '''
            tmp_feature = tmp_feature.reshape(list(input_var.shape) + [self.hidden_size])
            new_node_feature = torch.matmul(adjacency_matrix, tmp_feature)
            node_feature = self.gru_gnn(new_node_feature.reshape((-1, self.hidden_size)), node_feature.reshape((-1, self.hidden_size)))
        node_feature = node_feature.reshape([input_var.shape[0], -1, self.hidden_size])
        mention_score = self.mention_predictor(node_feature)
        linearized_node_pos = get_linearized_pos(batch.input_seq(), input_var.shape[2])
        node_feature_no_padding = [torch.cat([sample[pos].unsqueeze(0) for pos in linearized_node_pos[i]], dim=0) for i, sample in enumerate(node_feature)]
        mention_score_no_padding = [torch.cat([sample[pos].unsqueeze(0) for pos in linearized_node_pos[i]], dim=0) for i, sample in enumerate(mention_score)]
        node_feature = tensor_padding(node_feature_no_padding)
        mention_score = tensor_padding(mention_score_no_padding).reshape((len(input_var_ls), -1))

        # extract the features of mentions
        if gold_mention:
            coref_raw = batch.coreference()
            linearized_coref_pos = []
            linearized_last_coref = []
            coref_pos = []
            for i in range(len(batch.index)):
                processed_index = [[str(j) + '|' + index for index in sent] for j, sent in enumerate(batch.index[i])]
                index_linear = []
                for line in processed_index:
                    index_linear.extend(line)
                coref_one_discourse = [[index_linear.index(mention.split('|')[1] + '|' + mention.split('|')[-1]) for mention in relation] for relation in coref_raw[i]]
                linearized_one_discourse = []
                linearized_last_discourse = []
                for relation in coref_one_discourse:
                    linearized_one_discourse.extend(relation)
                    if len(relation):
                        linearized_last_discourse.extend([-1] + relation[0:-1])
                coref_pos.append(coref_one_discourse)
                linearized_coref_pos.append(linearized_one_discourse)
                linearized_last_coref.append(linearized_last_discourse)
            mention_feature = [torch.stack([node_feature[i][mention] for mention in sample]) for i, sample in enumerate(linearized_coref_pos)]
            mention_feature = tensor_padding(mention_feature)
            mention_score_mention = [torch.stack([mention_score[i][mention] for mention in sample]) for i, sample in enumerate(linearized_coref_pos)]
            mention_score_mention = tensor_padding(mention_score_mention)

            """
            for i in range(len(linearized_coref_pos)):
                for j in range(len(linearized_coref_pos[i])):
                    if linearized_coref_pos[i][j] <= linearized_last_coref[i][j]:
                        print(linearized_coref_pos[i][j], linearized_last_coref[i][j], i, j)
            """

            attention_result = self.attention(mention_feature, node_feature)
            score = attention_result + mention_score_mention.unsqueeze(-1) + mention_score.unsqueeze(1)
            score = torch.cat([-10 * torch.ones([score.shape[0], score.shape[1], K]), score], dim=-1)
            score_for_prediction = [torch.stack([score[i][j][idx:idx+K] for j, idx in enumerate(sample)]) for i, sample in enumerate(linearized_coref_pos)]
            result = [torch.softmax(sample, dim=-1) for sample in score_for_prediction]
            target = [[int(sample[i] != -1) * (K - 1 + sample[i] - linearized_coref_pos[j][i]) for i in range(len(sample))] for j, sample in enumerate(linearized_last_coref)]
        return result, target
