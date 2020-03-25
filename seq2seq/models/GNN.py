import torch.nn as nn
import torch
import numpy as np
from . import torch_utils


def list_padding(ls, dim, pad_id="<pad>", dtype="ls"):
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

    def forward(self, input_var, adjacency_matrix, vocab):
        input_var = list_padding(input_var, 3, vocab.stoi['<pad>'])
        input_var = torch.LongTensor(input_var)
        node_feature = self.node_embedder(input_var)
        adjacency_matrix = torch.Tensor(list_padding(adjacency_matrix, 3, dtype="np"))
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
        node_feature = node_feature.reshape(list(input_var.shape) + [self.hidden_size])
        mention_score = self.mention_predictor(node_feature)

        print(1)
