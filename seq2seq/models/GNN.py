import torch.nn as nn
import torch


def list_padding(ls, dim, pad_id):
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
                    ls[i].append([])
            for j in range(len(ls[i])):
                if len(ls[i][j]) < max_sent_len:
                    ls[i][j].extend((max_sent_len - len(ls[i][j])) * [pad_id])
    return ls


class GNN(nn.Module):
    def __init__(self, vocab_len, hidden_size=128, num_loop=10, pretrained_embedding=None):
        super(GNN, self).__init__()
        self.node_embedder = nn.Embedding(vocab_len, hidden_size)
        self.vocab_len = vocab_len
        if pretrained_embedding is not None:
            self.node_embedder.weight = nn.Parameter(pretrained_embedding)
        self.hidden_size = hidden_size
        self.num_loop = num_loop

    def forward(self, input_var, adjacency_matrix, vocab):
        input_var = list_padding(input_var, 3, vocab.stoi['<pad>'])
        input_var = torch.LongTensor(input_var)
        node_init = self.node_embedder(input_var)
        print(1)
