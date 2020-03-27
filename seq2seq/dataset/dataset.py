import logging
import os
import torchtext
import json
import numpy as np
from parse_eds import *
from collections import defaultdict


def test_empty(file):
    data = open(file).read()
    if not len(data):
        return True
    return False


def cal_coreference_dist(batch):
    coreference_dist = []
    for i in range(len(batch.index)):
        processed_index = [[str(j) + '|' + index for index in sent] for j, sent in enumerate(batch.index[i])]
        index_linear = []
        for line in processed_index:
            index_linear.extend(line)
        coreference = batch.data[i]['coreference']
        coreference_index = []
        for relation in coreference:
            coreference_relation = []
            for j in range(1, len(relation)):
                cur = relation[j].split('|')[1] + '|' + relation[j].split('|')[-1]
                last = relation[j-1].split('|')[1] + '|' + relation[j-1].split('|')[-1]
                dist = index_linear.index(cur) - index_linear.index(last)
                coreference_relation.append(cur + '|' + str(dist))
            coreference_index.append(coreference_relation)
        coreference_dist.append(coreference_index)
    return coreference_dist


class Vocabulary:
    def __init__(self, stoi, itos, frequency):
        self.stoi = stoi
        self.itos = itos
        self.frequency = frequency

    def __len__(self):
        return len(self.itos)


class BatchItem:
    def __init__(self, data, vocabulary):
        self.data = data
        self.vocabulary = vocabulary
        self.input_sequence = [[[vocabulary.stoi[word.split('|')[0]] for word in sentence['sequence'] if sentence['sentence']] for sentence in discourse['discourse']] for discourse in data]
        self.index = [[[word.split('|')[1] for word in sentence['sequence']] for sentence in discourse['discourse']] for discourse in data]
        self.adjacency_matrix = self.construct_adjacency_matrix()
        #self.coreference_pos = [[[mention + '|' + str(self.index[i][int(mention.split('|')[1])].index(mention.split('|')[-1])) for mention in relation] for relation in discourse['coreference']] for i, discourse in enumerate(data)]

    def input_seq(self):
        return self.input_sequence

    def adjacency(self):
        return self.adjacency_matrix

    def coreference(self):
        return [sample['coreference'] for sample in self.data]

    def construct_adjacency_matrix(self):
        adjacency_batch = []
        for i, discourse in enumerate(self.data):
            adjacency_discourse = []
            for j, sentence in enumerate(discourse['discourse']):
                index_to_pos = {}
                for k in range(len(self.index[i][j])):
                    index_to_pos[self.index[i][j][k]] = k
                eds = sentence['parsed_eds']
                adjacency_sent = np.zeros((len(index_to_pos), len(index_to_pos)))
                if not eds:
                    adjacency_discourse.append(adjacency_sent)
                    continue
                for node in eds.values():
                    if not isinstance(node, Q):
                        for arg in node.args:
                            if not isinstance(arg, Q):
                                adjacency_sent[index_to_pos[arg.index]][index_to_pos[node.index]] = 1
                                adjacency_sent[index_to_pos[node.index]][index_to_pos[arg.index]] = 1
                adjacency_discourse.append(adjacency_sent)
            adjacency_batch.append(adjacency_discourse)
        return adjacency_batch


class MyDataset:
    def __init__(self, data_dir, batch_size=3, vocab=None):
        all_list = sorted([file for file in os.listdir(data_dir) if file.startswith("dir")])
        data = []
        for file in all_list:
            if not test_empty(data_dir + '/' + file):
                raw_data = json.load(open(data_dir + '/' + file))
                for sent in raw_data['discourse']:
                    if sent['eds']:
                        sent['parsed_eds'] = parse_eds(sent['eds'][1:-1])
                        sent['sequence'] = [node.name + '|' + node.index for node in sent['parsed_eds'].values() if not isinstance(node, Q)]
                    else:
                        sent['parsed_eds'] = ""
                        sent['sequence'] = []

                count = sum([len(relation) for relation in raw_data['coreference']])
                if count:
                    data.append(raw_data)
        self.data = data
        self.iter_count = 0
        self.batch_size = batch_size
        self.vocab = vocab

    def construct_vocabulary(self, max_len=10000):
        frequency = {}
        stoi = defaultdict(int)
        itos = []
        for document in self.data:
            for sentence in document['discourse']:
                if not sentence['sentence']:
                    continue
                for word in sentence['sequence']:
                    name = word.split('|')[0]
                    if name in frequency:
                        frequency[name] += 1
                    else:
                        frequency[name] = 1
        sorted_freq = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
        itos.append('<unk>')
        itos.append('<pad>')
        stoi['<pad>'] = 1
        for i in range(min(max_len, len(sorted_freq))):
            itos.append(sorted_freq[i][0])
            stoi[sorted_freq[i][0]] = i + 2
        self.vocab = Vocabulary(stoi, itos, frequency)
        return None

    def __iter__(self):
        return self

    def __len__(self):
        return int(len(self.data) / self.batch_size)

    def __next__(self):
        self.iter_count += self.batch_size
        if self.iter_count > len(self.data):
            #start = self.iter_count - self.batch_size
            #self.iter_count = self.iter_count - len(self.data)
            #return self.data[start:] + self.data[:self.iter_count]
            self.iter_count = 0
            raise StopIteration
        else:
            return BatchItem(self.data[self.iter_count - self.batch_size:self.iter_count], self.vocab)

