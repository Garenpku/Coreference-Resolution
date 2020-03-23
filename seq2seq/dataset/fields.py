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


class Vocabulary:
    def __init__(self, stoi, itos, frequency):
        self.stoi = stoi
        self.itos = itos
        self.frequency = frequency

    def __len__(self):
        return len(self.stoi)


class BatchItem:
    def __init__(self, data, vocabulary):
        self.data = data
        self.vocabulary = vocabulary
        self.input_sequence = [[[vocabulary.stoi[word.split('|')[0]] for word in sentence['sequence'] if sentence['sentence']] for sentence in discourse['discourse']] for discourse in data]
        self.index = [[[word.split('|')[1] for word in sentence['sequence']] for sentence in discourse['discourse']] for discourse in data]
        self.adjacency_matrix = self.construct_adjacency_matrix()

    def input_seq(self):
        return self.input_sequence

    def adjacency(self):
        return self.adjacency_matrix

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
    def __init__(self, data_dir, batch_size=32):
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
                data.append(raw_data)
        self.data = data
        self.iter_count = 0
        self.batch_size = batch_size
        self.vocab = None

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


class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)

class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]
