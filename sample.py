import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import pickle

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField, MyDataset, cal_coreference_dist
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from collections import namedtuple
from seq2seq.models.GNN import GNN

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3
# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data', default='/Users/mac/PycharmProjects/DialogSystem/ConceptNet/train.tsv')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data', default='/Users/mac/PycharmProjects/DialogSystem/ConceptNet/valid.tsv')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--save_file', action='store_true', dest='save_file',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--concept', default=True,
                    help='Logging level.')
parser.add_argument('--log_dir', default='/Users/mac/PycharmProjects/DialogSystem/logs')
parser.add_argument('--embed_file', default='/Users/mac/PycharmProjects/DialogSystem/pytorch-seq2seq/data/glove.6B.50d.txt')
parser.add_argument('--concept_level', type=str, default='simple', help='Logging level.')
parser.add_argument('--conceptnet', type=str, default='/Users/mac/PycharmProjects/DialogSystem/ConceptNet/concept_dict_simple.json')
parser.add_argument('--dep_dir', type=str, default='/Users/mac/PycharmProjects/DialogSystem/ConceptNet/')
parser.add_argument('--stopword', type=str, default='/Users/mac/PycharmProjects/DialogSystem/ConceptNet/stopword.txt')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

dataset = MyDataset("/Users/mac/Documents/NLP/Discourse Parsing/db-on-align")
dataset.construct_vocabulary(10000)
batch_generator = dataset.__iter__()
gnn = GNN(len(dataset.vocab), hidden_size=128, num_loop=5)
for batch in batch_generator:
    input_seq = batch.input_sequence
    adjacency_matrix = batch.adjacency()
    gnn(input_seq, adjacency_matrix, dataset.vocab)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = NLLLoss(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()
    if opt.concept:
        weight = torch.ones(len(src.vocab))
        pad = src.vocab.stoi[src.pad_token]
        loss_plan = NLLLoss(weight, pad)
        loss_choose = NLLLoss(torch.ones(2))
        loss_reconstruct = NLLLoss(weight, pad)
        if torch.cuda.is_available():
            loss_plan.cuda()
            loss_choose.cuda()
            loss_reconstruct.cuda()
    else:
        loss_plan = None
        loss_choose = None
        loss_reconstruct = None

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        dialog_hidden_size = 128
        dropout = 0.5
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=False)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2,
                             dropout_p=0.2, use_attention=True, bidirectional=True,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id, embedding=hidden_size, use_concept=opt.concept)
        dialog_encoder = torch.nn.LSTM(input_size=hidden_size*2 if bidirectional else hidden_size,
                                       hidden_size=dialog_hidden_size, batch_first=True, dropout=dropout)
        speaker_encoder = torch.nn.LSTM(input_size=hidden_size*2 if bidirectional else hidden_size,
                                       hidden_size=dialog_hidden_size, batch_first=True, dropout=dropout)
        if opt.concept:
            stopword = [word.strip() for word in open(opt.stopword).readlines()]
            seq2seq = Seq2seq(encoder, decoder, dialog_encoder=dialog_encoder, speaker_encoder=speaker_encoder, cpt_vocab=cpt.vocab,
                              hidden_size=dialog_hidden_size, concept_level=opt.concept_level, conceptnet_file=opt.conceptnet,
                              filter_vocab=small_vocab, stopword=stopword)
        else:
            seq2seq = Seq2seq(encoder, decoder, dialog_encoder=dialog_encoder, hidden_size=dialog_hidden_size)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32, loss_plan=loss_plan, loss_choose=loss_choose,
                          checkpoint_every=50, loss_reconstruct=loss_reconstruct, train_dep=train_dep, valid_dep=valid_dep,
                          print_every=1, expt_dir=opt.expt_dir)

    if opt.concept:
        VOCAB = namedtuple('vocabs', ("src_vocab", "tgt_vocab", "cpt_vocab"))
        vocabs = VOCAB(src.vocab, tgt.vocab, cpt.vocab)
    else:
        VOCAB = namedtuple('vocabs', ("src_vocab", "tgt_vocab"))
        vocabs = VOCAB(src.vocab, tgt.vocab)

    seq2seq = t.train(seq2seq, train,
                      num_epochs=30, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=1,
                      resume=opt.resume,
                      use_concept=opt.concept,
                      vocabs=vocabs,
                      save_file=opt.save_file,
                      embed_file=opt.embed_file,
                      log_dir=opt.log_dir)

predictor = Predictor(seq2seq, input_vocab, output_vocab)

"""
while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
"""
print("Training Finished.")
