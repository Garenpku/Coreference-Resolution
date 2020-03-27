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
from seq2seq.dataset import MyDataset, cal_coreference_dist
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from collections import namedtuple
from seq2seq.models.GNN import GNN

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
try:
    raw_input  # Python 2
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
parser.add_argument('--log_dir', default='/Users/mac/Documents/NLP/Discourse Parsing/Coreference-Resolution/logs')
parser.add_argument('--embed_file',
                    default='/Users/mac/PycharmProjects/DialogSystem/pytorch-seq2seq/data/glove.6B.50d.txt')
parser.add_argument('--stopword', type=str, default='/Users/mac/PycharmProjects/DialogSystem/ConceptNet/stopword.txt')
parser.add_argument('--num_antecedents', type=int, default=400)

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

data_train = MyDataset("/Users/mac/Documents/NLP/Discourse Parsing/train")
data_train.construct_vocabulary(10000)
data_valid = MyDataset("/Users/mac/Documents/NLP/Discourse Parsing/valid", data_train.vocab)
"""
batch_generator = dataset.__iter__()
gnn = GNN(len(dataset.vocab), hidden_size=128, num_loop=5)
weight = torch.ones(opt.number_antecedents)
loss = NLLLoss(weight, dataset.vocab.stoi['<pad>'])
for batch in batch_generator:
    result, target = gnn(batch, dataset.vocab, opt.number_antecedents)
    loss.reset()
    result = [torch.log(sample) for sample in result]
    for i in range(len(result)):
        for j in range(len(result[i])):
            if target[i][j] > 0 and target[i][j] < 400:
                arg1 = result[i][j].unsqueeze(0)
                arg2 = torch.tensor([target[i][j]])
                if torch.cuda.is_available():
                    arg2 = arg2.cuda()
                loss.acc_loss += loss.criterion(arg1, arg2)
                loss.norm_term += 1
    print(loss.get_loss())
"""

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(
        os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    # Prepare loss
    weight = torch.ones(opt.num_antecedents)
    loss = NLLLoss(weight, data_train.vocab.stoi['<pad>'])
    if torch.cuda.is_available():
        loss.cuda()
    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 128
        dropout = 0.5
        gnn = GNN(len(data_train.vocab), hidden_size=128, num_loop=5)
        if torch.cuda.is_available():
            gnn.cuda()

        for param in gnn.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=16, checkpoint_every=50, print_every=1, expt_dir=opt.expt_dir)

    model = t.train(gnn, data_train, data_valid,
                    num_epochs=30,
                    num_antecedents=opt.num_antecedents,
                    optimizer=optimizer,
                    teacher_forcing_ratio=1,
                    resume=opt.resume,
                    save_file=opt.save_file,
                    embed_file=opt.embed_file,
                    log_dir=opt.log_dir)

predictor = Predictor(model, input_vocab, output_vocab)

"""
while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
"""
print("Training Finished.")
