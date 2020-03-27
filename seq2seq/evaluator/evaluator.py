from __future__ import print_function, division

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

smoothie = SmoothingFunction().method4


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), loss_plan=None, loss_reconstruct=None, batch_size=64, valid_dep=None):
        self.loss = loss
        self.loss_plan = loss_plan
        self.loss_reconstruct = loss_reconstruct
        self.batch_size = batch_size
        self.valid_dep = valid_dep

    def evaluate(self, model, data, num_antecedents=400, log_dir=None, embed=None, cur_step=0, log_file=None):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """

        eval_limit = 5000
        step_limit = int(eval_limit / self.batch_size)

        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = torch.device('cuda', 0) if torch.cuda.is_available() else None
        """
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        """
        batch_iterator = data.__iter__()
        pad = data.vocab.stoi['<pad>']

        cnt = 0
        loss_sum = 0

        context_corpus = []
        reference_corpus = []
        prediction_corpus = []
        multi_turn_corpus = []
        state_corpus = []
        multi_turn_state_corpus = []
        with torch.no_grad():
            for batch in batch_iterator:
                cnt += 1
                result, target = model(batch, data.vocab, num_antecedents)

        """
        # Evaluation
        bleu = corpus_bleu(reference_corpus, prediction_corpus, smoothing_function=smoothie)
        # embedding = embed.eval_embedding(reference_corpus, prediction_corpus)
        distinct_1 = distinct(prediction_corpus, 1)
        distinct_2 = distinct(prediction_corpus, 2)
        print("Corpus BLEU: ", bleu)
        # print("Embedding dist: ", embedding)
        print("Distinct-1: ", distinct_1)
        print("Distinct-2: ", distinct_2)

        with open(log_dir + '/log-' + str(cur_step), 'w', encoding='utf-8') as file:
            if log_file:
                log_file.write("Distinct-1: " + str(distinct_1) + '\n')
                log_file.write("Distinct-2: " + str(distinct_2) + '\n')
                if use_concept:
                    log_file.write("Plan Loss: " + str(plan_value) + '\n\n')
                else:
                    log_file.write('\n')
            file.write("Corpus BLEU: " + str(bleu) + '\n')
            # file.write("Embedding Dist: " + str(embedding) + '\n')
            file.write("Distinct-1: " + str(distinct_1) + '\n')
            file.write("Distinct-2: " + str(distinct_2) + '\n\n')
            if use_concept:
                file.write("Plan Loss: " + str(plan_value) + '\n\n')
            for i in range(len(reference_corpus)):
                file.write("Context: " + '\n')
                context_str = " ".join(context_corpus[i])
                context_list = context_str.split('<eou>')[:-1]
                for j in range(len(context_list)):
                    if j % 2 == 0:
                        file.write("Speaker 1: ")
                    else:
                        file.write("Speaker 2: ")
                    file.write(context_list[j] + '\n')
                if use_concept and state_corpus:
                    file.write("\nStates: " + '\n')
                    for j in range(len(state_corpus[i])):
                        file.write("Plan " + str(j) + " : ")
                        for k in range(len(state_corpus[i][j])):
                            file.write(state_corpus[i][j][k] + ' ')
                        file.write("\n")
                if not multi_turn_corpus:
                    file.write("\nGold: " + ' '.join(reference_corpus[i][0]) + '\n\n')
                    file.write("Response: " + ' '.join(prediction_corpus[i]) + '\n\n')
                    file.write('\n')
                else:
                    file.write("\nInteraction:\n")
                    interactive_list = " ".join(multi_turn_corpus[i]).split('<sep>')
                    for j in range(len(interactive_list)):
                        if multi_turn_state_corpus:
                            file.write("\nStates:\n")
                            for l in range(len(multi_turn_state_corpus[i][j])):
                                file.write("Plan " + str(l) + " : ")
                                for k in range(len(multi_turn_state_corpus[i][j][l])):
                                    file.write(multi_turn_state_corpus[i][j][l][k] + ' ')
                                file.write("\n")
                            file.write('\n')

                        if (j + len(context_list)) % 2 == 0:
                            file.write("Speaker 1: ")
                        else:
                            file.write("Speaker 2: ")
                        file.write(interactive_list[j] + '\n')
                    file.write('\n')
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total
        """

        return loss.get_loss()
