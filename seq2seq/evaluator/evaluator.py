from __future__ import print_function, division

import torch
import torchtext

import seq2seq
import autoeval
from seq2seq.loss import NLLLoss
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from autoeval.eval_embedding import Embed
from autoeval.eval_distinct import distinct

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

    def evaluate(self, model, data, vocabs=None, use_concept=False, log_dir=None, embed=None, cur_step=0,
                 log_file=None, multi_turn=1):
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
        loss_plan = self.loss_plan
        if use_concept:
            loss_plan.reset()
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
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        src_vocab = data.fields[seq2seq.src_field_name].vocab
        if use_concept:
            cpt_vocab = data.fields['cpt'].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        cnt = 0
        loss_sum = 0
        plan_value = 0

        context_corpus = []
        reference_corpus = []
        prediction_corpus = []
        multi_turn_corpus = []
        state_corpus = []
        multi_turn_state_corpus = []
        with torch.no_grad():
            for batch in batch_iterator:
                cnt += 1
                input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)

                if torch.cuda.is_available():
                    input_index = input_variables.cpu().numpy()
                else:
                    input_index = input_variables.numpy()
                input_words = [[src_vocab.itos[word] for word in line] for line in input_index]
                context_corpus.extend(input_words)

                if use_concept:
                    concept, _ = getattr(batch, seq2seq.cpt_field_name)
                    np_concept = concept.tolist() if not torch.cuda.is_available() else concept.cpu().tolist()
                    np_concept = [line + [cpt_vocab.stoi['<pad>']] for line in np_concept]
                    sample_index = [
                        line[line.index(cpt_vocab.stoi['<index>']) + 1:line.index(cpt_vocab.stoi['<pad>'])][0]
                        for line in np_concept]
                    sample_index = [int(cpt_vocab.itos[index]) for index in sample_index]
                else:
                    concept = []
                    sample_index = []
                target_variables = getattr(batch, seq2seq.tgt_field_name)

                if multi_turn:
                    all_response = []
                    all_plan = []

                    def list_pad(li, vocab):
                        max_len = max([len(line) for line in li])
                        for i in range(len(li)):
                            if len(li[i]) < max_len:
                                li[i].extend((max_len - len(li[i])) * [vocab.stoi['<pad>']])
                        return torch.LongTensor(li).cuda() if torch.cuda.is_available() else torch.LongTensor(li)

                    chat_turns = 4
                    if 1:
                        for t in range(chat_turns):
                            if use_concept:
                                (decoder_outputs, decoder_hidden, other), plan, (all_choose_rates, choose_ground), (
                                reconstruct_prob, reconstruct_ground), res_all = model(
                                    input_variables,
                                    input_lengths.tolist(),
                                    target_variables,
                                    concept=concept,
                                    vocabs=vocabs,
                                    use_concept=use_concept,
                                    track_state=True,
                                    dep_corpus=self.valid_dep,
                                    sample_index=sample_index)
                                if not all_plan:
                                    all_plan = [[line] for line in res_all]
                                else:
                                    for i in range(len(res_all)):
                                        all_plan[i].append(res_all[i])
                            else:
                                decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(),
                                                                               target_variables, vocabs=vocabs)
                            sequence = torch.cat(other['sequence'], dim=1).tolist()
                            sequence = [
                                [tgt_vocab.itos[word] for word in line if word != tgt_vocab.stoi['<pad>']] + ['<eos>']
                                for line in sequence]
                            sequence = [line[:line.index('<eos>')] for line in sequence]
                            if not all_response:
                                all_response = sequence
                            else:
                                all_response = [all_response[k] + ['<sep>'] + sequence[k] for k in range(len(sequence))]
                            sequence_index = [[src_vocab.stoi[word] for word in line] for line in sequence]
                            # 这里不允许生成<pad>
                            input_list = [line + [src_vocab.stoi['<pad>']] for line in input_variables.tolist()]
                            input_list = [line[:line.index(src_vocab.stoi['<pad>'])] for line in input_list]
                            input_list = [input_list[i] + sequence_index[i] + [src_vocab.stoi['<eou>']] for i in
                                          range(len(input_list))]
                            if use_concept:
                                cpt_list = [line + [cpt_vocab.stoi['<pad>']] for line in concept.tolist()]
                                cpt_list = [line[:line.index(cpt_vocab.stoi['<pad>'])] for line in cpt_list]
                                new_concept = [list(set([word for word in line if word in model.cn.cpt_dict])) for line in
                                               sequence]
                                new_concept = [[cpt_vocab.stoi[word] for word in line] for line in new_concept]
                                cpt_list = [cpt_list[i][:cpt_list[i].index(cpt_vocab.stoi['<expand>'])] + new_concept[i] + [
                                    cpt_vocab.stoi['<eou>']] + cpt_list[i][cpt_list[i].index(cpt_vocab.stoi['<expand>']):]
                                            for i in range(len(cpt_list))]
                                cpt_test = [[cpt_vocab.itos[word] for word in line] for line in cpt_list]
                                concept = list_pad(cpt_list, cpt_vocab)
                            input_lengths = input_lengths.tolist()
                            input_lengths = [input_lengths[i] + len(sequence[i]) for i in range(len(input_lengths))]
                            input_lengths = torch.tensor(input_lengths)
                            input_variables = list_pad(input_list, src_vocab)
                            continue
                    multi_turn_corpus.extend(all_response)
                    multi_turn_state_corpus.extend(all_plan)

                if use_concept and not multi_turn:
                    """
                    (decoder_outputs, decoder_hidden, other), state = model(input_variables, input_lengths.tolist(),
                                                                            target_variables,
                                                                            concept=concept, vocabs=vocabs,
                                                                            use_concept=use_concept,
                                                                            track_state=use_concept)
                    state_corpus.extend(state)
                    """
                    (decoder_outputs, decoder_hidden, other), plan, (all_choose_rates, choose_ground), (
                    reconstruct_prob, reconstruct_ground), res_all = model(
                        input_variables,
                        input_lengths.tolist(),
                        target_variables,
                        concept=concept,
                        vocabs=vocabs,
                        use_concept=use_concept,
                        track_state=True)
                    state_corpus.extend(res_all)
                    planned_prob, planned_ground = plan
                    loss_plan.reset()
                    planned_prob = [torch.log(line) for line in planned_prob]
                    for i in range(len(planned_ground)):
                        for j in range(len(planned_ground[i])):
                            arg1 = planned_prob[j][i].unsqueeze(0)
                            arg2 = torch.tensor([planned_ground[i][j]])
                            if torch.cuda.is_available():
                                arg2 = arg2.cuda()
                            loss_plan.acc_loss += loss_plan.criterion(arg1, arg2)
                            loss_plan.norm_term += 1
                    plan_value = loss_plan.get_loss()
                elif not multi_turn:
                    decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(),
                                                                   target_variables, vocabs=vocabs)
                # Evaluation
                seqlist = other['sequence']
                reference = []
                prediction = []
                for step, step_output in enumerate(decoder_outputs):
                    if step + 1 >= target_variables.shape[1]:
                        break
                    target = target_variables[:, step + 1]
                    loss.eval_batch(step_output.view(target_variables.size(0), -1), target)
                    non_padding = target.ne(pad)
                    correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
                    match += correct
                    total += non_padding.sum().item()
                    if torch.cuda.is_available():
                        pred = seqlist[step].view(-1).cpu().numpy()
                        tgt = target.view(-1).cpu().numpy()
                    else:
                        pred = seqlist[step].view(-1).numpy()
                        tgt = target.view(-1).numpy()
                    for i in range(len(step_output)):
                        target_char = tgt_vocab.itos[tgt[i]]
                        pred_char = tgt_vocab.itos[pred[i]]
                        if target_char != '<pad>':
                            if len(reference) >= i + 1:
                                reference[i].append(target_char)
                            else:
                                reference.append([target_char])
                        if len(prediction) >= i + 1:
                            if prediction[i][-1] != '<eos>':
                                prediction[i].append(pred_char)
                        else:
                            prediction.append([pred_char])

                for i in range(len(reference)):
                    reference[i] = reference[i][:-1]
                    prediction[i] = prediction[i][:-1]
                reference_corpus.extend([[line] for line in reference])
                prediction_corpus.extend(prediction)
                if cnt > step_limit:
                    break

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

        return loss.get_loss(), accuracy
