from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

import seq2seq
from seq2seq.evaluator import Evaluator
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.util.checkpoint import Checkpoint
from autoeval.eval_embedding import Embed


class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """

    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64, train_dep=None, valid_dep=None,
                 random_seed=None, loss_plan=None, loss_choose=None, loss_reconstruct=None,
                 checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.loss_plan = loss_plan
        self.loss_choose = loss_choose
        self.loss_reconstruct = loss_reconstruct
        self.evaluator = Evaluator(loss=self.loss, loss_plan=loss_plan, loss_reconstruct=loss_reconstruct, batch_size=batch_size, valid_dep=valid_dep)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.train_dep = train_dep
        self.valid_dep = valid_dep

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio,
                     concept=None, vocabs=None, use_concept=False):
        loss = self.loss
        loss_plan = self.loss_plan
        loss_choose = self.loss_choose
        loss_reconstruct = self.loss_reconstruct
        # Forward propagation
        if not use_concept:
            decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                           teacher_forcing_ratio=teacher_forcing_ratio, concept=concept,
                                                           vocabs=vocabs, use_concept=use_concept)
        else:
            src_vocab = vocabs.src_vocab
            np_concept = concept.tolist() if not torch.cuda.is_available() else concept.cpu().tolist()
            cpt_vocab = vocabs.cpt_vocab
            np_concept = [line + [cpt_vocab.stoi['<pad>']] for line in np_concept]
            sample_index = [line[line.index(cpt_vocab.stoi['<index>'])+1:line.index(cpt_vocab.stoi['<pad>'])][0] for line in np_concept]
            sample_index = [int(cpt_vocab.itos[index]) for index in sample_index]

            (decoder_outputs, decoder_hidden, other), (planned_prob, planned_ground), (all_choose_rates, choose_ground), (reconstruct_prob, reconstruct_ground) = model(
                input_variable,
                input_lengths,
                target_variable,
                teacher_forcing_ratio=teacher_forcing_ratio,
                concept=concept,
                vocabs=vocabs,
                use_concept=use_concept,
                sample_index=sample_index,
                dep_corpus=self.train_dep)
            all_choose_rates = [torch.log(torch.cat([1 - step, step], dim=-1)) for step in all_choose_rates]
        # Get loss
        loss.reset()
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
            #loss_choose.eval_batch(all_choose_rates[step], choose_ground[:, step])
        # Get planning loss
        if use_concept:
            loss_choose.reset()
            loss_plan.reset()
            loss_reconstruct.reset()
            planned_prob = [torch.log(line) for line in planned_prob]
            #reconstruct_prob = torch.log(reconstruct_prob)
            cnt = 1
            for i in range(len(planned_ground)):
                for j in range(len(planned_ground[i])):
                    index_to_update = 0
                    arg1 = planned_prob[index_to_update][i].unsqueeze(0)
                    arg2 = torch.tensor([planned_ground[i][j]])
                    if torch.cuda.is_available():
                        arg2 = arg2.cuda()
                    loss_plan.acc_loss += loss_plan.criterion(arg1, arg2)
                    loss_plan.norm_term += 1
                    cnt -= 1
                    if cnt == 0:
                        break
                if cnt == 0:
                    break
            loss_choose.reset()
            for i in range(len(all_choose_rates)):
                for j in range(len(choose_ground)):
                    if choose_ground[j][i] == 1:
                        arg1 = all_choose_rates[i][j].unsqueeze(0)
                        arg2 = torch.tensor([choose_ground[j][i]])
                        if torch.cuda.is_available():
                            arg2 = arg2.cuda()
                        loss_choose.acc_loss += loss_choose.criterion(arg1, arg2)
                        loss_choose.norm_term += 1
            for i in range(len(reconstruct_ground)):
                for j in range(len(reconstruct_ground[i])):
                    for word in reconstruct_ground[i][j]:
                        arg1 = torch.log(reconstruct_prob[i][j].unsqueeze(0))
                        arg2 = torch.tensor([src_vocab.stoi[word]])
                        if torch.cuda.is_available():
                            arg2 = arg2.cuda()
                        loss_reconstruct.acc_loss += loss_reconstruct.criterion(arg1, arg2)
                        loss_reconstruct.norm_term += 1

        # Backward propagation
        model.zero_grad()

        lvalue = loss.get_loss()
        if lvalue >= 0:
            if use_concept:
                plan_value = loss_plan.get_loss()
                reconstruct_value = loss_reconstruct.get_loss()
                choose_value = loss_choose.get_loss()
                loss_reconstruct.acc_loss *= 0.2
                #loss.backward(retain_graph=True)
                #loss_reconstruct.backward(retain_graph=True)
                #loss_choose.backward(retain_graph=True)
                #loss.backward()
                loss_plan.backward()
            else:
                loss.backward()
            self.optimizer.step()
        else:
            raise AssertionError("NAN Triggered!")
        if use_concept:
            return lvalue, plan_value, reconstruct_value
        else:
            return lvalue

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step, save_file=False, dev_data=None,
                       teacher_forcing_ratio=0, vocabs=None, use_concept=False, log_dir=None, embed_file=None):
        log = self.logger
        # embed = Embed(embed_file)
        embed = []

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch
        plan_loss_total = 0
        construct_loss_total = 0

        device = torch.device('cuda', 0) if torch.cuda.is_available() else None
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False,
            shuffle=False)

        steps_per_epoch = len(batch_iterator)
        print("Steps per epoch: ", steps_per_epoch)
        total_steps = steps_per_epoch * n_epochs

        """
        dev_loss, accuracy = self.evaluator.evaluate(model, dev_data, vocabs=vocabs,
                                                     use_concept=use_concept,
                                                     log_dir=log_dir,
                                                     cur_step=0
                                                     )
        exit(0)
        """

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log_file = open(log_dir + "/log.txt", "a+")
            print("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
                if use_concept:
                    concepts, _ = getattr(batch, seq2seq.cpt_field_name)
                else:
                    concepts = []
                target_variables = getattr(batch, seq2seq.tgt_field_name)

                all_param = []
                for par in model.named_parameters():
                    all_param.append(par)
                all_loss = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model,
                                             teacher_forcing_ratio, concept=concepts, vocabs=vocabs,
                                             use_concept=use_concept)
                for par in model.named_parameters():
                    print(par)
                if use_concept:
                    loss, loss_plan, loss_reconstruct = all_loss
                else:
                    loss = all_loss
                    loss_plan = 0
                    loss_reconstruct = 0

                # FOR NAN DEBUG
                if not loss >= 0:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields[seq2seq.src_field_name].vocab,
                               output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)
                    print("Nan Triggered! Model has been saved.")
                    exit(0)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss
                plan_loss_total += loss_plan
                construct_loss_total += loss_reconstruct

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Step %d, Progress: %d%%, Train %s: %.2f, Plan Loss: %.2f, Reconstruct Loss: %.2f' % (
                        step,
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg,
                        loss_plan,
                        loss_reconstruct)
                    log.info(log_msg)

                if step % 200 == 0:
                    if log_file:
                        log_file.write("Step " + str(step) + '\n')
                    dev_loss, accuracy = self.evaluator.evaluate(model, dev_data, vocabs=vocabs,
                                                                 use_concept=use_concept, cur_step=step,
                                                                 log_dir=log_dir, log_file=log_file,
                                                                 multi_turn=1)
                    # self.optimizer.update(dev_loss, epoch)
                    log_msg = "Step %d, Dev %s: %.4f, Accuracy: %.4f" % (step, self.loss.name, dev_loss, accuracy)
                    log.info(log_msg)
                    model.train(mode=True)

                # Checkpoint
                """
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields[seq2seq.src_field_name].vocab,
                               output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)
                """
            if epoch % 1 == 0 and save_file:
                Checkpoint(model=model,
                           optimizer=self.optimizer,
                           epoch=epoch, step=step,
                           input_vocab=data.fields[seq2seq.src_field_name].vocab,
                           output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            plan_loss_avg = plan_loss_total / min(steps_per_epoch, step - start_step)
            construct_loss_avg = construct_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            plan_loss_total = 0
            construct_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f  Plan Loss: %.4f" % (
            epoch, self.loss.name, epoch_loss_avg, plan_loss_avg)
            if dev_data is not None:
                if log_file:
                    log_file.write("Step " + str(step) + '\n')
                    log_file.write(
                        "Train Average Loss: " + str(epoch_loss_avg) + " Plan Loss: " + str(plan_loss_avg) + " Construct Loss: " + str(construct_loss_avg) + '\n')
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data, vocabs=vocabs,
                                                             use_concept=use_concept,
                                                             log_dir=log_dir,
                                                             cur_step=step,
                                                             log_file=log_file,
                                                             multi_turn=1)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)
            log_file.close()

        Checkpoint(model=model,
                   optimizer=self.optimizer,
                   epoch=n_epochs, step=step,
                   input_vocab=data.fields[seq2seq.src_field_name].vocab,
                   output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)

    def train(self, model, data, num_epochs=5, resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0, src_vocab=None, cpt_vocab=None, tgt_vocab=None,
              use_concept=False, vocabs=None, save_file=False, log_dir=None, embed_file=None):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
            :param log_dir:
        """
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters(), weight_decay=0), max_grad_norm=5)
            self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, num_epochs,
                            start_epoch, step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio, log_dir=log_dir, embed_file=embed_file,
                            vocabs=vocabs, use_concept=use_concept, save_file=save_file)
        return model
