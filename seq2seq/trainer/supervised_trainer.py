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

    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None, checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)

    def _train_batch(self, model, batch, vocab, num_antecedents=400, teacher_forcing_ratio=0):
        loss = self.loss
        # Forward propagation
        result, target = model(batch, vocab, num_antecedents)

        # Get loss
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

        # Backward propagation
        model.zero_grad()

        lvalue = loss.get_loss()
        if lvalue >= 0:
            loss.backward()
            self.optimizer.step()
        else:
            raise AssertionError("NAN Triggered!")
        return lvalue

    def _train_epoches(self, data_train, data_valid, model, n_epochs, start_epoch, start_step, save_file=False, dev_data=None,
                       teacher_forcing_ratio=0, log_dir=None, embed_file=None, num_antecedents=400):
        log = self.logger
        # embed = Embed(embed_file)
        embed = []

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch
        plan_loss_total = 0
        construct_loss_total = 0

        device = torch.device('cuda', 0) if torch.cuda.is_available() else None
        batch_generator = data_train.__iter__()

        steps_per_epoch = len(batch_generator)
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

            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                all_loss = self._train_batch(model, batch, data_train.vocab, num_antecedents=num_antecedents)
                loss = all_loss
                # FOR NAN DEBUG
                if not loss >= 0:
                    Checkpoint(model=model, optimizer=self.optimizer, epoch=epoch, step=step,
                               input_vocab=data_train.vocab).save(self.expt_dir)
                    print("Nan Triggered! Model has been saved.")
                    exit(0)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Step %d, Progress: %d%%, Train %s: %.2f, ' % (
                        step,
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

                if step % 200 == 0:
                    if log_file:
                        log_file.write("Step " + str(step) + '\n')
                    dev_loss  = self.evaluator.evaluate(model, data_valid,
                                                                 cur_step=step, log_dir=log_dir, log_file=log_file)
                    # self.optimizer.update(dev_loss, epoch)
                    #log_msg = "Step %d, Dev %s: %.4f, Accuracy: %.4f" % (step, self.loss.name, dev_loss, accuracy)
                    log_msg = "Step %d, Dev %s: %.4f" % (step, self.loss.name, dev_loss)
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
                        "Train Average Loss: " + str(epoch_loss_avg) + " Plan Loss: " + str(
                            plan_loss_avg) + " Construct Loss: " + str(construct_loss_avg) + '\n')
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data,
                                                             log_dir=log_dir,
                                                             cur_step=step,
                                                             log_file=log_file)
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
                   input_vocab=data_train.vocab).save(self.expt_dir)

    def train(self, model, data_train, data_valid, num_epochs=5, resume=False, dev_data=None,
              optimizer=None, teacher_forcing_ratio=0, src_vocab=None, cpt_vocab=None, tgt_vocab=None,
              use_concept=False, vocabs=None, save_file=False, log_dir=None, embed_file=None, num_antecedents=400):
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

        self._train_epoches(data_train, data_valid, model, num_epochs, start_epoch, step, dev_data=dev_data,
                            teacher_forcing_ratio=teacher_forcing_ratio, log_dir=log_dir, embed_file=embed_file,
                            save_file=save_file, num_antecedents=num_antecedents)
        return model
