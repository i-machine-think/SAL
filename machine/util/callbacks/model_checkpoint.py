import shutil
import os

import machine
from machine.util import Checkpoint
from machine.util.callbacks import Callback


class ModelCheckpoint(Callback):
    """
    Model checkpoint to save weights during training. 
    This callback is automatically applied for every model that
    is trained with the SupervisedTrainer.

    Args:
        save_last (optional, bool): if True, save last top_k models
            instead of the best top_k models
    """

    def __init__(self, top_k=5, monitor='val',
                 save_last=False):
        super(ModelCheckpoint, self).__init__()
        self.top_k = top_k
        self.monitor = monitor
        self.save_last = save_last
        self.next_index = 1

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.expt_dir = trainer.expt_dir

    def on_epoch_begin(self, info=None):
        pass

    def on_epoch_end(self, info=None):
        pass

    def on_batch_begin(self, batch, info=None):
        pass

    def on_batch_end(self, batch, info=None):

        if info['checkpoint']:
            total_loss, _, model_name = \
                self.get_losses(info['eval_losses'],
                                info['eval_metrics'],
                                info['step'])

            max_eval_loss = max(self.loss_best)

            if total_loss < max_eval_loss or self.save_last:
                if self.save_last:
                    index_to_overwrite = self.next_index
                    self.next_index = (self.next_index + 1) % self.top_k
                else:
                    index_to_overwrite = self.loss_best.index(max_eval_loss)
                # rm prev model
                if self.best_checkpoints[index_to_overwrite] is not None:
                    shutil.rmtree(os.path.join(
                        self.expt_dir, self.best_checkpoints[index_to_overwrite]))
                self.best_checkpoints[index_to_overwrite] = model_name
                self.loss_best[index_to_overwrite] = total_loss

                # save model
                Checkpoint(model=self.trainer.model,
                           optimizer=self.trainer.optimizer,
                           epoch=info['epoch'], step=info['step'],
                           input_vocab=self.trainer.train_data.dataset.fields[
                               machine.src_field_name].vocab,
                           output_vocab=self.trainer.train_data.dataset.fields[
                               machine.tgt_field_name].vocab).save(self.expt_dir,
                                                                   name=model_name)

    def on_train_begin(self, info):

        total_loss, _, model_name = self.get_losses(info['eval_losses'],
                                                    info['eval_metrics'],
                                                    info['step'])

        self.loss_best = self.top_k*[total_loss]
        self.best_checkpoints = self.top_k*[None]
        self.best_checkpoints[0] = model_name

        # store first model
        Checkpoint(model=self.trainer.model,
                   optimizer=self.trainer.optimizer,
                   epoch=info['start_epoch'], step=info['start_step'],
                   input_vocab=self.trainer.train_data.dataset.fields[
                       machine.src_field_name].vocab,
                   output_vocab=self.trainer.train_data.dataset.fields[
                       machine.tgt_field_name].vocab).save(self.expt_dir,
                                                           name=model_name)

    def on_train_end(self, info=None):
        # TODO perhaps here also the model should be saved?
        pass
