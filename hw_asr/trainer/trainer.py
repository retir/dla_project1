import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
import multiprocessing
from pyctcdecode import build_ctcdecoder

from hw_asr.base import BaseTrainer
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.logger.utils import plot_spectrogram_to_buf
from hw_asr.metric.utils import calc_cer, calc_wer
from hw_asr.utils import inf_loop, MetricTracker, InfiniteWrapper


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            train_metrics,
            test_metrics,
            #metrics,
            optimizer,
            config,
            device,
            dataloaders,
            text_encoder,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, train_metrics, test_metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.text_encoder = text_encoder
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.train_dataloader = InfiniteWrapper(self.train_dataloader, infinite=False)
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = InfiniteWrapper(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50
        vocab = list(text_encoder.ind2char.values())
        vocab[0] = ''
        self.decoder = build_ctcdecoder(vocab)
        self.metrics = {'train': train_metrics, 'test': test_metrics}

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in train_metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in test_metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["spectrogram", "text_encoded"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx == 0 and epoch == 2:
                self.start_spec_augmentations()
            if batch_idx == 0 and epoch == 2:
                self.start_wave_augmentations()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        batch["probs"] = F.softmax(batch["logits"], dim=-1)
        batch["log_probs_length"] = self.model.transform_input_lengths(
            batch["spectrogram_length"]
        )
        batch["loss"] = self.criterion(**batch)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        metrics_to_use = self.metrics['train'] if is_train else self.metrics['test']
        for met in metrics_to_use:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(**batch)
            self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            text,
            log_probs,
            probs,
            log_probs_length,
            audio_path,
            audio,
            examples_to_log=10,
            *args,
            **kwargs,
    ):
        # TODO: implement logging of beam search results
        #print('LOG PRED', probs.shape)
        if self.writer is None:
            return
        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        #bs_texts = [self.text_encoder.ctc_beam_search(prob, prob_len, 100)[0][0] for prob, prob_len in zip(probs.detach().cpu().numpy(), log_probs_length.numpy())]
        #for i in range(len(bs_texts)):
        #    if len(bs_texts[i]) > log_probs_length[i]:
        #        print('WHAAAAAT')
        #        bs_texts[i] = bs_texts[i][:log_probs_length[i]]
        #bs_texts = [text[:log_probs_length[i]] for i in range(len(bs_texts))]
        
        #logits_list = [prob[:prob_len] for prob, prob_len in zip(probs.detach().cpu().numpy(), log_probs_length.numpy())]
        #with multiprocessing.get_context("fork").Pool(4) as pool:
        #    bs_texts = self.decoder.decode_batch(pool, logits_list, beam_width=10)
        tuples = list(zip(argmax_texts, text, argmax_texts_raw, audio_path))
        shuffle(tuples)
        rows = {}
        for max_pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = BaseTextEncoder.normalize_text(target)
            #bs_wer = calc_wer(target, bs_pred) * 100
            #bs_cer = calc_cer(target, bs_pred) * 100
            max_wer = calc_wer(target, max_pred) * 100
            max_cer = calc_cer(target, max_pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "max predictions": max_pred,
                #"bs predictions" : bs_pred,
                #"bs_wer": bs_wer,
                #"bs_cer": bs_cer,
                "max_wer": max_wer,
                "max_cer": max_cer,
            }
        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))
        self.writer.add_audio("augmentations", audio[0], 16000)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
