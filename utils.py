import os
import numpy as np
from typing import List, Optional
from torch import nn
from torch.utils.data import DataLoader

from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput, denumpify_detensorize, has_length
from transformers.trainer_pt_utils import nested_concat, nested_numpify, find_batch_size
from transformers.trainer import logger

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self, return_separate_IoU=False):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        if return_separate_IoU:
            return MIoU
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype("int") + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


class TrainerForSemSegEval(Trainer):
    def __init__(self, num_class, pred_save_dir=None, *args, **kwargs):
        self.num_class = num_class
        self.evaluator = Evaluator(num_class)
        self.eval_results_per_image = []
        super().__init__(*args, **kwargs)
        
    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, args.prediction_loss_only, ignore_keys=ignore_keys)

            pred_labels = nn.functional.interpolate(
                logits,
                size=labels.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
            self.evaluator.add_batch(labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy())

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
                
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)

                # Set back to None to begin a new accumulation
                # losses_host, preds_host, inputs_host, labels_host = None, None, None, None
                losses_host= None

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        
        if has_length(dataloader):
            num_samples = self.num_examples(dataloader)
        else:  # both len(dataloader.dataset) and len(dataloader) fail
            num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        # if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(self.evaluator)

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        
        # mynote: it really does not make sense that put all the logits into one big array for semantic segmentation
        # as the predictions themselves are also image-like different from outputs of other tasks
        # `all_preds`==None and `all_labels`==None would only affect self.predict() that would use this function
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    


def get_dataset_dict(dataset_root):
    filenames = os.listdir(os.path.join(dataset_root, "image"))
    image_paths = [os.path.join(dataset_root, "image", ele) for ele in filenames]
    mask_paths = [ele.replace("image", "label").replace("jpg", "png").replace("JPG", "png") for ele in image_paths]
    dataset_dict = {"image": image_paths, "annotation": mask_paths}
    return dataset_dict
