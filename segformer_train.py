import os
from PIL import Image
import numpy as np
import torch
from torch import nn

from transformers import AutoImageProcessor
from transformers import AutoConfig, AutoModelForSemanticSegmentation, TrainingArguments
import datasets
from utils import get_dataset_dict, TrainerForSemSegEval
import sys

@torch.no_grad()
def compute_metrics(evaluator):
    MIoU = evaluator.Mean_Intersection_over_Union()
    metric_outputs = {}
    # the naming fashion will automatically add eval_{`dict_name_of_the_test_set`}_...
    metric_outputs["acc"] = evaluator.Pixel_Accuracy()
    metric_outputs["mean_iou"] = np.nanmean(MIoU)
    metric_outputs["fwIoU"] = evaluator.Frequency_Weighted_Intersection_over_Union()
    evaluator.reset()
    return metric_outputs

if __name__ == "__main__":
    '''
    Define the dataset
    '''
    training_dir = sys.argv[1]
    validation_dir = sys.argv[2] if len(sys.argv) > 2 else training_dir
    train_dict = get_dataset_dict(training_dir)
    crack500_dict = get_dataset_dict(os.path.join(validation_dir, 'Crack500/resized/'))
    cfd_dict = get_dataset_dict(os.path.join(validation_dir, 'CrackForest/resized/'))
    deepcrack_dict = get_dataset_dict(os.path.join(validation_dir, 'DeepCrack/resized/'))


    def data_transforms(example_batch):
        images = [Image.open(x) for x in example_batch["image"]]
        labels = [Image.open(x).convert("L") for x in example_batch["annotation"]]
        inputs = image_processor(images, labels)
        return inputs
    
    train_ds = datasets.Dataset.from_dict(train_dict)
    train_ds.set_transform(data_transforms)
    train_ds.shuffle(seed=42)
    test_ds = {}
    test_ds["cfd"] = datasets.Dataset.from_dict(cfd_dict)
    test_ds["cfd"].set_transform(data_transforms)
    test_ds["crack500"] = datasets.Dataset.from_dict(crack500_dict)
    test_ds["crack500"].set_transform(data_transforms)
    test_ds["deepcrack"] = datasets.Dataset.from_dict(deepcrack_dict)
    test_ds["deepcrack"].set_transform(data_transforms)

    checkpoint = "nvidia/mit-b0"
    exp_name = "segformer+synthcrack42k"
    id2label = {0: "background", 1: "crack"}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", reduce_labels=False)
    '''
    train the model from scratch or pretrained from hugging face hub
    '''
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
    # model = AutoModelForSemanticSegmentation.from_config(
    #     AutoConfig.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
    # )


    training_args = TrainingArguments(
        output_dir=exp_name,
        learning_rate=6e-5,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        save_strategy="epoch",
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    training_args = TrainingArguments(
        output_dir=exp_name,
        learning_rate=6e-5,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        save_strategy="steps",
        save_total_limit=1,
        evaluation_strategy="steps",
        save_steps=1000,
        logging_steps =1000,
        metric_for_best_model="eval_deepcrack_mean_iou",
        load_best_model_at_end=True,
        logging_strategy="steps",
        remove_unused_columns=False,
        push_to_hub=False,
        seed=42,
        report_to="tensorboard"
    )

    trainer = TrainerForSemSegEval(
        2,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()