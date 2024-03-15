import sys
import os
from PIL import Image
import numpy as np
import torch
from torch import nn
from transformers import AutoImageProcessor
from transformers import AutoModelForSemanticSegmentation

import cv2
import numpy as np
from utils import Evaluator
import sys

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    palette = np.array([[255, 0, 0], [0, 255, 0]]) # background, crack
    
    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2]
    test_image_dir = sys.argv[3]
    os.makedirs(output_dir, exist_ok=True)

    id2label = {0: "background", 1: "crack"}
    label2id = {v: k for k, v in id2label.items()}
    image_processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", reduce_labels=False)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        checkpoint_path, id2label=id2label, label2id=label2id, local_files_only=True
    )
    model.to(device)

    if os.path.isdir(test_image_dir):
        image_paths = [os.path.join(test_image_dir, ele) for ele in os.listdir(test_image_dir) if "json" not in ele]
    else:
        image_paths = [test_image_dir]

    evaluator = Evaluator(len(id2label))
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path)
            inputs = image_processor(image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits.cpu()
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)
            pred_seg = upsampled_logits[0]
            color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
            img = np.array(image)
            for label, color in enumerate(palette[:len(id2label)]):
                # if id2label[label] == 'background':
                #     continue
                img[pred_seg == label, :] = img[pred_seg == label, :] * 0.5 + color * 0.5
            img = img.astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), img[:, :, ::-1])

            mask_path = image_path.replace("image", "label").replace("jpg", "png").replace("JPG", "png")
            mask = Image.open(mask_path).convert('L')
            assert image.size == mask.size, print('Size mismatch between image and mask.')
            mask = np.array(mask)
            pred_seg = pred_seg.cpu().numpy()
            evaluator.add_batch(mask, pred_seg)

    MIoU = evaluator.Mean_Intersection_over_Union(return_separate_IoU=True)
    mIoU = np.nanmean(MIoU)
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    print(f"Validation on {test_image_dir}:")
    print("mIoU:{}".format(MIoU))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
