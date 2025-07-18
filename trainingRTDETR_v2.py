from keys import HF_TOKEN, ROBOFLOW_API_KEY

import torch
import requests

import numpy as np
import supervision as sv
import albumentations as A
import matplotlib.pyplot as plt

from PIL import Image
from pprint import pprint
from roboflow import Roboflow
from dataclasses import dataclass, replace
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import TrainerCallback
import albumentations as A
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(checkpoint = "PekingU/rtdetr_r50vd_coco_o365"):
    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        anchor_image_size=None,
        ignore_mismatched_sizes=True,
    ).to(device)
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    return model, processor


def load_dataset(location=None):
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=f"{location}/train/images",
        annotations_path=f"{location}/train/_annotations_train.coco.json",
    )
    ds_valid = sv.DetectionDataset.from_coco(
        images_directory_path=f"{location}/valid/images",
        annotations_path=f"{location}/valid/_annotations_valid.coco.json",
    )
    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=f"{location}/test/images",
        annotations_path=f"{location}/test/_annotations.coco.json",
    )
    return ds_train, ds_valid, ds_test



def define_transformations():
    train_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"],
            clip=True,
            min_area=25
        ),
    )

    # Nessuna augmentazione per validation
    valid_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"],
            clip=True,
            min_area=1
        ),
    )
    return train_transform, valid_transform



class PyTorchDetectionDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset, processor, transform: A.Compose = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    @staticmethod
    def annotations_as_coco(image_id, categories, boxes):
        annotations = []
        for category, bbox in zip(categories, boxes):
            x1, y1, x2, y2 = bbox
            formatted_annotation = {
                "image_id": image_id,
                "category_id": category,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "iscrowd": 0,
                "area": (x2 - x1) * (y2 - y1),
            }
            annotations.append(formatted_annotation)

        return {
            "image_id": image_id,
            "annotations": annotations,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        _, image, annotations = self.dataset[idx]

        # Convert image to RGB numpy array
        image = image[:, :, ::-1]
        boxes = annotations.xyxy
        categories = annotations.class_id

        # Applica le trasformazioni se presenti
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=categories
            )
            image = transformed["image"].copy()
            boxes = transformed["bboxes"].copy()
            categories = transformed["class_labels"].copy()

        # Verifica che ci siano ancora bounding box dopo le trasformazioni
        if len(boxes) == 0:
            # Se non ci sono più box, ritorna l'immagine originale
            _, image, annotations = self.dataset[idx]
            image = image[:, :, ::-1]
            boxes = annotations.xyxy
            categories = annotations.class_id

        formatted_annotations = self.annotations_as_coco(
            image_id=idx, categories=categories, boxes=boxes)
        
        result = self.processor(
            images=image, annotations=formatted_annotations, return_tensors="pt")

        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

class MAPEvaluator:
    def __init__(self, image_processor, threshold=0.5, id2label=None):  # Threshold aumentato
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            batch_image_sizes = torch.tensor(np.array([x["size"] for x in batch]))
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, (height, width) in zip(target_batch, image_size_batch):
                boxes = target["boxes"]
                boxes = sv.xcycwh_to_xyxy(boxes)
                boxes = boxes * np.array([width, height, width, height])
                boxes = torch.tensor(boxes)
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
            post_processed_output = self.image_processor.post_process_object_detection(
                output, threshold=self.threshold, target_sizes=target_sizes
            )
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):
        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        mar_100_per_class = metrics.pop("mar_100_per_class")
        for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
            class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
            metrics[f"map_{class_name}"] = class_map
            metrics[f"mar_100_{class_name}"] = class_mar

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
        return metrics



class LearningRateTracker(TrainerCallback):
    def __init__(self):
        self.learning_rates = []
        self.steps = []
        self.epochs = []
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None and 'learning_rate' in logs:
            self.learning_rates.append(logs['learning_rate'])
            self.steps.append(state.global_step)
            self.epochs.append(state.epoch)
    
    def plot_learning_rate(self):
        plt.figure(figsize=(12, 4))
        
        # Plot vs steps
        plt.subplot(1, 2, 1)
        plt.plot(self.steps, self.learning_rates, 'b-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate vs Steps')
        plt.grid(True, alpha=0.3)
        
        # Plot vs epochs
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.learning_rates, 'r-', linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate vs Epochs')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
        
        # Stampa alcuni valori
        print(f"Initial LR: {self.learning_rates[0]:.2e}")
        print(f"Final LR: {self.learning_rates[-1]:.2e}")
        print(f"Max LR: {max(self.learning_rates):.2e}")
        print(f"Min LR: {min(self.learning_rates):.2e}")

if __name__ == '__main__':
    # Parametri di training migliorati

    #checkpoint = "./cartellaOutput-finetune/checkpoint-4250"
    checkpoint = "PekingU/rtdetr_r50vd_coco_o365"  # Modificato per usare un checkpoint pre-addestrato
    ds_train, ds_valid, ds_test = load_dataset(location = "./YOLO_to_COCO-1")

    train_transform, valid_transform = define_transformations()
    
    id2label = {id: label for id, label in enumerate(ds_train.classes)}
    label2id = {label: id for id, label in enumerate(ds_train.classes)}

    print("ID to Label Mapping:")
    pprint(id2label)
    print("Label to ID Mapping:")
    pprint(label2id)
    model, processor = load_model(checkpoint=checkpoint)

    pytorch_dataset_train = PyTorchDetectionDataset(ds_train, processor, transform=train_transform)
    pytorch_dataset_valid = PyTorchDetectionDataset(ds_valid, processor, transform=valid_transform)
    pytorch_dataset_test = PyTorchDetectionDataset(ds_test, processor, transform=valid_transform)

    eval_compute_metrics_fn = MAPEvaluator(image_processor=processor, threshold=0.5, id2label=id2label)


    lr_tracker = LearningRateTracker()

    training_args = TrainingArguments(
        output_dir="cartellaOutput-finetune",
        num_train_epochs=5,  # Ridotto da 70
        max_grad_norm=1.0,    # Aumentato da 0.1
        learning_rate=1e-4,   # Aumentato da 3e-5
        warmup_steps=300,     # Ridotto da 700
        fp16=True,
        per_device_train_batch_size=4,  # Ridotto per stabilità
        gradient_accumulation_steps=4,   # Aumentato per compensare batch size
        dataloader_num_workers=2,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="steps",      # Cambiato da "epoch"
        eval_steps=500,             # Valuta ogni 500 step
        save_strategy="steps",      # Cambiato da "epoch"
        save_steps=500,             # Salva ogni 500 step
        save_total_limit=3,         # Aumentato da 2
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        logging_steps=100,  
        lr_scheduler_type="cosine",        
        report_to=None,             
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pytorch_dataset_train,
        eval_dataset=pytorch_dataset_valid,
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
        callbacks=[lr_tracker],  # Aggiungi il tracker della learning rate
    )
    
    trainer.train(resume_from_checkpoint=None)
    trainer.save_model("./final_model")
    processor.save_pretrained("./final_model")
    lr_tracker.plot_learning_rate()
    plt.show()