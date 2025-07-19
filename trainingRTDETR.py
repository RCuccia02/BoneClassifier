from keys import HF_TOKEN, ROBOFLOW_API_KEY

import torch
import requests
import os
import numpy as np
import supervision as sv
import albumentations as A
import math
import time
import psutil
import gc
import json

from PIL import Image
from pprint import pprint
from roboflow import Roboflow
from dataclasses import dataclass, replace
from torch.utils.data import Dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    TrainingArguments,
    Trainer,
    get_cosine_schedule_with_warmup
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

CHECKPOINT = "PekingU/rtdetr_r50vd_coco_o365"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForObjectDetection.from_pretrained(CHECKPOINT).to(DEVICE)
processor = AutoImageProcessor.from_pretrained(CHECKPOINT)        

location = "./YOLO_to_COCO-1"
ds_train = sv.DetectionDataset.from_coco(
    images_directory_path=f"{location}/train/",
    annotations_path=f"{location}/train/_annotations.coco.json",
)
ds_valid = sv.DetectionDataset.from_coco(
    images_directory_path=f"{location}/valid/",
    annotations_path=f"{location}/valid/_annotations.coco.json",
)
ds_test = sv.DetectionDataset.from_coco(
    images_directory_path=f"{location}/test/",
    annotations_path=f"{location}/test/_annotations.coco.json",
)

id2label = {id: label for id, label in enumerate(ds_train.classes)}
label2id = {label: id for id, label in enumerate(ds_train.classes)}

train_transform = A.Compose([
    A.HorizontalFlip(p=0.3),  
    A.RandomBrightnessContrast(
        brightness_limit=0.02,  
        contrast_limit=0.02,    
        p=0.05                 
    ),
    A.Rotate(limit=5, p=0.1),
], bbox_params=A.BboxParams(
    format="coco",
    label_fields=["class_labels"],
    clip=True,
    min_area=5,         
    min_visibility=0.05 
))

valid_transform = A.Compose([
], bbox_params=A.BboxParams(
    format="coco",
    label_fields=["class_labels"],
    clip=True,
    min_area=1
))

class PytorchDetectionDataset(Dataset):
    def __init__(self, dataset: sv.DetectionDataset, processor, transform: A.Compose = None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform
        self.failed_samples = 0

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

        if image.shape[2] == 3:
            image = image[:, :, ::-1]  
        
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        image = np.clip(image, 0, 255)
        
        original_boxes = annotations.xyxy.copy()
        original_categories = annotations.class_id.copy()
        
        if self.transform:
            try:
                formatted_boxes = []
                for box in original_boxes:
                    x1, y1, x2, y2 = box
                    formatted_boxes.append([x1, y1, x2 - x1, y2 - y1])  
                
                transformed = self.transform(
                    image=image,
                    bboxes=formatted_boxes,
                    class_labels=original_categories
                )
                
                image = transformed["image"]
                transformed_boxes = transformed["bboxes"]
                categories = transformed["class_labels"]
                
                boxes = []
                for box in transformed_boxes:
                    x, y, w, h = box
                    boxes.append([x, y, x + w, y + h])
                boxes = np.array(boxes) if boxes else np.array([]).reshape(0, 4)
                
                if len(boxes) == 0 and len(original_boxes) > 0:
                    self.failed_samples += 1       
                    boxes = original_boxes
                    categories = original_categories
                    
            except Exception as e:
                print(f"Errore nella trasformazione per {idx}: {e}")
                boxes = original_boxes
                categories = original_categories
        else:
            boxes = original_boxes
            categories = original_categories

        if len(boxes) == 0:
            h, w = image.shape[:2]
            boxes = np.array([[w*0.25, h*0.25, w*0.75, h*0.75]])
            categories = np.array([0])  

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = np.clip(image, 0, 255)

        formatted_annotations = self.annotations_as_coco(
            image_id=idx, categories=categories, boxes=boxes)
        
        try:
            result = self.processor(
                images=image, 
                annotations=formatted_annotations, 
                return_tensors="pt"
            )
            
            result = {k: v[0] for k, v in result.items()}
            
            return result
            
        except Exception as e:
            print(f"Errore per {idx}: {e}")
            raise e

pytorch_dataset_train = PytorchDetectionDataset(ds_train, processor, transform=train_transform)
pytorch_dataset_valid = PytorchDetectionDataset(ds_valid, processor, transform=valid_transform)
pytorch_dataset_test = PytorchDetectionDataset(ds_test, processor, transform=valid_transform)

@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor

class calcoloMAP:
    def __init__(self, image_processor, threshold=0.05, id2label=None):  
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label
        self.debug_count = 0

    def collect_image_sizes(self, targets):
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
        total_detections_by_class = {}
        
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
            
            try:
                post_processed_output = self.image_processor.post_process_object_detection(
                    output, threshold=self.threshold, target_sizes=target_sizes
                )
                
                for pred in post_processed_output:
                    for label in pred['labels']:
                        class_name = self.id2label[label.item()] if self.id2label else str(label.item())
                        total_detections_by_class[class_name] = total_detections_by_class.get(class_name, 0) + 1
                
                if self.debug_count < 3:
                    total_detections = sum(len(pred['boxes']) for pred in post_processed_output)
                    max_score = max([pred['scores'].max().item() if len(pred['scores']) > 0 else 0 
                                   for pred in post_processed_output])
                    print(f"  Batch {self.debug_count}: {total_detections} previsioni, score massimo: {max_score:.3f}")
                    self.debug_count += 1
                
                post_processed_predictions.extend(post_processed_output)
                
            except Exception as e:
                print(f"Errore nelle predizioni post-processing: {e}")
                for _ in range(len(target_sizes)):
                    post_processed_predictions.append({
                        "boxes": torch.empty((0, 4)),
                        "scores": torch.empty((0,)),
                        "labels": torch.empty((0,), dtype=torch.long)
                    })
        
        if total_detections_by_class:
            for class_name, count in sorted(total_detections_by_class.items()):
                print(f"  {class_name}: {count} numero predizioni")
        else:
            print(f"\nNessuna predizione con questa soglia {self.threshold}")
        
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):
        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        total_target_boxes = sum(len(t["boxes"]) for t in post_processed_targets)
        total_pred_boxes = sum(len(p["boxes"]) for p in post_processed_predictions)
        
        print(f"  Boxes: {total_target_boxes}")
        print(f"  Box predetti: {total_pred_boxes}")
        print(f"  Soglia: {self.threshold}")

        try:
            evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
            evaluator.warn_on_many_detections = False
            evaluator.update(post_processed_predictions, post_processed_targets)
            metrics = evaluator.compute()

            classes = metrics.pop("classes")
            map_per_class = metrics.pop("map_per_class")
            mar_100_per_class = metrics.pop("mar_100_per_class")
            
            for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
                class_name = self.id2label[class_id.item()] if self.id2label is not None else class_id.item()
                metrics[f"map_{class_name}"] = class_map
                metrics[f"mar_100_{class_name}"] = class_mar

            metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
            
            print(f"  mAP: {metrics.get('map', 0):.4f}")
            print(f"  mAP@50: {metrics.get('map_50', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Errore nel calcolo delle metriche: {e}")
            return {
                "map": 0.0,
                "map_50": 0.0,
                "map_75": 0.0,
                "map_small": 0.0,
                "map_medium": 0.0,
                "map_large": 0.0,
                "mar_1": 0.0,
                "mar_10": 0.0,
                "mar_100": 0.0,
            }

eval_compute_metrics_fn = calcoloMAP(
    image_processor=processor, 
    threshold=0.05,  
    id2label=id2label
)

model = AutoModelForObjectDetection.from_pretrained(
    CHECKPOINT,
    id2label=id2label,
    label2id=label2id,
    anchor_image_size=None,
    ignore_mismatched_sizes=True,
).to(DEVICE)

def inferenzaPerformance(dataset, processor, model, device, num_samples=50):
    model.eval()
    total_time = 0.0
    used_samples = min(len(dataset), num_samples)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    with torch.no_grad():
        start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  
        if device.type == 'cuda':
            start_gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)  

        for _ in range(5):
            sample = dataset[0]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            _ = model(pixel_values)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()

        for i in range(used_samples):
            sample = dataset[i]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
            
            outputs = model(pixel_values)

        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()

        end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  
        if device.type == 'cuda':
            end_gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)  

    total_time = end_time - start_time
    avg_time = total_time / used_samples
    fps = 1.0 / avg_time

    print(f"Tempo di Inferenza Medio: {avg_time:.4f} sec")
    print(f"FPS: {fps:.2f}")
    print(f"RAM: {end_mem - start_mem:.2f} MB")

    if device.type == 'cuda':
        gpu_memory_used = end_gpu_mem - start_gpu_mem
        print(f"GPU: {gpu_memory_used:.2f} MB")
        return fps, end_mem - start_mem, gpu_memory_used
    else:
        return fps, end_mem - start_mem, None

def plotInferenza(fps, ram_used, gpu_used=None):
    labels = ["FPS", "RAM (MB)"]
    values = [fps, ram_used]
    colors = ["skyblue", "orange"]
    
    if gpu_used is not None:
        labels.append("GPU (MB)")
        values.append(gpu_used)
        colors.append("green")
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Metriche di Inferenza RT-DETR", fontsize=14, fontweight='bold')
    plt.ylabel("Value")
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height * 1.01, 
                f"{value:.2f}", ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0, max(values)*1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("rtdetr_performance.png", dpi=300, bbox_inches='tight')
    plt.show()

def collateFn(batch):
    try:
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        return data
    except Exception as e:
        print(f"Errore in collate_fn: {e}")
        print(f"Batch size: {len(batch)}")
        raise e

if __name__ == '__main__':
    total_epochs = 100
    steps_per_epoch = len(pytorch_dataset_train) // (8 * 1) 
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = int(10 * steps_per_epoch)  
    
    training_args = TrainingArguments(
        output_dir="rtdetr_improved_params",
        
        num_train_epochs=total_epochs,
        max_grad_norm=1.0,
        
        learning_rate=0.0001,      
        warmup_ratio=0.1,          
        lr_scheduler_type="cosine",
        
        optim="adamw_torch",
        weight_decay=0.0005,
        adam_epsilon=1e-8,
        
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,  
        gradient_accumulation_steps=1,
        dataloader_num_workers=2,      
        
        metric_for_best_model="eval_map_50",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=250,                
        save_strategy="steps",
        save_steps=500,                
        save_total_limit=5,           
        
        fp16=False,  
        dataloader_pin_memory=True,
        
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        logging_steps=50,              
        report_to=None,
        skip_memory_metrics=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pytorch_dataset_train,
        eval_dataset=pytorch_dataset_valid,
        tokenizer=processor,
        data_collator=collateFn,
        compute_metrics=eval_compute_metrics_fn,
    )
    
    trainer.train()
    
    final_model_path = "./rtdetr_improved_final"
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    
    if len(pytorch_dataset_test) > 0:
        test_results = trainer.evaluate(eval_dataset=pytorch_dataset_test)
        
        print(f"  mAP@50-95: {test_results.get('eval_map', 0):.4f}")
        print(f"  mAP@50: {test_results.get('eval_map_50', 0):.4f}")
        print(f"  mAP@75: {test_results.get('eval_map_75', 0):.4f}")
        
        for class_name in id2label.values():
            map_key = f"eval_map_{class_name}"
            if map_key in test_results:
                print(f"  {class_name}: mAP = {test_results[map_key]:.4f}")
        
        metrics_path = f"{final_model_path}/benchmark_metrics.json"
        benchmark_metrics = {
            "model_type": "RT-DETR",
            "training_config": "Parametri Equivalenti a YOLO",
            "epochs": total_epochs,
            "batch_size": training_args.per_device_train_batch_size,
            "learning_rate": training_args.learning_rate,
            "eval_threshold": eval_compute_metrics_fn.threshold,
            "test_metrics": test_results
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(benchmark_metrics, f, indent=2)