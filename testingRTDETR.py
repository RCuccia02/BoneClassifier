import torch
import numpy as np
import supervision as sv
import os
import matplotlib.pyplot as plt
import time
import psutil
import gc
import json
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

from PIL import Image
from transformers import AutoImageProcessor
from transformers.models.rt_detr import RTDetrForObjectDetection

FINE_TUNED_MODEL_PATH = r"C:/Users/w11/Desktop/trainingDETR/rtdetr_improved_params/checkpoint-16900"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_IMAGE_DIR = "rtdetr_yolo_style_analysis"
DATASET_LOCATION = "./YOLO_to_COCO-1"

INFERENCE_THRESHOLD = 0.3  
NMS_THRESHOLD = 0.1
NUM_PERFORMANCE_SAMPLES = 50
NUM_QUALITATIVE_SAMPLES = 10

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

try:
    model = RTDetrForObjectDetection.from_pretrained(FINE_TUNED_MODEL_PATH).to(DEVICE)
    processor = AutoImageProcessor.from_pretrained(FINE_TUNED_MODEL_PATH)
except Exception as e:
    exit(1)

try:
    ds_train = sv.DetectionDataset.from_coco(
        images_directory_path=f"{DATASET_LOCATION}/train/",
        annotations_path=f"{DATASET_LOCATION}/train/_annotations.coco.json",
    )
    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=f"{DATASET_LOCATION}/test/",
        annotations_path=f"{DATASET_LOCATION}/test/_annotations.coco.json",
    )
    
    id2label = {id: label for id, label in enumerate(ds_train.classes)}
    label2id = {label: id for id, label in enumerate(ds_train.classes)}
    
except Exception as e:
    print(f"Errore nel caricamento del dataset: {e}")
    exit(1)


def calcolaPredizioni(dataset, model, processor, device):  
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
                
            path, source_image, annotations = dataset[i]
            
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            w, h = image.size
            
            results = processor.post_process_object_detection(
                outputs, target_sizes=[(h, w)], threshold=0.01  
            )
            
            if len(results) > 0 and len(results[0]['scores']) > 0:
                pred_scores = results[0]['scores'].cpu().numpy()
                pred_labels = results[0]['labels'].cpu().numpy()
                pred_boxes = results[0]['boxes'].cpu().numpy()
                
                for score, label, box in zip(pred_scores, pred_labels, pred_boxes):
                    all_predictions.append({
                        'confidence': score,
                        'class': label,
                        'box': box,
                        'image_id': i
                    })
            
            if len(annotations.class_id) > 0:
                target_labels = annotations.class_id
                target_boxes = annotations.xyxy
                
                for label, box in zip(target_labels, target_boxes):
                    all_targets.append({
                        'class': label,
                        'box': box,
                        'image_id': i
                    })
    
    return all_predictions, all_targets

def calcolaMetriche(all_predictions, all_targets, confidence_thresholds):
    
    metrics_per_threshold = {}
    
    for conf_thresh in confidence_thresholds:
        filtered_preds = [p for p in all_predictions if p['confidence'] >= conf_thresh]
        
        class_metrics = {}
        overall_tp, overall_fp, overall_fn = 0, 0, 0
        
        for class_id, class_name in id2label.items():
            class_preds = [p for p in filtered_preds if p['class'] == class_id]
            class_targets = [t for t in all_targets if t['class'] == class_id]
            
            tp = min(len(class_preds), len(class_targets))  
            fp = max(0, len(class_preds) - len(class_targets))
            fn = max(0, len(class_targets) - len(class_preds))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
        
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics_per_threshold[conf_thresh] = {
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1
            },
            'per_class': class_metrics
        }
    
    return metrics_per_threshold

def costruzioneGrafici(metrics_per_threshold, confidence_thresholds):
     
    precisions_per_class = {class_name: [] for class_name in id2label.values()}
    recalls_per_class = {class_name: [] for class_name in id2label.values()}
    f1s_per_class = {class_name: [] for class_name in id2label.values()}
    
    overall_precisions = []
    overall_recalls = []
    overall_f1s = []
    
    for conf_thresh in confidence_thresholds:
        metrics = metrics_per_threshold[conf_thresh]
        
        overall_precisions.append(metrics['overall']['precision'])
        overall_recalls.append(metrics['overall']['recall'])
        overall_f1s.append(metrics['overall']['f1'])
        
        for class_name in id2label.values():
            if class_name in metrics['per_class']:
                precisions_per_class[class_name].append(metrics['per_class'][class_name]['precision'])
                recalls_per_class[class_name].append(metrics['per_class'][class_name]['recall'])
                f1s_per_class[class_name].append(metrics['per_class'][class_name]['f1'])
            else:
                precisions_per_class[class_name].append(0)
                recalls_per_class[class_name].append(0)
                f1s_per_class[class_name].append(0)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(id2label)))
    
    ax1.set_title("Precision-Recall Curve", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.grid(True, alpha=0.3)
    
    class_maps = {}
    for i, (class_name, color) in enumerate(zip(id2label.values(), colors)):
        recalls = recalls_per_class[class_name]
        precisions = precisions_per_class[class_name]
        
        if len(recalls) > 0 and max(recalls) > 0:
            ap = np.trapz(precisions, recalls) if len(recalls) > 1 else 0
            class_maps[class_name] = ap
            ax1.plot(recalls, precisions, color=color, label=f"{class_name} {ap:.3f}")
        else:
            class_maps[class_name] = 0
            ax1.plot([0], [0], color=color, label=f"{class_name} 0.000")
    
    overall_map = np.mean(list(class_maps.values()))
    ax1.plot(overall_recalls, overall_precisions, 'b-', linewidth=3, 
             label=f"all classes {overall_map:.3f} mAP@0.5")
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    ax2.set_title("Recall-Confidence Curve", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Recall")
    ax2.grid(True, alpha=0.3)
    
    for i, (class_name, color) in enumerate(zip(id2label.values(), colors)):
        recalls = recalls_per_class[class_name]
        ax2.plot(confidence_thresholds, recalls, color=color, label=class_name)
    
    max_recall_conf = confidence_thresholds[np.argmax(overall_recalls)] if overall_recalls else 0
    ax2.plot(confidence_thresholds, overall_recalls, 'b-', linewidth=3,
             label=f"all classes {max(overall_recalls):.2f} at {max_recall_conf:.3f}")
    ax2.legend()
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    ax3.set_title("F1-Confidence Curve", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Confidence")
    ax3.set_ylabel("F1")
    ax3.grid(True, alpha=0.3)
    
    for i, (class_name, color) in enumerate(zip(id2label.values(), colors)):
        f1s = f1s_per_class[class_name]
        ax3.plot(confidence_thresholds, f1s, color=color, label=class_name)
    
    max_f1 = max(overall_f1s) if overall_f1s else 0
    best_f1_conf = confidence_thresholds[np.argmax(overall_f1s)] if overall_f1s else 0
    ax3.plot(confidence_thresholds, overall_f1s, 'b-', linewidth=3,
             label=f"all classes {max_f1:.2f} at {best_f1_conf:.3f}")
    ax3.legend()
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    ax4.set_title("Precision-Confidence Curve", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Confidence")
    ax4.set_ylabel("Precision")
    ax4.grid(True, alpha=0.3)
    
    for i, (class_name, color) in enumerate(zip(id2label.values(), colors)):
        precisions = precisions_per_class[class_name]
        ax4.plot(confidence_thresholds, precisions, color=color, label=class_name)
    
    max_precision_conf = confidence_thresholds[np.argmax(overall_precisions)] if overall_precisions else 0
    ax4.plot(confidence_thresholds, overall_precisions, 'b-', linewidth=3,
             label=f"all classes {max(overall_precisions):.2f} at {max_precision_conf:.3f}")
    ax4.legend()
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    curves_path = os.path.join(OUTPUT_IMAGE_DIR, "yolo_style_curves.png")
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return curves_path, {
        'best_f1_threshold': best_f1_conf,
        'max_f1_score': max_f1,
        'overall_map': overall_map,
        'class_maps': class_maps
    }

def annotazione(image, annotations, classes_list):
    labels = [
        classes_list[class_id]
        for class_id in annotations.class_id
    ]

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)

    annotated_image = image.copy()
    annotated_image = bounding_box_annotator.annotate(annotated_image, annotations)
    annotated_image = label_annotator.annotate(annotated_image, annotations, labels=labels)
    return annotated_image

def inferenzaPerformance(dataset, model, processor, device, num_samples=50):
    model.eval()
    used_samples = min(len(dataset), num_samples)
        
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        if device.type == 'cuda':
            start_gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)

        for _ in range(5):
            path, _, _ = dataset[0]
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            _ = model(**inputs)

        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.time()

        for i in range(used_samples):
            path, _, _ = dataset[i % len(dataset)]
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            outputs = model(**inputs)
            w, h = image.size
            results = processor.post_process_object_detection(
                outputs, target_sizes=[(h, w)], threshold=INFERENCE_THRESHOLD
            )

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
        if device.type == 'cuda':
            end_gpu_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)

    total_time = end_time - start_time
    avg_time = total_time / used_samples
    fps = 1.0 / avg_time
    ram_used = end_mem - start_mem
    
    print(f"FPS: {fps:.2f}")
    print(f"RAM usata: {ram_used:.2f} MB")

    if device.type == 'cuda':
        gpu_used = end_gpu_mem - start_gpu_mem
        print(f"Memoria GPU usata: {gpu_used:.2f} MB")
        return fps, ram_used, gpu_used, avg_time
    else:
        return fps, ram_used, None, avg_time

def main():
    
    all_predictions, all_targets = calcolaPredizioni(
        ds_test, model, processor, DEVICE
    )
    
    confidence_thresholds = np.linspace(0.0, 1.0, 21)  
    
    metrics_per_threshold = calcolaMetriche(
        all_predictions, all_targets, confidence_thresholds
    )
    
    curves_path, curve_metrics = costruzioneGrafici(
        metrics_per_threshold, confidence_thresholds
    )
    
    fps, ram_used, gpu_used, avg_time = inferenzaPerformance(
        ds_test, model, processor, DEVICE, NUM_PERFORMANCE_SAMPLES
    )
    
    complete_results = {
        "model_info": {
            "model_path": FINE_TUNED_MODEL_PATH,
            "device": str(DEVICE),
            "inference_threshold": INFERENCE_THRESHOLD
        },
        "curve_analysis": {
            "best_f1_threshold": curve_metrics['best_f1_threshold'],
            "max_f1_score": curve_metrics['max_f1_score'],
            "overall_map": curve_metrics['overall_map'],
            "class_maps": curve_metrics['class_maps']
        },
        "performance_metrics": {
            "fps": round(fps, 2),
            "avg_inference_time_ms": round(avg_time * 1000, 2),
            "ram_usage_mb": round(ram_used, 2),
            "gpu_usage_mb": round(gpu_used, 2) if gpu_used is not None else 0
        },
        "dataset_info": {
            "test_images": len(ds_test),
            "classes": list(id2label.values()),
            "total_predictions": len(all_predictions),
            "total_targets": len(all_targets)
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_file = os.path.join(OUTPUT_IMAGE_DIR, "yolo_style_analysis_results.json")
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"mAP@0.5: {curve_metrics['overall_map']:.3f}")
    print(f"Miglior punteggio F1: {curve_metrics['max_f1_score']:.3f}")
    print(f"Migliore soglia F1: {curve_metrics['best_f1_threshold']:.3f}")
    
    print(f"FPS: {fps:.2f}")
    print(f"Tempo medio: {avg_time*1000:.1f}ms")
    print(f"RAM: {ram_used:.2f} MB")
    if gpu_used is not None:
        print(f"GPU: {gpu_used:.2f} MB")

if __name__ == "__main__":
    main()