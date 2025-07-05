from ultralytics import YOLO
import torch
import torchvision
from pathlib import Path
import shutil

print("torchvision:", torchvision.__version__)
print("nms:", torchvision.ops.nms)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

model_path = "yolo11n.pt"
data_path = "augmented_dataset/dishes.yaml"

runs = [
    (
        "color_aug_lr0.005",
        {
            "lr0": 0.005,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4
        }
    ),
]

if __name__ == "__main__":
    for run_name, params in runs:
        print(f"\nStarting training: {run_name} "
              f"with params: {params}\n")

        model = YOLO(model_path)

        train_results = model.train(
            data=data_path,
            epochs=100,
            imgsz=640,
            device="0",
            batch=16,
            name=run_name,
            project="runs_zebra_dishes",
            patience=20,
            **params
        )

        val_metrics = model.val()
        test_metrics = model.val(split="test")
        best_pt = Path("runs_zebra_dishes") / run_name / "weights" / "best.pt"

        try:
            export_result = model.export(format="onnx", imgsz=640, simplify=True)
            if export_result is None:
                raise RuntimeError("Ultralytics export returned None.")

            export_path = Path(export_result)
            target_path = Path(f"./{run_name}.onnx")
            shutil.copy(export_path, target_path)

        except Exception as e:
            print(f"⚠️ Ошибка при экспорте ONNX: {e}")