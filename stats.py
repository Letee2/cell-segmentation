import os
import json

# Carpeta que contiene los JSON
folder = "output_segmentations"

# Inicialización de sumas y contador
metrics_sum = {
    "iou": 0.0,
    "dice": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "accuracy": 0.0
}
count = 0

# Recorrido de los archivos esperados
for i in range(64):
    for j in range(99):
        filename = f"Maxt0_{i:02d}_{j:03d}_metadata.json"
        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                eval_data = data.get("evaluation", {})
                if all(key in eval_data for key in metrics_sum):
                    for key in metrics_sum:
                        metrics_sum[key] += eval_data[key]
                    count += 1

# Calcular medias
if count == 0:
    print("No se encontraron archivos válidos.")
else:
    mean_metrics = {key: value / count for key, value in metrics_sum.items()}

    # Guardar en archivo .txt
    with open("mean_metrics.txt", "w") as out:
        out.write(f"Numero de archivos evaluados: {count}\n")
        for key, value in mean_metrics.items():
            out.write(f"{key}: {value:.6f}\n")

    print("Archivo 'mean_metrics.txt' creado con éxito.")
