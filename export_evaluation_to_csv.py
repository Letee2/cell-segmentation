# export_evaluation_to_csv.py
import os
import json
import csv
from glob import glob

def export_evaluations(output_dir, output_csv="evaluation_metrics.csv"):
    # Buscar todos los archivos de metadatos
    metadata_paths = glob(os.path.join(output_dir, "*_metadata.json"))

    if not metadata_paths:
        print("❌ No se encontraron archivos *_metadata.json en:", output_dir)
        return

    rows = []
    for path in metadata_paths:
        with open(path, 'r') as f:
            data = json.load(f)
        
        filename = os.path.basename(path).replace("_metadata.json", "")
        evaluation = data.get("evaluation", None)
        if evaluation:
            row = {
                "filename": filename,
                "iou": evaluation.get("iou", ""),
                "dice": evaluation.get("dice", ""),
                "precision": evaluation.get("precision", ""),
                "recall": evaluation.get("recall", ""),
                "accuracy": evaluation.get("accuracy", "")
            }
            rows.append(row)

    if not rows:
        print("⚠️ No se encontraron métricas de evaluación en los metadatos.")
        return

    csv_path = os.path.join(output_dir, output_csv)
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Exportación completada: {csv_path} ({len(rows)} archivos evaluados)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exportar métricas de evaluación a CSV")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directorio con los archivos *_metadata.json generados")
    parser.add_argument("--output-csv", type=str, default="evaluation_metrics.csv",
                        help="Nombre del archivo CSV de salida")
    args = parser.parse_args()

    export_evaluations(args.output_dir, args.output_csv)
