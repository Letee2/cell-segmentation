import os
import shutil
import subprocess
import yaml
from datetime import datetime


# Lista de configuraciones a probar
experiments = [
    {"name": "cyto_d45_t06_c035",  "model_type": "cyto",  "diameter": 45, "flow_threshold": 0.6, "cellprob_threshold": 0.35},
    {"name": "cyto_d35_t07_c03",   "model_type": "cyto",  "diameter": 35, "flow_threshold": 0.7, "cellprob_threshold": 0.3},
    {"name": "cyto2_d30_t03_c01",  "model_type": "cyto2", "diameter": 30, "flow_threshold": 0.3, "cellprob_threshold": 0.1},
    {"name": "cyto2_d60_t06_c045", "model_type": "cyto2", "diameter": 60, "flow_threshold": 0.6, "cellprob_threshold": 0.45},
    {"name": "cyto2_d40_t07_c035", "model_type": "cyto2", "diameter": 40, "flow_threshold": 0.7, "cellprob_threshold": 0.35},
    {"name": "cyto3_d50_t03_c015", "model_type": "cyto3", "diameter": 50, "flow_threshold": 0.3, "cellprob_threshold": 0.15},
    {"name": "cyto3_d35_t05_c05",  "model_type": "cyto3", "diameter": 35, "flow_threshold": 0.5, "cellprob_threshold": 0.5},
    {"name": "cyto3_d45_t04_c025", "model_type": "cyto3", "diameter": 45, "flow_threshold": 0.4, "cellprob_threshold": 0.25},
    {"name": "cyto_d20_t06_c02",   "model_type": "cyto",  "diameter": 20, "flow_threshold": 0.6, "cellprob_threshold": 0.2},
    {"name": "cyto2_d0_t05_c035",  "model_type": "cyto2", "diameter": 0,  "flow_threshold": 0.5, "cellprob_threshold": 0.35}
]



def update_config(model_type, diameter, flow_threshold, cellprob_threshold):
    config_path = "config/config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["cellpose"]["model_type"] = model_type
    config["cellpose"]["diameter"] = diameter
    config["cellpose"]["flow_threshold"] = flow_threshold
    config["cellpose"]["cellprob_threshold"] = cellprob_threshold

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def run_experiment(exp):
    print(f"\n🚀 Ejecutando experimento: {exp['name']}")

    # 1. Actualizar config
    update_config(
        exp["model_type"],
        exp["diameter"],
        exp["flow_threshold"],
        exp["cellprob_threshold"],
    )

    # 2. Ejecutar segmentación
    subprocess.run(["python3", "run_segmentation.py"])

    # 3. Exportar métricas
    subprocess.run([
        "python3", "export_evaluation_to_csv.py",
        "--output-dir", "output_segmentations/Fluo-N2DL-HeLa_01/"
    ])

    # 4. Copiar CSV con nombre descriptivo
    original_csv = "output_segmentations/Fluo-N2DL-HeLa_01/evaluation_metrics.csv"
    if os.path.exists(original_csv):
        target_name = f"csv_metrics/{exp['name']}.csv"
        shutil.copyfile(original_csv, target_name)
        print(f"✅ Resultados guardados en: {target_name}")
    else:
        print("❌ No se encontró el archivo de métricas")



def analyze_results():
    import os
    import pandas as pd

    # Ruta a la carpeta con los CSV
    CSV_FOLDER = "csv_metrics"

    # Cargar todos los archivos CSV de la carpeta
    csv_files = [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]

    # Lista para guardar todos los DataFrames
    all_data = []

    for file in csv_files:
        df = pd.read_csv(os.path.join(CSV_FOLDER, file))
        df['source_file'] = file  # Para saber de qué CSV vino cada fila
        all_data.append(df)

    # Combinar todos los CSVs en uno solo
    combined_df = pd.concat(all_data, ignore_index=True)

    # Mostrar los mejores resultados según cada métrica
    print("🏆 Mejores resultados por métrica:\n")

    metrics = ['iou', 'dice', 'precision', 'recall', 'accuracy']
    for metric in metrics:
        best = combined_df.loc[combined_df[metric].idxmax()]
        print(f"🔹 {metric.upper()}:")
        print(f"  - Filename: {best['filename']}")
        print(f"  - Value: {best[metric]:.4f}")
        print(f"  - CSV: {best['source_file']}")
        print()

    # Opcional: Top 5 por IoU con resumen
    top_iou = combined_df.sort_values(by='iou', ascending=False).head(5)
    print("📈 Top 5 resultados por IoU:")
    print(top_iou[['filename', 'iou', 'dice', 'precision', 'recall', 'accuracy', 'source_file']])


if __name__ == "__main__":
    for exp in experiments:
        run_experiment(exp)
        analyze_results()