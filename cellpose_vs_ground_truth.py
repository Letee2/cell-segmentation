import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import imageio.v3 as iio
import os
import csv
import sys

# Lista de identificadores
ids = ['t012', 't013', 't014', 't015', 't020', 't021', 't022', 't023', 't025', 't029', 
       't038', 't039', 't040', 't044', 't050', 't051', 't052', 't053', 't054', 't055', 
       't062', 't076', 't077', 't078', 't079', 't080', 't081', 't088']

def get_mask_metrics(mask_gt, mask_pred):
    mask_gt = mask_gt > 0
    mask_pred = mask_pred > 0

    # IoU
    intersection = np.logical_and(mask_gt, mask_pred).sum()
    union = np.logical_or(mask_gt, mask_pred).sum()
    iou = intersection / union if union != 0 else 0

    # DICE
    dice = 2 * intersection / (mask_gt.sum() + mask_pred.sum()) if (mask_gt.sum() + mask_pred.sum()) != 0 else 0

    # Pixel-wise metrics
    y_true = mask_gt.flatten()
    y_pred = mask_pred.flatten()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        'iou': iou,
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

def process_image(file_name):
    try:
        # Extraer el número del identificador (sin la 't')
        tnum = file_name[1:].zfill(3)  # Rellenamos con ceros a la izquierda hasta 3 dígitos
        
        # Paths
        base_path = "output_segmentations/Fluo-N2DL-HeLa_01"
        input_path = "input_images/Fluo-N2DL-HeLa_01"
        img_path = os.path.join(input_path, f"{file_name}.tif")
        mask_path = os.path.join(base_path, f"{file_name}_mask.tif")
        composite_path = os.path.join(base_path, f"{file_name}_composite.tif")
        # El ground truth tiene un formato diferente (man_segXXX.tif)
        ground_truth_path = os.path.join("ground_truth/Fluo-N2DL-HeLa_01", f"man_seg{tnum}.tif")
        
        # Verificar que todos los archivos existan
        for path in [img_path, mask_path, composite_path, ground_truth_path]:
            if not os.path.exists(path):
                print(f"Advertencia: El archivo {path} no existe. Saltando {file_name}.")
                return None
        
        # Leer archivos
        img = iio.imread(img_path)
        mask = iio.imread(mask_path)
        composite = iio.imread(composite_path)
        gt = iio.imread(ground_truth_path)
        
        # Generar visualización
        fig, ax = plt.subplots(1, 4, figsize=(18, 6))
        
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title("Imagen Original")
        ax[1].imshow(mask, cmap='viridis')
        ax[1].set_title("Máscara Segmentada")
        ax[2].imshow(composite)
        ax[2].set_title("Composite")
        ax[3].imshow(gt, cmap='viridis')
        ax[3].set_title("Ground Truth")
        
        for a in ax:
            a.axis('off')
        
        plt.tight_layout()
        
        # Guardar la visualización en la carpeta true_comparison
        output_img_path = os.path.join("true_comparison", f"{file_name}_comparison.png")
        plt.savefig(output_img_path)
        plt.close()  # Cerrar la figura para liberar memoria
        
        # Calcular métricas
        metrics = get_mask_metrics(gt, mask)
        
        print(f"Procesado {file_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error procesando {file_name}: {str(e)}")
        return None

def main():
    # Crear la carpeta true_comparison si no existe
    if not os.path.exists("true_comparison"):
        os.makedirs("true_comparison")
    
    # Crear un archivo CSV para guardar todas las métricas
    csv_path = os.path.join("true_comparison", "metrics_summary.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'iou', 'dice', 'precision', 'recall', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Procesar cada imagen
        for file_id in ids:
            metrics = process_image(file_id)
            if metrics:
                # Añadir el nombre del archivo a las métricas y guardar en CSV
                metrics['filename'] = file_id
                writer.writerow(metrics)
    
    print(f"Proceso completado. Las métricas se han guardado en {csv_path}")

if __name__ == "__main__":
    main()