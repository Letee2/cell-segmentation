# 8. main.py
import argparse
import yaml
import os
import json
import numpy as np
from segmentation.segmentation import SegmentationPipeline

def main():
    parser = argparse.ArgumentParser(description="Segmentación de células usando CellPose")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Ruta al archivo de configuración")
    parser.add_argument("--mode", type=str, choices=["all", "scene"], default="all",
                        help="Modo de procesamiento: todas las imágenes o por escena")
    parser.add_argument("--save-viz", action="store_true",
                        help="Guardar visualizaciones de matplotlib")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Ruta para guardar resultados en JSON")
    
    args = parser.parse_args()
    
    # Verificar que el archivo de configuración existe
    if not os.path.exists(args.config):
        print(f"Error: El archivo de configuración {args.config} no existe.")
        return
    
    # Inicializar pipeline
    pipeline = SegmentationPipeline(args.config)
    
    # Procesar imágenes según el modo
    if args.mode == "all":
        results = pipeline.process_all_images(save_visualization=args.save_viz)
    else:  # mode == "scene"
        results = pipeline.process_by_scene(save_visualization=args.save_viz)
    
    # Imprimir estadísticas básicas
    if args.mode == "all":
        stats = results['statistics']
        print("\nEstadísticas generales:")
        print(f"Total de imágenes procesadas: {stats['total_images']}")
        print(f"Total de células detectadas: {stats['total_cells']}")
        print(f"Promedio de células por imagen: {stats['avg_cells_per_image']:.2f}")
        print(f"Tiempo total de procesamiento: {stats['processing_time_seconds']:.2f} segundos")
    else:
        print("\nEstadísticas por escena:")
        for scene_id, scene_data in results.items():
            stats = scene_data['statistics']
            print(f"\nEscena {scene_id}:")
            print(f"  Imágenes: {stats['total_images']}")
            print(f"  Células: {stats['total_cells']}")
            print(f"  Promedio: {stats['avg_cells_per_image']:.2f} células/imagen")
    
    # Guardar resultados en JSON si se especificó
    if args.output_json:
        # Convertir arrays de NumPy a listas para serialización JSON
        def numpy_to_list(obj):
            if isinstance(obj, dict):
                return {key: numpy_to_list(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [numpy_to_list(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        serializable_results = numpy_to_list(results)
        
        with open(args.output_json, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResultados guardados en {args.output_json}")

if __name__ == "__main__":
    main()
