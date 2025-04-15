# 9. run_segmentation.py
from segmentation.segmentation import SegmentationPipeline
import argparse
import os
import yaml

def main():
    parser = argparse.ArgumentParser(description="Ejecutar segmentación de células en imágenes")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Ruta al archivo de configuración")
    parser.add_argument("--input", type=str, default=None,
                        help="Directorio de entrada (sobrescribe la configuración)")
    parser.add_argument("--output", type=str, default=None,
                        help="Directorio de salida (sobrescribe la configuración)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Tamaño del lote para procesar imágenes")
    parser.add_argument("--scene", type=str, default=None,
                        help="ID de escena específica para procesar (opcional)")
    
    args = parser.parse_args()
    
    # Cargar configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Sobrescribir configuración si se proporcionan argumentos
    if args.input:
        config['data']['input_dir'] = args.input
    
    if args.output:
        config['data']['output_dir'] = args.output
        os.makedirs(args.output, exist_ok=True)
    
    # Guardar configuración modificada
    temp_config_path = "config/temp_config.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Inicializar pipeline
    pipeline = SegmentationPipeline(temp_config_path)
    
    # Procesar imágenes
    if args.scene:
        # Procesar escena específica
        scenes = pipeline.data_loader.organize_by_scene()
        if args.scene in scenes:
            print(f"Procesando escena {args.scene} ({len(scenes[args.scene])} imágenes)...")
            results = pipeline.process_batch(scenes[args.scene], batch_size=args.batch_size)
            
            # Mostrar estadísticas
            cell_counts = [r['cell_count'] for r in results]
            avg_cells = sum(cell_counts) / len(cell_counts) if cell_counts else 0
            
            print(f"\nImágenes procesadas: {len(results)}")
            print(f"Total de células: {sum(cell_counts)}")
            print(f"Promedio de células por imagen: {avg_cells:.2f}")
        else:
            print(f"Error: La escena {args.scene} no se encontró.")
    else:
        # Procesar todas las imágenes
        all_results = pipeline.process_all_images(batch_size=args.batch_size)
        stats = all_results['statistics']
        
        print("\nEstadísticas generales:")
        print(f"Total de imágenes procesadas: {stats['total_images']}")
        print(f"Total de células detectadas: {stats['total_cells']}")
        print(f"Promedio de células por imagen: {stats['avg_cells_per_image']:.2f}")
        print(f"Tiempo total de procesamiento: {stats['processing_time_seconds']:.2f} segundos")
    
    # Limpiar configuración temporal
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

if __name__ == "__main__":
    main()