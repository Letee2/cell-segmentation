# 7. segmentation/segmentation.py
from typing import List, Dict, Any, Tuple, Optional
import os
import numpy as np
import yaml
from tqdm import tqdm
import time
import re
from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from models.cellpose_model import CellPoseModel
from utils.visualization import Visualizer
from utils.file_utils import FileUtils

from utils.evaluate_segmentation import evaluate_masks
#import imageio.v3 as iio  # For reading TIFFs
import cv2 as cv  # Para leer imágenes normales


class SegmentationPipeline:
    def __init__(self, config_path: str):
        """Inicializa el pipeline de segmentación."""
        self.config_path = config_path
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Inicializar componentes
        self.data_loader = DataLoader(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.model = CellPoseModel(config_path)
        self.visualizer = Visualizer(config_path)
        self.file_utils = FileUtils(config_path)

    def evaluate_single_image(self, pred_mask_path: str, gt_mask_path: str) -> Dict[str, float]:
        """
        Compara máscara predicha vs ground truth y devuelve métricas.
        """
        pred_mask = cv.imread(pred_mask_path, cv.IMREAD_GRAYSCALE)
        gt_mask = cv.imread(gt_mask_path, cv.IMREAD_GRAYSCALE)
        return evaluate_masks(gt_mask, pred_mask)

    
    def process_single_image(self, image_path: str, 
                           save_mask: bool = True,
                           save_composite: bool = True,
                           save_metadata: bool = True,
                           save_visualization: bool = True,
                           save_flow_map: bool = True,
                           save_flow_animation: bool = False,
                           save_metrics_image: bool = True) -> Dict[str, Any]:
        """
        Procesa una sola imagen a través del pipeline completo.
        
        Args:
            image_path: Ruta a la imagen
            save_mask: Si se guarda la máscara de segmentación
            save_composite: Si se guarda la imagen compuesta
            save_metadata: Si se guardan los metadatos
            save_visualization: Si se guarda la visualización con matplotlib
            
        Returns:
            Diccionario con resultados y rutas de archivos guardados
        """
        # Cargar y preprocesar imagen
        image = self.preprocessor.load_image(image_path)
        processed_image = self.preprocessor.preprocess_image(image)
        
        # Segmentar imagen
        mask, info = self.model.segment_image(processed_image)
        
        # Preparar rutas de salida
        base_path = self.data_loader.get_output_path(image_path, suffix='')
        base_name, ext = os.path.splitext(base_path)
        
        # Guardar resultados
        results = {
            'image_path': image_path,
            'cell_count': len(np.unique(mask)) - 1,  # Restar 1 para excluir el fondo
            'files': {}
        }
        
        if save_mask:
            mask_path = f"{base_name}_mask{ext}"
            self.file_utils.save_mask(mask, mask_path)
            results['files']['mask'] = mask_path
        
        if save_composite:
            composite_path = f"{base_name}_composite{ext}"
            self.file_utils.save_composite(image, mask, composite_path)
            results['files']['composite'] = composite_path
        
        
        if save_visualization:
            viz_path = f"{base_name}_visualization.png"
            self.visualizer.create_segmentation_figure(image, mask, viz_path)
            results['files']['visualization'] = viz_path

        flows = info['flows'][1]  

        dy = flows[0]  # Componente Y (vertical)
        dx = flows[1]  # Componente X (horizontal)
        if flows.shape[0] != 2:
            raise ValueError(f"Formato inesperado de flows: se esperaba (2, H, W), pero se obtuvo {flows.shape}")

        if save_flow_map:
            flow_map_path = f"{base_name}_flowmap.png"
            self.visualizer.create_flowmap_figure(dy, dx, flow_map_path)
            results['files']['flow_map'] = flow_map_path

        if save_flow_animation:
            flow_anim_path = f"{base_name}_flowanim.gif"
            self.visualizer.create_flow_animation(
                image=processed_image,
                dy=dy,
                dx=dx,
                mask=mask,  
                save_path=flow_anim_path,
                n_frames=30,
                fps=5,
                flow_amplification=10
            )
            results['files']['flow_animation'] = flow_anim_path

            
        # Evaluar contra ground truth si está disponible y se guardan las máscaras
        gt_dir = self.config['data'].get('ground_truth_dir', None)
        evaluation = None
        if gt_dir and save_mask:
            basename = os.path.basename(image_path)                  # Maxt0_00_001.jpg
            name_no_ext = os.path.splitext(basename)[0]             # Maxt0_00_001
            gt_filename = f"{name_no_ext}_gt.png"                   # Maxt0_00_001_gt.png
            gt_path = os.path.join(gt_dir, gt_filename)

            if os.path.exists(gt_path):
                mask_path = self.data_loader.get_output_path(image_path, suffix='_mask')
                evaluation = self.evaluate_single_image(mask_path, gt_path)

            if evaluation:
                results['evaluation'] = evaluation
                # Guardar imagen de pixel accuracy si se solicita y se realiza evaluación
                if save_metrics_image:
                    metrics_image_path = self.visualizer.create_metrics_image(
                        mask_path = mask_path,      # máscara de la imagen
                        gt_path = gt_path,         # ground truth 
                    )
                    results['files']['metrics_image'] = metrics_image_path

        if save_metadata:
            metadata_path = f"{base_name}_metadata.json"
            self.file_utils.save_metadata(image_path, mask, info, metadata_path,extra_data = evaluation)
            results['files']['metadata'] = metadata_path

        return results
    
    def process_batch(self, image_paths: List[str], batch_size: int = 10, 
                     **kwargs) -> List[Dict[str, Any]]:
        """
        Procesa un lote de imágenes.
        
        Args:
            image_paths: Lista de rutas a imágenes
            batch_size: Tamaño del lote para procesamiento
            **kwargs: Argumentos para process_single_image
            
        Returns:
            Lista de diccionarios con resultados
        """
        results = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Procesando lotes"):
            batch_paths = image_paths[i:i+batch_size]
            batch_results = []
            
            for path in batch_paths:
                result = self.process_single_image(path, **kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
     # asegúrate de tener esto importado

    def process_all_images(self, **kwargs) -> Dict[str, Any]:
        """
        Procesa todas las imágenes encontradas por el DataLoader.
        
        Args:
            **kwargs: Argumentos para process_single_image
        Returns:
            Diccionario con resultados y estadísticas
        """
        image_paths = self.data_loader.get_image_paths()
        start_time = time.time()

        results = self.process_batch(image_paths, **kwargs)

        # Si existe un directorio de ground truth, procesamos las evaluaciones
        gt_dir = self.config['data'].get('ground_truth_dir', None)
        if gt_dir:
            for result in results:
                pred_path = result['files'].get('mask')
                if not pred_path:
                    continue

                # Obtener el nombre del archivo de la imagen
                basename = os.path.basename(pred_path)
                name_no_ext = os.path.splitext(basename)[0]  # Obtiene el nombre sin extensión
                name_no_ext = name_no_ext.replace("_mask", "")

                # Crear el nombre del archivo ground truth siguiendo el formato de process_single_image
                gt_filename = f"{name_no_ext}_gt.png"  
                gt_path = os.path.join(gt_dir, gt_filename)
                gt_path = os.path.normpath(gt_path)  # Asegura que la ruta esté bien formateada
                
                # Depuración: imprimir el nombre y la ruta de ground truth
                print(f"Buscando Ground Truth en: {gt_path}")

                if os.path.exists(gt_path):
                    # Evaluar la segmentación con el ground truth
                    metrics = self.evaluate_single_image(pred_path, gt_path)
                    result['evaluation'] = metrics
                else:
                    print(f"⚠️ Ground truth no encontrado: {gt_filename}")


        # Calcular estadísticas generales
        cell_counts = [r['cell_count'] for r in results]
        
        stats = {
            'total_images': len(results),
            'total_cells': sum(cell_counts),
            'avg_cells_per_image': np.mean(cell_counts),
            'processing_time_seconds': time.time() - start_time
        }
        
        return {
            'results': results,
            'statistics': stats
        }

    
    def process_by_scene(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Procesa imágenes organizadas por escena.
        
        Args:
            **kwargs: Argumentos para process_single_image
            
        Returns:
            Diccionario con resultados por escena
        """
        scenes = self.data_loader.organize_by_scene()
        scene_results = {}
        
        for scene_id, image_paths in scenes.items():
            print(f"Procesando escena {scene_id} ({len(image_paths)} imágenes)...")
            results = self.process_batch(image_paths, **kwargs)
            
            # Calcular estadísticas de la escena
            cell_counts = [r['cell_count'] for r in results]
            
            stats = {
                'total_images': len(results),
                'total_cells': sum(cell_counts),
                'avg_cells_per_image': np.mean(cell_counts),
            }
            
            scene_results[scene_id] = {
                'results': results,
                'statistics': stats
            }
        
        return scene_results