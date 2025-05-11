# 6. utils/file_utils.py
import os
import yaml
import json
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Any, List, Optional

from utils.visualization import Visualizer

class FileUtils:
    def __init__(self, config_path: str):
        """Inicializa utilidades de archivo con la configuración."""
        self.config_path = config_path  # Guardar la ruta del archivo de configuración
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.output_dir = self.config['data']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
    
    def save_mask(self, mask: np.ndarray, output_path: str) -> None:
        """Guarda la máscara de segmentación como imagen."""
        # Las máscaras de CellPose son enteros donde cada célula tiene un ID único
        # Para visualización, normalizamos y convertimos a uint8 o uint16 según el rango
        if mask.max() > 255:
            mask_save = mask.astype(np.uint16)
        else:
            mask_save = mask.astype(np.uint8)
        
        cv2.imwrite(output_path, mask_save)
    
    def save_composite(self, image: np.ndarray, mask: np.ndarray, output_path: str) -> None:
        """Guarda la imagen compuesta con contornos celulares."""
        visualizer = Visualizer(self.config_path)
        composite = visualizer.overlay_mask_on_image(image, mask)
        cv2.imwrite(output_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    def save_metadata(self, image_path: str, mask: np.ndarray, info: Dict[str, Any],
                  output_path: str,extra_data = None) -> None:
        """Guarda metadatos de la segmentación."""
        # Extraer información relevante
        cell_count = len(np.unique(mask)) - 1  # Restar 1 para excluir el fondo (ID 0)
        
        # Calcular propiedades básicas de las células
        cell_areas = []
        for cell_id in range(1, mask.max() + 1):
            cell_mask = (mask == cell_id)
            area = np.sum(cell_mask)
            cell_areas.append(int(area))
        
        # Obtener el diámetro de manera segura
        diameter = 0
        if isinstance(info, dict):
            if 'diameters' in info:
                diameters = info['diameters']
                if isinstance(diameters, (list, tuple, np.ndarray)) and len(diameters) > 0:
                    diameter = float(diameters[0])
                elif isinstance(diameters, (int, float)):
                    diameter = float(diameters)

        if len(cell_areas)==0:
            cell_areas.append(0)
        
        metadata = {
            'image_filename': os.path.basename(image_path),
            'cell_count': cell_count,
            'cell_areas': cell_areas,
            'median_cell_area': int(np.median(cell_areas)) if cell_areas else 0,
            'mean_cell_area': sum(cell_areas)/len(cell_areas),
            'cellpose_diameter': diameter
        }
        if extra_data:
            metadata["evaluation"] = extra_data

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

