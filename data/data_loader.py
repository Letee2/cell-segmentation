# 2. data/data_loader.py
import os
import glob
import re
import numpy as np
from pathlib import Path
import yaml
from typing import List, Dict, Tuple, Any

class DataLoader:
    def __init__(self, config_path: str):
        """Inicializa el cargador de datos con la configuración proporcionada."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.input_dir = self.config['data']['input_dir']
        self.output_dir = self.config['data']['output_dir']
        self.pattern = self.config['data']['pattern']
        
        # Crear directorio de salida si no existe
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_image_paths(self) -> List[str]:
        """Obtiene las rutas de todas las imágenes que coinciden con el patrón."""
        pattern_path = os.path.join(self.input_dir, self.pattern)
        return sorted(glob.glob(pattern_path))
    
    def organize_by_scene(self) -> Dict[str, List[str]]:
        """Organiza las imágenes por escena basándose en el patrón de nombre."""
        image_paths = self.get_image_paths()
        scenes = {}
        
        for path in image_paths:
            filename = os.path.basename(path)
            # Extraer el número de escena usando regex
            match = re.search(r'Maxt0_(\d+)_', filename)
            if match:
                scene_num = match.group(1)
                if scene_num not in scenes:
                    scenes[scene_num] = []
                scenes[scene_num].append(path)
        
        return scenes
    
    def get_output_path(self, input_path: str, suffix: str = '_mask') -> str:
        """Genera la ruta de salida para una imagen dada."""
        filename = os.path.basename(input_path)
        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}{suffix}{ext}"
        return os.path.join(self.output_dir, output_filename)