# 3. data/data_preprocessor.py
import cv2
import numpy as np
from typing import Tuple, List, Optional

class DataPreprocessor:
    def __init__(self, config_path: str):
        """Inicializa el preprocesador con la configuración."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Carga una imagen desde la ruta especificada."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        return img
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesa la imagen para la segmentación con CellPose."""
        # Convertir a RGB si está en BGR (formato OpenCV)
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalizar la imagen si es necesario
        # CellPose funciona bien con imágenes en el rango [0, 1] o [0, 255]
        if image.max() > 1 and image.dtype != np.uint8:
            image = image / image.max()
            image = (image * 255).astype(np.uint8)
        
        return image
    
    def process_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        """Procesa un lote de imágenes."""
        processed_images = []
        for path in image_paths:
            img = self.load_image(path)
            processed = self.preprocess_image(img)
            processed_images.append(processed)
        return processed_images

# 4. models/cellpose_model.py
from cellpose import models
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import yaml

class CellPoseModel:
    def __init__(self, config_path: str):
        """Inicializa el modelo CellPose con la configuración dada."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_type = self.config['cellpose']['model_type']
        self.channels = self.config['cellpose']['channels']
        self.diameter = self.config['cellpose']['diameter']
        self.gpu = self.config['cellpose']['gpu']
        self.flow_threshold = self.config['cellpose']['flow_threshold']
        self.cellprob_threshold = self.config['cellpose']['cellprob_threshold']
        
        # Inicializar modelo CellPose
        self.model = models.Cellpose(
            gpu=self.gpu,
            model_type=self.model_type
        )
    
    def segment_image(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segmenta una sola imagen utilizando CellPose.
        
        Args:
            image: Imagen preprocesada para segmentar
            
        Returns:
            Tupla de (máscara, info)
            - máscara: Array donde cada célula tiene un ID único
            - info: Información adicional de la segmentación
        """
        masks, flows, styles, diams = self.model.eval(
            image,
            diameter=self.diameter,
            channels=self.channels,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            do_3D=False
        )
        
        info = {
            'flows': flows,
            'styles': styles,
            'diameters': diams
        }
        
        return masks, info
    
    def segment_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Segmenta un lote de imágenes."""
        results = []
        for img in images:
            mask, info = self.segment_image(img)
            results.append((mask, info))
        return results