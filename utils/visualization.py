# 5. utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Optional, Tuple, List
import yaml

class Visualizer:
    def __init__(self, config_path: str):
        """Inicializa el visualizador con la configuración."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def overlay_mask_on_image(self, image: np.ndarray, mask: np.ndarray, 
                              color: Tuple[int, int, int] = (255, 0, 0),
                              alpha: float = 0.5) -> np.ndarray:
        """
        Superpone la máscara de segmentación en la imagen original.
        
        Args:
            image: Imagen original
            mask: Máscara de segmentación donde cada célula tiene un ID único
            color: Color RGB para los contornos
            alpha: Transparencia de la superposición
            
        Returns:
            Imagen con contornos superpuestos
        """
        # Asegurarse de que la imagen esté en RGB
        if len(image.shape) == 2 or image.shape[2] == 1:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        
        # Crear imagen para superposición
        overlay = np.zeros_like(image_rgb)
        
        # Encontrar contornos de la máscara
        mask_binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dibujar contornos
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        # Combinar imagen original con overlay
        output = cv2.addWeighted(image_rgb, 1, overlay, alpha, 0)
        
        return output
    
    def create_segmentation_figure(self, image: np.ndarray, mask: np.ndarray, 
                                   save_path: Optional[str] = None) -> None:
        """
        Crea una figura con la imagen original, la máscara y la superposición.
        
        Args:
            image: Imagen original
            mask: Máscara de segmentación
            save_path: Ruta para guardar la figura (opcional)
        """
        # Convertir imagen a RGB si es necesario
        if len(image.shape) == 2:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        
        # Crear figura
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Imagen original
        ax[0].imshow(img_rgb)
        ax[0].set_title('Imagen Original')
        ax[0].axis('off')
        
        # Máscara de segmentación
        # Uso de viridis para que cada célula se vea con un color diferente
        mask_display = ax[1].imshow(mask, cmap='viridis')
        ax[1].set_title('Máscara de Segmentación')
        ax[1].axis('off')
        plt.colorbar(mask_display, ax=ax[1], fraction=0.046, pad=0.04)
        
        # Superposición
        overlay = self.overlay_mask_on_image(image, mask)
        ax[2].imshow(overlay)
        ax[2].set_title('Superposición')
        ax[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
