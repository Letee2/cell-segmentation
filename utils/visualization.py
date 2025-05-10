# 5. utils/visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Optional, Tuple, List
import matplotlib.animation as animation
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

    def create_flowmap_figure(self, dy, dx, save_path: Optional[str] = None) -> None:
            """
            Crea y guarda una visualización del mapa de vectores.
            
            Args:
                flow_map: Array de shape (2, H, W) con vectores X e Y
                save_path: Ruta para guardar la imagen
            """

            step = 5 # Flechas cada 5 píxeles
            H, W = dy.shape

            Y, X = np.mgrid[0:H:step, 0:W:step] 
            U = dx[::step, ::step]  
            V = dy[::step, ::step]  

            plt.figure(figsize=(8, 8))
            plt.quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=1)
            plt.axis('off')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()


    def create_flow_animation(self,
                            image: np.ndarray,
                            dy: np.ndarray,
                            dx: np.ndarray,
                            mask: np.ndarray,
                            save_path: str,
                            n_frames: int = 20,
                            fps: int = 5,
                            flow_amplification: float = 2.0) -> None:
        """
        Genera una animación GIF/MP4 donde los píxeles de cada célula
        se desplazan siguiendo los vectores, pero se reproduce en reversa
        para simular flujo hacia el centro.
        """

        H, W = dy.shape
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pos_y = Y.astype(np.float32)
        pos_x = X.astype(np.float32)

        step_size = 1.0 / n_frames
        mask_ids = np.unique(mask)
        mask_ids = mask_ids[mask_ids != 0]  # Ignora fondo

        # Simular todos los pasos hacia adelante
        pos_ys = []
        pos_xs = []

        for _ in range(n_frames):
            for mid in mask_ids:
                m = (mask == mid)
                pos_y[m] += step_size * dy[m] * flow_amplification
                pos_x[m] += step_size * dx[m] * flow_amplification
            pos_ys.append(pos_y.copy())
            pos_xs.append(pos_x.copy())

        # Invertir el orden de los pasos
        pos_ys = pos_ys[::-1]
        pos_xs = pos_xs[::-1]

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis('off')

        def update(frame):
            ax.clear()
            ax.axis('off')

            frame_img = np.dstack([image] * 3) if image.ndim == 2 else image.copy()
            frame_img = frame_img.astype(np.float32)
            frame_img /= frame_img.max()
            frame_img = (255 * frame_img).astype(np.uint8)

            py = pos_ys[frame]
            px = pos_xs[frame]

            for mid in mask_ids:
                m = (mask == mid)
                ys = np.clip(np.round(py[m]).astype(int), 0, H - 1)
                xs = np.clip(np.round(px[m]).astype(int), 0, W - 1)
                frame_img[ys, xs] = [255, 0, 0]  # Rojo

            ax.imshow(frame_img)
            return []

        ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.gif':
            ani.save(save_path, writer='pillow', fps=fps)
        else:
            ani.save(save_path, fps=fps, codec='libx264')

        plt.close(fig)
