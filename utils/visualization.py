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
                            n_frames: int = 5,
                            fps: int = 2,
                            flow_amplification: float = 5.0) -> None:
        """
        Genera una animación GIF/MP4 donde los píxeles de cada célula
        se desplazan hacia el centro de la célula, mostrando las posiciones iniciales
        en el primer frame y moviéndose hacia el centro en los frames siguientes.
        """

        H, W = dy.shape
        Y, X = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pos_y = Y.astype(np.float32)
        pos_x = X.astype(np.float32)

        step_size = 1.0 / n_frames
        mask_ids = np.unique(mask)
        mask_ids = mask_ids[mask_ids != 0]  # Ignora fondo

        # Calcular los centros de las células
        cell_centers = {}
        for mid in mask_ids:
            m = (mask == mid)
            center_y = np.mean(np.where(m)[0])  # Media de las posiciones en Y
            center_x = np.mean(np.where(m)[1])  # Media de las posiciones en X
            cell_centers[mid] = (center_y, center_x)

        # Simular todos los pasos hacia el centro de la célula
        pos_ys = []
        pos_xs = []

        for _ in range(n_frames):
            for mid in mask_ids:
                m = (mask == mid)
                center_y, center_x = cell_centers[mid]

                # Calcular la dirección hacia el centro
                direction_y = center_y - pos_y[m]
                direction_x = center_x - pos_x[m]

                # Normalizar la dirección y amplificar el movimiento
                norm = np.sqrt(direction_y**2 + direction_x**2)
                direction_y /= norm
                direction_x /= norm

                # Actualizar las posiciones de los píxeles
                pos_y[m] += step_size * direction_y * flow_amplification
                pos_x[m] += step_size * direction_x * flow_amplification

            pos_ys.append(pos_y.copy())
            pos_xs.append(pos_x.copy())

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis('off')

        def update(frame):
            ax.clear()
            ax.axis('off')

            # Crear la imagen de fondo
            frame_img = np.dstack([image] * 3) if image.ndim == 2 else image.copy()
            frame_img = frame_img.astype(np.float32)
            frame_img /= frame_img.max()
            frame_img = (255 * frame_img).astype(np.uint8)

            # Obtener las posiciones actuales de los píxeles
            py = pos_ys[frame]
            px = pos_xs[frame]

            for mid in mask_ids:
                m = (mask == mid)

                # Obtener las posiciones de los píxeles
                ys = np.clip(np.round(py[m]).astype(int), 0, H - 1)
                xs = np.clip(np.round(px[m]).astype(int), 0, W - 1)

                if frame == 0:  # Primer frame: mostrar las posiciones iniciales en rojo
                    frame_img[ys, xs] = [255, 0, 0]  # Rojo para posiciones iniciales
                else:  # En los siguientes frames, mover hacia el centro
                    frame_img[ys, xs] = [255, 0, 0]  # Rojo para mostrar el movimiento

            ax.imshow(frame_img)
            return []

        ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.gif':
            ani.save(save_path, writer='pillow', fps=fps)
        else:
            ani.save(save_path, fps=fps, codec='libx264')

        plt.close(fig)

    def create_metrics_image(self, mask_path: str, gt_path: str, save_path: Optional[str] = None) -> Optional[str]:
        """
        Crea una imagen que muestra gráficamente las métricas que evaluan

        Args:
            mask_pah: Ruta de la máscara
            gt_path: Ruta del gt
            save_path: Ruta para guardar la imagen resultante (opcional).

        Returns:
            Ruta donde se guardó la imagen si se guarda; None si no se guarda.
        """
        img1 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # Convertir a blanco y negro: todo >10 se vuelve 255 (blanco), el resto 0 (negro)
        _, img1 = cv2.threshold(img1, 10, 255, cv2.THRESH_BINARY)
        img2 = cv2.imread(gt_path, 0)
        _, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

        if img1 is None or img2 is None:
            raise FileNotFoundError("Una de las imágenes no se pudo cargar.")

        if img1.shape != img2.shape:
            raise ValueError("Las imágenes deben tener el mismo tamaño.")

        resultado = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)

        # Verde fuerte para coincidencias
        resultado[(img1 == 255) & (img2 == 255)] = [0, 255, 0]

        # Rojo para los falsos positivos
        resultado[(img1 == 255) & (img2 == 0)] = [0, 0, 255]

        # Azul para los verdaderos negativos
        resultado[(img1 == 0) & (img2 == 255)] = [255, 0, 0]

        # Añadir la leyenda a la imagen
        leyenda = np.zeros((100, resultado.shape[1], 3), dtype=np.uint8)  # Espacio para la leyenda
        cv2.putText(leyenda, "Verde: Coincidencias (True Positives)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(leyenda, "Rojo: Falsos Positivos", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(leyenda, "Azul: Falsos Negativos", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Concatenar la leyenda con la imagen de resultado
        resultado_con_leyenda = np.vstack((resultado, leyenda))

        # Determinar ruta de guardado si no se proporciona
        if save_path is None:
            dir_base = os.path.dirname(mask_path)
            nombre_base = os.path.splitext(os.path.basename(mask_path))[0]
            save_path = os.path.join(dir_base, f"{nombre_base}_metrics_image.png")

        cv2.imwrite(save_path, resultado_con_leyenda)
        return save_path