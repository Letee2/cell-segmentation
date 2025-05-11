import cv2
import numpy as np
import os
from tifffile import TiffFile


def acentuar(img, max_val):
    """ Aumenta el brillo de la imagen normalizando los valores a 255 """
    return np.clip((img.astype(np.float32) * 255 / max_val), 0, 255).astype(np.uint8)


def intensificar_verde(image):
    return np.where(cv2.medianBlur(image, 3) >= 200, 255, 0).astype(np.uint8)


def ruido_deleter(image):
    return np.where(image < 70, 0, cv2.medianBlur(image, 9)*2).astype(np.uint8)


def procesar_imagenes(max_value):
    new_foldername = f"MASK_{max_value}"
    os.makedirs(new_foldername, exist_ok=True)

    for i in range(64):

        # Modificar nombres si se generan nuevas (hasta antes de f00...)
        patron = "MAX230426_bCatTracking_Exp91_20230426_73604 AM_"
        if i<=9:
            filename = f"{patron}f000{i}_t0000.tif"
            outname = f"MASK_0{i}_"
        else:
            filename = f"{patron}f00{i}_t0000.tif"
            outname = f"MASK_{i}_"

        input_path = os.path.join("original", filename)
        if not os.path.exists(input_path):
            print(f"Archivo no encontrado: {input_path}")
            continue

        with TiffFile(input_path) as tif:
            for j, page in enumerate(tif.pages):
                imagen = page.asarray()
                b, g, r = cv2.split(imagen)

                kernel = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8)

                g_1 = acentuar(g, max_value)
                g_2 = ruido_deleter(g_1)

                g_2 = np.where(g_2.astype(np.float32) + g_1.astype(np.float32) >= 200, 255, 0).astype(np.uint8)

                g_2 = cv2.morphologyEx(g_2, cv2.MORPH_CLOSE, kernel, iterations = 1, borderType = cv2.BORDER_CONSTANT)
                g_2 = cv2.morphologyEx(g_2, cv2.MORPH_OPEN, kernel, iterations = 1, borderType = cv2.BORDER_CONSTANT)

                g_2 = intensificar_verde(g_2)


                if j<10:
                    final_name = f"{outname}00{j}_g.png"
                elif j<100:
                    final_name = f"{outname}0{j}_g.png"
                else:
                    final_name = f"{outname}{j}_g.png"

                output_path = os.path.join(new_foldername, final_name)
                cv2.imwrite(output_path, g_2)

        print(f"Imagen procesada: {i}")


if __name__ == "__main__":
    print("El umbral actual es de 400, pulsar enter si se quiere mantener, escribir otro umbral en caso contrario.")
    max_value = input("Valor del umbral máximo: ").strip()
    if max_value == "":
        max_value = 400
    else:
        max_value = int(max_value)
    procesar_imagenes(max_value)
