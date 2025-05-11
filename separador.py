import cv2
import numpy as np
import os
from tifffile import TiffFile


def procesar_imagenes():
    new_foldername = f"separadas"
    os.makedirs(new_foldername, exist_ok=True)

    for i in range(64):

        if i<=9:
            filename = f"MAX230426_bCatTracking_Exp91_20230426_73604 AM_f000{i}_t0000.tif"
            outname = f"Maxt0_0{i}_"
        else:
            filename = f"MAX230426_bCatTracking_Exp91_20230426_73604 AM_f00{i}_t0000.tif"
            outname = f"Maxt0_{i}_"

        input_path = os.path.join("original", filename)
        if not os.path.exists(input_path):
            print(f"Archivo no encontrado: {input_path}")
            continue

        with TiffFile(input_path) as tif:
            for j, page in enumerate(tif.pages):
                imagen = page.asarray()

                if j<10:
                    final_name = f"{outname}00{j}_a.jpg"
                elif j<100:
                    final_name = f"{outname}0{j}_a.jpg"
                else:
                    final_name = f"{outname}{j}_a.jpg"

                output_path = os.path.join(new_foldername, final_name)
                cv2.imwrite(output_path, imagen)

        print(f"Imagen procesada: {i}")


if __name__ == "__main__":

    procesar_imagenes()
