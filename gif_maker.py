from PIL import Image
from moviepy.editor import ImageSequenceClip
import re
import os


def filename_extractor(foldername):
    directory = os.fsencode(foldername)
    file_list = []
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if re.match("Maxt0_[0-9]{2}_[0-9]{3}_composite.jpg", filename):
            file_list.append(filename)

    file_list.sort()
    dict_res = {}

    for file in file_list:
        num = file[6:8]
        if num in dict_res:
            dict_res[num].append(foldername+"/"+file)
        else:
            dict_res[num] = []
            dict_res[num].append(foldername+"/"+file)

    return dict_res


def create_mp4(image_paths, output_path, fps=7):
    clip = ImageSequenceClip(image_paths, fps=fps)

    clip.write_videofile(output_path, fps=fps, codec="libx264")
    print(f"MP4 created and saved at {output_path}")


if __name__ == "__main__":
    ordered_dict = filename_extractor("output_segmentations")

    for key in ordered_dict:
        output_gif_path = key+"_secuence.mp4"
        create_mp4(ordered_dict[key], output_gif_path)

