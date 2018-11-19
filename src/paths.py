import os.path as osp
from os.path import join as ospj

from src.util import file_util

_prj_root_path = osp.dirname(osp.dirname(__file__))

_src_path = ospj(_prj_root_path, "src")
_data_path = ospj(_prj_root_path, "data")

_imgs_path = ospj(_data_path, "img")
_rapid_path = ospj(_data_path, "rapid")
_stroke_stuff_path = ospj(_data_path, "stroke")

vgg_weights_file = ospj(_data_path, "vgg_conv.pth")

stroke_images_path = ospj(_stroke_stuff_path, "Stroke_Imgs")
stroke_movements_npy_path = ospj(_stroke_stuff_path, "Stroke_Movements_Npy")
GA_stroke_npy_path = ospj(_stroke_stuff_path, "learn_strokes")

target_imgs_path = ospj(_imgs_path, "Target_Img")
simulated_paintings_imgs_path = ospj(_imgs_path, "Simulated_Paintings")
style_transferred_imgs_path = ospj(_imgs_path, "Style_Transferred")
style_imgs_path = ospj(_imgs_path, "Style_Img")

main_fn_command_seqs_path = ospj(_rapid_path, "command_sequences")
rapid_functions_path = ospj(_rapid_path, "Rapid_Functions", "Functions")
rapid_robtargets_path = ospj(_rapid_path, "Rapid_Functions", "Robtargets")
rapid_GA_path = ospj(_rapid_path, "GA")
paintings_path = ospj(_rapid_path, "Paintings")

for p in [_imgs_path,
          _rapid_path,
          _stroke_stuff_path,
          stroke_images_path,
          stroke_movements_npy_path,
          GA_stroke_npy_path,
          target_imgs_path,
          simulated_paintings_imgs_path,
          style_transferred_imgs_path,
          style_imgs_path,
          main_fn_command_seqs_path,
          rapid_functions_path,
          rapid_robtargets_path,
          rapid_GA_path,
          paintings_path]:
    file_util.create_dir_if_not_exist(p)
