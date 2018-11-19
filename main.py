"""
Using this script, you can create your own NST & ISS simulations. The default parameters will produce a high-quality
A4 Painting of Sydney's Opera house painted in smART's style. The ABB RAPID code for our robot is also exported, but
you can find the simulated painting (from ISS) and the style transferred image (from NST) in data/img/ in their
respective folders.
"""
import argparse
import os
import time

import PIL.Image as Image
import torch

from src import paths
from src.iss.IterativeStrokeSampler import IterativeStrokeSampler
from src.rapid.RapidFunctionWriter import RapidFunctionWriter
from src.style_transfer.NeuralStyleTransfer import NeuralStyleTransfer
from src.util import img_util


def make_painting(args):
    if args.generate_rapid_fn:
        writer = RapidFunctionWriter(args=args)
        writer.write_all()

    if args.do_nst:
        if not args.is_already_style_transferred:
            start_time = time.time()

            _p = os.path.dirname(os.path.abspath(__file__))
            nst = NeuralStyleTransfer(
                content_img_path=os.path.join(paths.target_imgs_path, args.target_image_filename),
                style_img_path=os.path.join(paths.style_imgs_path, args.style_image_filename),
                output_img_path=os.path.join(_p, paths.style_transferred_imgs_path, args.name + ".jpg"),
                vgg_weights_path=paths.vgg_weights_file,
                n_iterations=args.nst_n_iterations,
                optimization_img_size=args.nst_optimization_img_size,
                use_gpu=args.use_gpu,
            )

            nst.run()
            print("Neural Style Transfer took: " + str(time.time() - start_time) + " seconds")

        iss_target_img = Image.open(os.path.join(paths.style_transferred_imgs_path, args.name + ".jpg"))
    else:
        iss_target_img = Image.open(os.path.join(paths.target_imgs_path, args.name + ".jpg"))
        iss_target_img.show()

    # _____________________________ potentially rescale target_image to desired metric size ____________________________
    iss_target_img = img_util.resize_to_mm(image=iss_target_img, px_per_mm=args.px_per_mm,
                                           max_mm_x=args.painting_size_x_mm, max_mm_y=args.painting_size_y_mm)

    iss = IterativeStrokeSampler(args=args, target_img=iss_target_img)

    if args.do_iss:
        start_time = time.time()
        cmd_seq, painting = iss.run(num_max_strokes_to_paint=args.n_iss_iters,
                                    num_stroke_samples_per_iter=args.n_samples_per_iss_iter,
                                    save=False)
        painting.show()
        print("ISS took: " + str(time.time() - start_time) + " seconds")

    else:  # import
        try:
            iss.env.load_state_from_disk()
            cmd_seq = iss.env.command_sequence
        except FileNotFoundError:
            print("Warning: Could not find simulated painting, so the RAPID code could not be generated. Try to enable"
                  "--do-iss to generate it.")
            return

    # ___________________________________________ generate final RAPID code ____________________________________________
    cmd_seq.write_rapid_code_to_file()
    print("Generated RAPID code.")


def get_args(name=None):
    parser = argparse.ArgumentParser(description="args for running NST and/or ISS")

    parser.add_argument("--target-image-filename", type=str, default=None,
                        help="Name (with file ending like .jpg!) of your target image. The image is expected to sit in"
                             " data/img/Target_Img")

    # _______________________________________________ Simulation Config ________________________________________________

    parser.add_argument("--do-nst", type=bool, default=True, help="Do you want to do Neural Style Transfer?")
    parser.add_argument("--is-already-style-transferred", type=bool, default=False,
                        help="Put this to True if you already did NST and just want to use the style transferred image"
                             "for ISS")
    parser.add_argument("--generate-rapid-fn", type=bool, default=True,
                        help="Generate RAPID support and stroke function code?")
    parser.add_argument("--do-iss", type=bool, default=True, help="Do you want to do Iterative Stroke Sampling?")

    parser.add_argument("--use_gpu", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--painting-size-x-mm", type=int, default=220, help="maximum pixel size of image on the X axis")
    parser.add_argument("--painting-size-y-mm", type=int, default=220, help="maximum pixel size of image on the Y axis")

    # _____________________ Neural Style Transfer
    parser.add_argument("--nst-n-iterations", type=int, help="Number of Neural Style Transfer Iterations",
                        default=1000)

    parser.add_argument("--nst-optimization-img-size", type=int,
                        help="Size of the image in the Neural Style Transfer process; GPUs with 4GB can do up to 400.",
                        default=400)

    parser.add_argument("--style-image-filename", type=str, help="Name (with file ending like .jpg!) of your NST style"
                                                                 "image. The image is expected to sit in"
                                                                 "'data/img/Target_Img'",
                        default="smART_style.jpg")

    # _____________________ Iterative Stroke Sampling
    parser.add_argument("--n-iss-iters", type=int, default=8000, help="Num of iterations of Iterative Stroke Sampling")
    parser.add_argument("--n-samples-per-iss-iter", type=int, default=700, help="Num of strokes to try per ISS iter")

    parser.add_argument("--max-batch-size-iss", type=int, default=300,
                        help="Max number of strokes to sample in a batch. Make this as high as your (v)RAM allows.")

    parser.add_argument("--n_strokes_between_getting_color", type=int, default=1,
                        help="Hyperparameter. Only relevant for physical setup")
    parser.add_argument("--min-n-strokes-with-same-brush-in-a-row", type=int, default=50,
                        help="Hyperparameter. Only relevant for physical setup")
    parser.add_argument("--min-n-strokes-with-same-color-in-a-row", type=int, default=50,
                        help="Hyperparameter. Only relevant for physical setup")

    parser.add_argument("--min-segmentation-width-px", type=int, default=250,
                        help="Hyperparameter. The min size of the subspace of the image ISS tries strokes in each iter")
    parser.add_argument("--error-threshold", type=int, default=-1000,
                        help="Hyperparameter. The max error a stroke can have to be painted; worse strokes aren't.")
    parser.add_argument("--reconsider-threshold", type=int, default=6,
                        help="After this number of iterations in a row where no stroke was over the error threshold,"
                             "color and brush are reconsidered")

    # _____________________ Painting Simulation
    parser.add_argument("--max-batch-size-env", type=int, default=100,
                        help="Max number of strokes to sample in a batch. Make this as high as your (v)RAM allows.")
    parser.add_argument("--paper-brightness", type=int, default=15, help="0 is white, 255 is black")
    parser.add_argument("--opacity-from-dark-to-bright", type=float, default=0.6,
                        help="0 is transparent. 1 is 100% opacity")
    parser.add_argument("--opacity-from-bright-to-dark", type=float, default=0.7,
                        help="0 is transparent. 1 is 100% opacity")

    parser.add_argument("--overpaint-punishment-threshold", type=int, default=2,
                        help=" How many 100% overpaints of a pixel are OK? Setting this to i.e. 2 means 4x"
                             "overpainting with 50 opacity goes unpunished")
    parser.add_argument("--overpaint-punish-factor", type=float, default=0.04,
                        help="This number regulates how much overpainting over the threshold is punished")

    # _____________________________________________ Physical Setup Config ______________________________________________
    parser.add_argument("--px-per-mm", type=int, default=8,
                        help="Digital to Analog resolution as pixels per mm?")

    parser.add_argument("--n-stroke-rotations", type=int, default=32,
                        help="How many rotational variations of each stroke do you want?")

    parser.add_argument("--stroke-size-mm", type=float, default=17.5,
                        help="How big are your stroke images in the real world in millimeters?")

    # _______________________________________________ RAPID code Config ________________________________________________
    parser.add_argument("--generate-io-calls-in-rapid", type=bool, default=True)
    parser.add_argument("--change-brush-func-name-prefix", type=str, default="ChBr")
    parser.add_argument("--mount-brush-func-name-prefix", type=str, default="Mount_Brush")
    parser.add_argument("--unmount-brush-func-name-prefix", type=str, default="Unmount_Brush")
    parser.add_argument("--clean-func-name-prefix", type=str, default="Clean")
    parser.add_argument("--towel-func-name-prefix", type=str, default="Towel")
    parser.add_argument("--getcolor-func-name-prefix", type=str, default="get")

    args = parser.parse_args()

    if args.target_image_filename is None:
        if name is None:
            raise ValueError("No argument for target-image-filename was not provided.")
        else:
            args.target_image_filename = name
    args.name = os.path.splitext(args.target_image_filename)[0]

    return args


if __name__ == "__main__":
    make_painting(args=get_args())
