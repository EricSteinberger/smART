from main import make_painting, get_args

if __name__ == '__main__':
    args = get_args(name="test.jpg")
    args.n_iss_iters = 100
    args.n_samples_per_iss_iter = 50
    args.nst_n_iterations = 150
    args.painting_size_x_mm = 120
    args.painting_size_y_mm = 120
    make_painting(args=args)
