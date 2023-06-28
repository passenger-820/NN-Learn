import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, default='dcgan_1')

    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches") # For fast training
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate") # For fast training
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--imsize", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")

    parser.add_argument('--g_num', type=int, default=5)
    # parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--z_dim', type=int, default=128, help='length of noise')
    # parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument('--total_step', type=int, default=100000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--g_lr', type=float, default=0.0001) # 0.0001
    parser.add_argument('--d_lr', type=float, default=0.0005) # 0.0004
    # parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='animefaces')
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    # parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=1.0)

    return parser.parse_args()