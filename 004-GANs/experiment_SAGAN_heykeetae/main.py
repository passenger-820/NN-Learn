from parameter import *
from trainer import Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder

import matplotlib.pyplot as plt

def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader（True, celeba, ./data, 64, 64, True）
    data_loader = Data_Loader(config.train, config.dataset, config.image_path, config.imsize,
                             config.batch_size, shuf=config.train)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    trainer = Trainer(data_loader.loader(), config)
    trainer.train()


if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)