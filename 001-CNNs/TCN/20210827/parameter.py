import argparse

def get_parameters():

    parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')

    parser.add_argument('--root', type=str, default=r'F:\PycharmProjects\NeuralNetwork\data\mnist', metavar='N',
                        help='root of mnist')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size (default: 64)')
    parser.add_argument('--cuda', action='store_false',
                        help='use CUDA (default: True)')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='dropout applied to layers (default: 0.05)')
    parser.add_argument('--clip', type=float, default=-1,
                        help='gradient clip, -1 means no clip (default: -1)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit (default: 20)')
    parser.add_argument('--ksize', type=int, default=7,
                        help='kernel size (default: 7)')
    parser.add_argument('--levels', type=int, default=8,
                        help='# of levels (default: 8)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval (default: 100')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='initial learning rate (default: 2e-3)')
    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--nhid', type=int, default=25,
                        help='number of hidden units per layer (default: 25)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed (default: 1111)')
    parser.add_argument('--permute', action='store_true',
                        help='use permuted MNIST (default: false)')

    return parser.parse_args()