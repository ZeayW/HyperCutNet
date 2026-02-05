import argparse


def get_args(args=None):
    parser = argparse.ArgumentParser()
    # 这里的 data_root 是包含多个 design/graph 的顶层目录
    parser.add_argument('--data_root', type=str, required=True, default='../rawdata',help='Root directory containing all graphs')
    parser.add_argument('--checkpoint', type=str, help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of graphs per batch')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()
    return args
