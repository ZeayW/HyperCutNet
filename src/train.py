import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from dgl.dataloading import GraphDataLoader
import dgl
import os
import argparse
import glob
import random
import numpy
import tee

from parser import buildGraph
from model import NetPredictor
from args import get_args


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
args = get_args()

def init(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

def init_model(args):
    model = NetPredictor(
        pin_in_dim=2,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        n_layers=args.layers
    ).to(device)

    return model

class GraphDataset(dgl.data.DGLDataset):
    """
    自定义数据集：扫描指定目录下的所有子文件夹，将每个合法的 graph 文件夹加载为一个图。
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.graphs = []
        super().__init__(name='GraphDataset')

    def process(self):
        # 1. 递归查找所有包含 'nodes.txt' 的目录，视为一个 graph
        # 使用 glob 查找所有子目录
        print(f"Scanning {self.root_dir} for graphs...")
        search_path = os.path.join(self.root_dir, "**", "nodes.txt")
        found_files = glob.glob(search_path, recursive=True)

        graph_paths = [os.path.dirname(f) for f in found_files]
        print(f"Found {len(graph_paths)} graphs.")

        for path in graph_paths:
            try:
                # 调用 buildGraph
                g = buildGraph(path)
                if g is not None:
                    self.graphs.append(g)
            except Exception as e:
                print(f"Failed to load graph {path}: {e}")

        print(f"Successfully loaded {len(self.graphs)} graphs.")

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


def collate_fn(graphs):
    """
    将一个 batch 的图列表合并为一个 Batched Graph
    """
    return dgl.batch(graphs)


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0
    total_mae = 0
    num_batches = 0

    with torch.no_grad():
        for batched_g in dataloader:
            batched_g = batched_g.to(device)

            # 准备特征
            pin_feat = batched_g.nodes['pin'].data['feat']
            # 简单的 Batch 内归一化
            mean = pin_feat.mean(dim=0, keepdim=True)
            std = pin_feat.std(dim=0, keepdim=True)
            pin_feat = (pin_feat - mean) / (std + 1e-6)

            # 准备权重
            if 'weight' in batched_g.edges['overlap'].data:
                overlap_weights = batched_g.edges['overlap'].data['weight']
                overlap_weights = torch.log(overlap_weights + 1.0)
            else:
                overlap_weights = None

            # 准备标签
            labels = batched_g.nodes['net'].data['label']

            # Forward
            logits, _ = model(batched_g, pin_feat, overlap_weights)

            loss = loss_fn(logits, labels)
            mae = torch.nn.L1Loss()(logits, labels)

            total_loss += loss.item()
            total_mae += mae.item()
            num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def train(args,model):
    # -------------------------------------------------------------
    # 1. 加载所有图数据
    # -------------------------------------------------------------
    dataset = GraphDataset(args.data_root)

    if len(dataset) == 0:
        print("No graphs found. Exiting.")
        return

    # -------------------------------------------------------------
    # 2. 按图粒度划分 (Graph-Level Split)
    # -------------------------------------------------------------
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = max(1,int(0.1 * total_size))
    test_size = max(1,total_size - train_size - val_size)
    train_size = total_size - val_size - test_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset Split: Train {len(train_set)} graphs, Val {len(val_set)} graphs, Test {len(test_set)} graphs")

    # -------------------------------------------------------------
    # 3. 创建 DataLoader (Batching)
    # -------------------------------------------------------------
    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = GraphDataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = GraphDataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # -------------------------------------------------------------
    # 4. 训练循环
    # -------------------------------------------------------------
    best_val_mae = float('inf')

    print("\n--- Start Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0

        for batch_g in train_loader:
            batch_g = batch_g.to(device)

            # --- 特征处理 ---
            # 从 Batched Graph 中提取特征
            # 注意: 这里的 feat 是这一批次所有图中 pin 的特征拼接
            pin_feat = batch_g.nodes['pin'].data['feat']

            # 归一化 (Z-Score)
            mean = pin_feat.mean(dim=0, keepdim=True)
            std = pin_feat.std(dim=0, keepdim=True)
            pin_feat = (pin_feat - mean) / (std + 1e-6)

            # 边权重处理
            if 'weight' in batch_g.edges['overlap'].data:
                overlap_weights = batch_g.edges['overlap'].data['weight']
                overlap_weights = torch.log(overlap_weights + 1.0)
            else:
                overlap_weights = None

            # 标签
            labels = batch_g.nodes['net'].data['label']

            # --- Forward & Backward ---
            logits, _ = model(batch_g, pin_feat, overlap_weights)

            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation ---
        if epoch % 5 == 0:
            val_loss, val_mae = evaluate(model, val_loader, device, loss_fn)
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val MAE: {val_mae:.4f}")

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), os.path.join('../checkpoints',args.checkpoint,"best_model_graph_split.pth"))

    # -------------------------------------------------------------
    # 5. 测试
    # -------------------------------------------------------------
    print("\n--- Final Testing ---")
    model.load_state_dict(torch.load("best_model_graph_split.pth"))
    _, test_mae = evaluate(model, test_loader, device, loss_fn)
    print(f"Best Test MAE (Unseen Graphs): {test_mae:.4f}")


if __name__ == "__main__":
    seed = random.randint(1, 10000)
    init(seed)
    args = get_args()

    if args.checkpoint:
        print('Saving logs and models to ../checkpoints/{}'.format(args.checkpoint))
        # 创建模型保存目录并存储args
        checkpoint_path = '../checkpoints/{}'.format(args.checkpoint)
        os.makedirs(checkpoint_path)  # exist not ok
        torch.save(args, os.path.join(checkpoint_path, 'args.pkl'))
        # 模型初始化
        model = init_model(args)
        stdout_f = '../checkpoints/{}/stdout.log'.format(args.checkpoint)
        with tee.StdoutTee(stdout_f):
            print('seed:',seed)
            train(args,model)
    else:
        print('No checkpoint is specified. abandoning all model checkpoints and logs')
        # 模型初始化
        model = init_model(args)
        train(args,model)
