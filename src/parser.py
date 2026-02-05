import dgl
import torch as th
import os
import numpy as np
from collections import defaultdict


def buildGraph(graph_path):
    print(f"Parsing graph: {graph_path}")

    # -------------------------------------------------------
    # 1. 读取 Pin 节点特征 (P512_nodes.txt)
    # -------------------------------------------------------
    # 格式: nid, weight, degree
    pin_feats_list = []
    nidMap = {}

    nodes_file = os.path.join(graph_path, "nodes.txt")
    if os.path.exists(nodes_file):
        with open(nodes_file, 'r') as f:
            for idx, line in enumerate(f):
                parts = line.strip().split(' ')
                if not parts: continue

                original_id = int(parts[0])
                weight = float(parts[1])
                degree = float(parts[2])
                nidMap[original_id] = idx
                pin_feats_list.append([weight, degree])
    else:
        print(f"Error: {nodes_file} not found.")
        return None, None

    # 转为 Tensor (N_pin, 3)
    pin_feat_tensor = th.tensor(pin_feats_list, dtype=th.float32)

    # -------------------------------------------------------
    # 2. 读取 Net、建立连接并提取 Label (P512_hedges.txt)
    # -------------------------------------------------------
    # 格式: net_id; pin_id1 pin_id2 ...; connectivity

    p2n_src = []
    p2n_dst = []
    net_labels = []  # 用于存储 connectivity label
    original_net_ids = []  # 用于存储原始 net id

    # 辅助倒排索引
    pin_to_nets = defaultdict(list)

    edges_file = os.path.join(graph_path, "hedges.txt")
    if os.path.exists(edges_file):
        with open(edges_file, 'r') as f:
            lines = f.readlines()

        # 注意：这里我们假设文件中的行顺序即为 Net 在图中的 ID (0 ~ num_nets-1)
        for net_idx, line in enumerate(lines):
            # 分割为三部分: ID; Pins; Label
            parts = line.strip().split(';')

            if len(parts) < 3:
                print(f"Warning: Line {net_idx} format error, skipped.")
                continue

            # (1) 解析 Net ID
            orig_net_id = int(parts[0])
            original_net_ids.append(orig_net_id)

            # (2) 解析 Pins
            node_list_raw = parts[1].strip().split()
            for n_str in node_list_raw:
                if not n_str: continue
                raw_nid = int(n_str)
                if raw_nid in nidMap:
                    pin_idx = nidMap[raw_nid]
                    p2n_src.append(pin_idx)
                    p2n_dst.append(net_idx)
                    pin_to_nets[pin_idx].append(net_idx)

            # (3) 解析 Label (Connectivity)
            try:
                label_val = float(parts[2])
                net_labels.append(label_val)
            except ValueError:
                print(f"Warning: Invalid label at line {net_idx}, set to 0.")
                net_labels.append(0.0)

    else:
        print(f"Error: {edges_file} not found.")
        return None, None

    # -------------------------------------------------------
    # 4. 计算 Net-Net 重叠边
    # -------------------------------------------------------
    net_overlaps = defaultdict(int)
    HUGE_NET_THRESHOLD = 100

    for pin_idx, connected_nets in pin_to_nets.items():
        if len(connected_nets) > HUGE_NET_THRESHOLD: continue
        if len(connected_nets) > 1:
            for i in range(len(connected_nets)):
                u = connected_nets[i]
                for j in range(i + 1, len(connected_nets)):
                    v = connected_nets[j]
                    net_overlaps[(u, v)] += 1
                    net_overlaps[(v, u)] += 1

    n2n_src = []
    n2n_dst = []
    n2n_weight = []
    for (u, v), w in net_overlaps.items():
        n2n_src.append(u)
        n2n_dst.append(v)
        n2n_weight.append(w)

    # -------------------------------------------------------
    # 5. 构建 DGL Heterograph
    # -------------------------------------------------------
    t_p2n_src = th.tensor(p2n_src, dtype=th.int64)
    t_p2n_dst = th.tensor(p2n_dst, dtype=th.int64)

    graph_data = {
        ('pin', 'connected', 'net'): (t_p2n_src, t_p2n_dst),
        ('net', 'connected', 'pin'): (t_p2n_dst, t_p2n_src),
        ('net', 'overlap', 'net'): (th.tensor(n2n_src, dtype=th.int64),
                                    th.tensor(n2n_dst, dtype=th.int64))
    }

    g = dgl.heterograph(graph_data)

    # -------------------------------------------------------
    # 6. 赋予特征与标签
    # -------------------------------------------------------
    # Pin 特征
    g.nodes['pin'].data['feat'] = pin_feat_tensor

    # Net 标签 (connectivity) -> 形状 (Num_Nets, 1)
    g.nodes['net'].data['label'] = th.tensor(net_labels, dtype=th.float32).reshape(-1, 1)

    # 原始 Net ID (可选，用于后续调试回溯)
    g.nodes['net'].data['id'] = th.tensor(original_net_ids, dtype=th.int32).reshape(-1, 1)

    # Net-Net 边权重
    if len(n2n_weight) > 0:
        g.edges['overlap'].data['weight'] = th.tensor(n2n_weight, dtype=th.float32).reshape(-1, 1)

    print(f'\t--- Graph built: {g.num_nodes("pin")} Pins, {g.num_nodes("net")} Nets ---')
    print(f'\tNet Labels shape: {g.nodes["net"].data["label"].shape}')

    # 返回 graph
    return g
