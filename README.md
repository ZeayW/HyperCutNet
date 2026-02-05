# HyperCutNet: Net Cut Connectivity Prediction

This project implements a **multi-stage Heterogeneous Graph Neural Network (GNN)** to predict the connectivity or cut probability of nets in graphs.

## 🚀 Key Features

*   **Heterogeneous Graph Modeling**: Constructs a DGL graph with `Pin` and `Net` nodes, supporting:
    *   `Pin -> Net` edges (membership)
    *   `Net -> Pin` edges (feedback)
    *   `Net <-> Net` edges (weighted overlap)
*   **3-Stage Sequential GNN**: A novel message passing scheme designed specifically for circuit logic:
    1.  **Upload**: Nets aggregate features from connected Pins.
    2.  **Propagate**: Nets exchange information with neighboring Nets (weighted by overlap count).
    3.  **Feedback**: Pins update their states based on the refined Net embeddings.

## 📂 Project Structure

```text
.
├── rawdata/                       # Root directory for graph data
│   ├── design_A/               # Example Design
│   │   ├── P512_nodes.txt      # Pin features (id, weight, degree...)
│   │   └── P512_hedges.txt     # Net connectivity & labels
│   └── ...
├── src/                        # Source code
│   ├── hypergraph_generator.py # Data Parser: Converts text files to DGL Graphs
│   ├── model.py                # GNN Architecture (ThreeStageGNNLayer)
│   └── train.py                # Training loop, Batching, and Evaluation
├── environment.yml             # Conda environment configuration
└── README.md
```

## 🛠️ Installation

This project requires **PyTorch** and **DGL (Deep Graph Library)**. We recommend using Conda to manage dependencies.

### Option 1: Conda

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/HyperCutNet.git
    cd HyperCutNet
    ```

2.  **Create the environment**
    You can create the environment using the provided `environment.yml`. This setup defaults to **CUDA 12.1**.
    ```bash
    conda env create -f environment.yml
    conda activate gnn_design
    ```

### Option 2: Manual Install

If you prefer pip or use a different CUDA version (e.g., CPU-only or CUDA 11.8), follow these steps:

1.  **Install PyTorch**
    Check [pytorch.org](https://pytorch.org/get-started/locally/) for your specific version.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **Install DGL**
    Check [dgl.ai](https://www.dgl.ai/pages/start.html) for your specific version.
    ```bash
    pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
    ```

3.  **Install other dependencies**
    ```bash
    pip install numpy scipy tqdm
    ```

## 🏃‍♂️ Usage

### 1. Prepare Data
Ensure your data follows this structure inside each block folder (e.g., `data/design_A/`):

*   **`nodes.txt`**: Each line contains node features:
    ```text
    node_id weight in_degree out_degree
    ```
*   **`hedges.txt`**: Each line contains net information and the target label:
    ```text
    net_id; pin_id_1 pin_id_2 ...; label
    ```
    *Note: The `label` is the regression target (e.g., connectivity or cut probability).*

### 2. Run Training
Run the training script from the **root** directory of the project. The script automatically handles graph construction, batching, and dataset splitting.

```bash
python src/train.py --data_root ./rawdata --checkpoint try1 --batch_size 4 --layers 3 --hidden_dim 128 --epochs 200 --lr 0.001
```
