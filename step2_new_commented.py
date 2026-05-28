# -*- coding: utf-8 -*-
"""
Train a dynamic SC -> FC baseline directly from per-subject CSV files.

Data layout expected by this script:
    SC/<subject_id>.csv
        Structural connectivity matrix, shape [regions, regions], no header.

    FC/AAL90/<subject_id>_AAL90_timeseries.csv
        Resting-state BOLD time series with a header. The first column is
        "timepoint"; the remaining columns are region signals.

This version uses a torch Dataset/DataLoader and does not read or write .npy
data caches. Dynamic FC labels are computed in memory from each subject's BOLD
time series.
"""

# 这份文件是 step2_new.py 的中文注释版：
# - 代码逻辑、变量名、函数签名、训练流程均保持不变。
# - 注释重点解释数据形状、模型结构、训练指标和每个步骤的作用。

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split


# 项目根目录：当前脚本所在文件夹。
PROJECT_DIR = Path(__file__).resolve().parent

# SC 数据目录：每个被试一个 CSV，内容是结构连接矩阵。
SC_DIR = PROJECT_DIR / "SC"

# FC/BOLD 时间序列目录：每个被试一个 AAL90 时间序列 CSV。
FC_DIR = PROJECT_DIR / "FC" / "AAL90"

# 输出目录：保存训练结果和最佳模型权重。
OUTPUT_DIR = PROJECT_DIR / "results_dynamic_gru_new"
MODEL_PATH = OUTPUT_DIR / "dynamic_gcn_gru_best.pt"


@dataclass
class Config:
    # 动态 FC 的滑动窗口长度。每个窗口内计算一次脑区相关矩阵。
    window_size: int = 83

    # 滑动窗口步长。stride 越小，生成的动态 FC 窗口越密集。
    stride: int = 2

    # Fisher z 变换前对相关系数做裁剪，避免 arctanh(1) 或 arctanh(-1) 得到无穷大。
    fisher_clip: float = 0.999999

    # 第一层 GCN 输出维度。
    gcn_hidden_dim: int = 128

    # SC 编码器最终输出的全局上下文维度。
    sc_context_dim: int = 64

    # GRU 解码器隐藏状态维度。
    gru_hidden_dim: int = 128

    # 训练超参数。
    learning_rate: float = 1e-3
    batch_size: int = 2
    epochs: int = 10
    early_stopping_patience: int = 3

    # Teacher forcing 比例：
    # 训练时有一定概率把真实上一窗口 FC 作为下一步输入，而不是用模型预测值。
    teacher_forcing_ratio: float = 0.5

    # 测试集比例。
    test_ratio: float = 0.2

    # 从训练+验证集合中再划出多少比例作为验证集。
    val_ratio_in_train: float = 0.2

    # 随机种子，用于数据划分和模型训练的可复现性。
    random_seed: int = 42

    # SC 矩阵阈值，小于该阈值的边会被置零。
    sc_threshold: float = 0.0

    # DataLoader 读取数据的子进程数量。Windows 上设为 0 更稳。
    num_workers: int = 0

    # 是否把已经读取并处理过的样本缓存在内存中，避免重复计算动态 FC。
    cache_samples_in_memory: bool = True


def preprocess_graph(sc: np.ndarray, sc_threshold: float) -> np.ndarray:
    # 复制并转成 float32，避免修改原始 SC 数组。
    sc = sc.astype(np.float32, copy=True)

    # 根据阈值去除较弱的结构连接。
    sc[sc < sc_threshold] = 0.0

    # 给邻接矩阵加自环，让每个节点在 GCN 中保留自身信息。
    adj_with_loop = sc + np.eye(sc.shape[0], dtype=np.float32)

    # 计算每个节点的度。
    degree = adj_with_loop.sum(axis=-1)

    # 防止孤立节点导致除零。
    degree[degree == 0] = 1.0

    # 计算 D^(-1/2)，用于对称归一化邻接矩阵。
    inverse_sqrt_degree = np.power(degree, -0.5).astype(np.float32)

    # 标准 GCN 归一化形式：D^(-1/2) A D^(-1/2)。
    normalized = inverse_sqrt_degree[:, None] * adj_with_loop * inverse_sqrt_degree[None, :]
    return normalized.astype(np.float32)


def upper_triangle_indices(num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    # 只取矩阵上三角且不包含对角线的位置。
    # FC 矩阵是对称的，所以只预测上三角边即可减少输出维度。
    return np.triu_indices(num_nodes, k=1)


def compute_dynamic_fc_edges(
    bold_timeseries: np.ndarray,
    window_size: int,
    stride: int,
    fisher_clip: float,
) -> np.ndarray:
    # 输入 BOLD 时间序列应为二维数组：[时间点数, 脑区数]。
    if bold_timeseries.ndim != 2:
        raise ValueError("BOLD time series must have shape [timepoints, regions].")

    # 窗口至少需要 2 个时间点才能计算相关；步长至少为 1。
    if window_size < 2 or stride < 1:
        raise ValueError("window_size must be >= 2 and stride must be >= 1.")

    num_timepoints, num_nodes = bold_timeseries.shape

    # 窗口不能比整段时间序列还长。
    if window_size > num_timepoints:
        raise ValueError("window_size is larger than the available BOLD timepoints.")

    # 准备上三角索引，后续把完整 FC 矩阵压缩成边向量。
    triu = upper_triangle_indices(num_nodes)
    windows = []

    # 按滑动窗口遍历 BOLD 时间序列。
    for start in range(0, num_timepoints - window_size + 1, stride):
        # 当前窗口的 BOLD 数据，形状：[window_size, num_nodes]。
        window = bold_timeseries[start : start + window_size]

        # 计算脑区之间的 Pearson 相关矩阵，形状：[num_nodes, num_nodes]。
        correlation = np.corrcoef(window, rowvar=False)

        # 处理常数序列或数值异常导致的 NaN/Inf。
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)

        # 裁剪相关系数，保证 Fisher z 变换稳定。
        correlation = np.clip(correlation, -fisher_clip, fisher_clip)

        # Fisher z 变换，使相关系数更接近正态分布，常用于 FC 分析。
        fisher_z_fc = np.arctanh(correlation)

        # 保存上三角边向量，而不是完整矩阵。
        windows.append(fisher_z_fc[triu].astype(np.float32))

    # 返回形状：[动态窗口数, 边数]。
    return np.stack(windows, axis=0).astype(np.float32)


class SCFCDataset(Dataset):
    def __init__(self, sc_dir: Path, fc_dir: Path, config: Config) -> None:
        self.sc_dir = sc_dir
        self.fc_dir = fc_dir
        self.config = config

        # 自动匹配有 SC 文件和 FC 时间序列文件的被试。
        self.samples = self._discover_samples()

        # 样本缓存：key 是样本索引，value 是处理后的张量三元组。
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        if not self.samples:
            raise FileNotFoundError(f"No matched SC/FC subjects found under {sc_dir} and {fc_dir}.")

        # 读取第一个样本，用它推断模型所需的维度信息。
        first_adj, first_features, first_targets = self[0]
        self.num_nodes = first_adj.shape[0]
        self.input_dim = first_features.shape[-1]
        self.num_windows = first_targets.shape[0]
        self.num_edges = first_targets.shape[1]

    def _discover_samples(self) -> list[tuple[str, Path, Path]]:
        # 建立 subject_id -> SC 文件路径 的映射。
        sc_paths = {path.stem: path for path in self.sc_dir.glob("*.csv")}
        fc_paths = {}

        # FC 文件名格式预期为：<subject_id>_AAL90_timeseries.csv。
        for path in self.fc_dir.glob("*_AAL90_timeseries.csv"):
            subject_id = path.name.split("_")[0]
            fc_paths[subject_id] = path

        # 只保留 SC 和 FC 都存在的被试。
        subject_ids = sorted(set(sc_paths) & set(fc_paths))

        # 输出缺失匹配文件的提醒，方便检查数据完整性。
        missing_fc = sorted(set(sc_paths) - set(fc_paths))
        missing_sc = sorted(set(fc_paths) - set(sc_paths))
        if missing_fc:
            print(f"Warning: {len(missing_fc)} SC files have no matching FC file.")
        if missing_sc:
            print(f"Warning: {len(missing_sc)} FC files have no matching SC file.")

        # 每个样本保存为：(被试 ID, SC 路径, FC 路径)。
        return [(subject_id, sc_paths[subject_id], fc_paths[subject_id]) for subject_id in subject_ids]

    def __len__(self) -> int:
        # Dataset 长度等于匹配到的被试数量。
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 如果开启缓存且当前样本已经处理过，直接返回缓存结果。
        if self.config.cache_samples_in_memory and index in self._cache:
            return self._cache[index]

        _, sc_path, fc_path = self.samples[index]

        # 读取 SC 矩阵，预期无表头，逗号分隔。
        sc = np.loadtxt(sc_path, delimiter=",", dtype=np.float32, encoding="utf-8-sig")

        # 读取 BOLD 时间序列，跳过表头。
        # 第一列是 timepoint，后续列才是脑区信号。
        bold_with_time = np.loadtxt(
            fc_path,
            delimiter=",",
            skiprows=1,
            dtype=np.float32,
            encoding="utf-8-sig",
        )
        bold = bold_with_time[:, 1:]

        # SC 必须是方阵：[脑区数, 脑区数]。
        if sc.ndim != 2 or sc.shape[0] != sc.shape[1]:
            raise ValueError(f"{sc_path} must be a square SC matrix.")

        # FC 时间序列中的脑区数必须与 SC 节点数一致。
        if bold.shape[1] != sc.shape[0]:
            raise ValueError(
                f"{fc_path} has {bold.shape[1]} regions, but {sc_path} has {sc.shape[0]} nodes."
            )

        # 对 SC 矩阵做阈值、自环和 GCN 归一化。
        adj = preprocess_graph(sc, self.config.sc_threshold)

        # 节点特征使用单位矩阵，即每个脑区用 one-hot 向量表示。
        features = np.eye(sc.shape[0], dtype=np.float32)

        # 从 BOLD 时间序列计算动态 FC 标签，输出为每个窗口的上三角边向量。
        targets = compute_dynamic_fc_edges(
            bold_timeseries=bold,
            window_size=self.config.window_size,
            stride=self.config.stride,
            fisher_clip=self.config.fisher_clip,
        )

        # 转成 PyTorch 张量，返回给 DataLoader。
        sample = (
            torch.from_numpy(adj),
            torch.from_numpy(features),
            torch.from_numpy(targets),
        )

        # 根据配置决定是否缓存处理结果。
        if self.config.cache_samples_in_memory:
            self._cache[index] = sample
        return sample


class GraphConvolutionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        # 线性变换 W，用于把节点特征从 input_dim 映射到 output_dim。
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        # Xavier 初始化适合线性层，帮助训练初期保持梯度稳定。
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # GCN 基本形式：ReLU(A_norm X W)。
        # adj 形状通常为：[batch, nodes, nodes]
        # features 形状通常为：[batch, nodes, input_dim]
        return torch.relu(torch.matmul(adj, self.linear(features)))


class SCEncoder(nn.Module):
    def __init__(self, input_dim: int, config: Config) -> None:
        super().__init__()

        # 两层 GCN：先提取局部结构表示，再压缩到 SC 上下文维度。
        self.gcn1 = GraphConvolutionLayer(input_dim, config.gcn_hidden_dim)
        self.gcn2 = GraphConvolutionLayer(config.gcn_hidden_dim, config.sc_context_dim)

    def forward(self, adj: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        # 第一层图卷积。
        hidden = self.gcn1(adj, features)

        # 第二层图卷积。
        hidden = self.gcn2(adj, hidden)

        # 对节点维度做 max pooling，得到每个被试一个全局 SC 表征。
        return hidden.max(dim=1).values


class DynamicSCToFCModel(nn.Module):
    def __init__(self, input_dim: int, num_nodes: int, config: Config) -> None:
        super().__init__()
        self.num_nodes = num_nodes

        # 无向图上三角边数：N * (N - 1) / 2。
        self.num_edges = num_nodes * (num_nodes - 1) // 2
        self.teacher_forcing_ratio = config.teacher_forcing_ratio

        # 编码器：把静态 SC 图编码成一个上下文向量。
        self.encoder = SCEncoder(input_dim=input_dim, config=config)

        # 解码器：GRUCell 按时间窗口逐步生成动态 FC 边向量。
        # 输入由 [SC 上下文, 上一个窗口的 FC 边向量] 拼接而成。
        self.decoder_cell = nn.GRUCell(
            config.sc_context_dim + self.num_edges,
            config.gru_hidden_dim,
        )

        # 输出头：把 GRU 隐状态映射成当前窗口的 FC 边向量。
        self.output_head = nn.Linear(config.gru_hidden_dim, self.num_edges)

        # 起始 token：第一个窗口前没有上一帧 FC，因此用可学习向量作为初始输入。
        self.start_token = nn.Parameter(torch.zeros(1, self.num_edges))

        # 初始化输出层。
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
        num_windows: int,
        targets_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # batch_size 是当前批次中的被试数量。
        batch_size = adj.shape[0]

        # 编码 SC，得到每个被试的结构连接上下文。
        sc_context = self.encoder(adj, features)

        # 初始化 GRU 隐状态为 0。
        hidden = torch.zeros(
            batch_size,
            self.decoder_cell.hidden_size,
            dtype=adj.dtype,
            device=adj.device,
        )

        # 用一帧全零 FC 做 warmup，让 GRU 隐状态先吸收 SC 上下文。
        empty_fc = torch.zeros(batch_size, self.num_edges, dtype=adj.dtype, device=adj.device)
        warmup_input = torch.cat([sc_context, empty_fc], dim=-1)
        hidden = self.decoder_cell(warmup_input, hidden)

        # 第一个正式预测窗口的上一帧输入使用可学习 start_token。
        previous_fc = self.start_token.expand(batch_size, -1)
        predictions = []

        # 逐窗口自回归预测动态 FC。
        for window_index in range(num_windows):
            # 当前输入：静态 SC 上下文 + 上一个窗口 FC。
            decoder_input = torch.cat([sc_context, previous_fc], dim=-1)

            # 更新 GRU 隐状态。
            hidden = self.decoder_cell(decoder_input, hidden)

            # 输出当前窗口预测的 FC 边向量。
            predicted_fc = self.output_head(hidden)
            predictions.append(predicted_fc)

            # 默认下一步使用模型自己的预测作为上一帧。
            previous_fc = predicted_fc

            # 训练阶段可使用 teacher forcing：
            # 按概率改用真实上一窗口 FC，减少早期训练时误差累积。
            if self.training and targets_edges is not None and window_index < num_windows - 1:
                use_true_previous = torch.rand((), device=adj.device) < self.teacher_forcing_ratio
                if bool(use_true_previous):
                    previous_fc = targets_edges[:, window_index]

        # 返回形状：[batch, num_windows, num_edges]。
        return torch.stack(predictions, dim=1)


def edges_to_symmetric_matrices(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # 根据节点数生成上三角索引。
    triu = torch.triu_indices(num_nodes, num_nodes, offset=1, device=edges.device)

    # 创建全零矩阵，前面的维度沿用 edges 的 batch/时间窗口维度。
    matrices = torch.zeros(
        *edges.shape[:-1],
        num_nodes,
        num_nodes,
        dtype=edges.dtype,
        device=edges.device,
    )

    # 把边向量填回矩阵上三角和下三角，形成对称 FC 矩阵。
    matrices[..., triu[0], triu[1]] = edges
    matrices[..., triu[1], triu[0]] = edges
    return matrices


def temporal_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    # 普通均方误差：衡量预测 FC 边值和真实 FC 边值的平均平方差。
    mse = torch.mean((prediction - target) ** 2).item()

    # 平均绝对误差：相比 MSE 对异常大误差不那么敏感。
    mae = torch.mean(torch.abs(prediction - target)).item()

    # 对每个窗口的边向量做中心化，用于计算预测向量和真实向量的相关性。
    pred_centered = prediction - prediction.mean(dim=-1, keepdim=True)
    true_centered = target - target.mean(dim=-1, keepdim=True)

    # Pearson 相关系数的分子和分母。
    numerator = (pred_centered * true_centered).sum(dim=-1)
    denominator = torch.sqrt(
        (pred_centered.square().sum(dim=-1) * true_centered.square().sum(dim=-1)).clamp_min(1e-12)
    )

    # 所有 batch 和时间窗口上的平均相关性。
    window_corr = (numerator / denominator).mean().item()

    # Delta MSE：比较相邻动态 FC 窗口之间的变化量是否预测准确。
    # 它不只看 FC 数值本身，还看时间变化趋势。
    if prediction.shape[1] > 1:
        pred_delta = prediction[:, 1:] - prediction[:, :-1]
        true_delta = target[:, 1:] - target[:, :-1]
        delta_mse = torch.mean((pred_delta - true_delta) ** 2).item()
    else:
        delta_mse = float("nan")

    return {"mse": mse, "mae": mae, "corr": window_corr, "delta_mse": delta_mse}


def move_batch_to_device(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # DataLoader 返回的 batch 包含：
    # adj: 归一化 SC 邻接矩阵
    # features: 节点特征
    # targets: 动态 FC 边向量标签
    adj, features, targets = batch
    return adj.to(device), features.to(device), targets.to(device)


@torch.no_grad()
def evaluate(
    model: DynamicSCToFCModel,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    # 评估阶段关闭 dropout/batchnorm 等训练行为。
    model.eval()
    all_predictions = []
    all_targets = []

    # 不计算梯度，节省显存并加快验证/测试。
    for batch in data_loader:
        adj, features, targets = move_batch_to_device(batch, device)

        # 验证/测试时不传 targets_edges，因此模型完全自回归预测。
        predictions = model(adj, features, num_windows=targets.shape[1])
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    # 合并所有 batch 后统一计算指标。
    return temporal_metrics(torch.cat(all_predictions), torch.cat(all_targets))


def split_dataset(dataset: Dataset, config: Config) -> tuple[Dataset, Dataset, Dataset]:
    total = len(dataset)

    # 先按比例划出测试集，至少保留 1 个样本用于测试。
    test_size = max(1, int(round(total * config.test_ratio)))
    train_val_size = total - test_size

    # 再从剩余数据中划出验证集，至少保留 1 个样本用于验证。
    val_size = max(1, int(round(train_val_size * config.val_ratio_in_train)))
    train_size = train_val_size - val_size

    # 如果训练集为空，说明被试数量太少，无法完成三分法划分。
    if train_size < 1:
        raise ValueError("Not enough subjects for train/validation/test split.")

    # 使用固定随机种子，保证每次划分一致。
    generator = torch.Generator().manual_seed(config.random_seed)
    train_val_dataset, test_dataset = random_split(dataset, [train_val_size, test_size], generator)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size], generator)
    return train_dataset, val_dataset, test_dataset


def make_loader(dataset: Dataset, config: Config, shuffle: bool) -> DataLoader:
    # 封装 DataLoader 创建逻辑，训练集一般 shuffle，验证/测试集不 shuffle。
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_model(
    dataset: SCFCDataset,
    config: Config,
    device: torch.device,
) -> DynamicSCToFCModel:
    # 将被试划分为训练、验证、测试三部分。
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, config)
    train_loader = make_loader(train_dataset, config, shuffle=True)
    val_loader = make_loader(val_dataset, config, shuffle=False)
    test_loader = make_loader(test_dataset, config, shuffle=False)

    # 根据数据集维度创建模型。
    model = DynamicSCToFCModel(
        input_dim=dataset.input_dim,
        num_nodes=dataset.num_nodes,
        config=config,
    ).to(device)

    # Adam 优化器和 MSE 损失函数。
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # 记录验证集 MSE 最好的模型。
    best_val_mse = float("inf")
    best_state = None
    patience_count = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch in train_loader:
            adj, features, targets = move_batch_to_device(batch, device)

            # 每个 batch 前清空梯度。
            optimizer.zero_grad()

            # 训练时传入 targets_edges，以便模型内部执行 teacher forcing。
            predictions = model(
                adj,
                features,
                num_windows=targets.shape[1],
                targets_edges=targets,
            )

            # 预测动态 FC 边向量与真实动态 FC 边向量之间的 MSE。
            loss = criterion(predictions, targets)

            # 反向传播并更新参数。
            loss.backward()
            optimizer.step()

            # 累计训练损失，按样本数加权，避免最后一个 batch 较小带来偏差。
            batch_size = adj.shape[0]
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        train_mse = train_loss_sum / train_samples

        # 每个 epoch 后在验证集上评估。
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1:03d} | Train MSE {train_mse:.6f} | "
            f"Val MSE {val_metrics['mse']:.6f} | Val Corr {val_metrics['corr']:.6f} | "
            f"Val Delta MSE {val_metrics['delta_mse']:.6f}"
        )

        # 如果验证集 MSE 改善，则保存当前最佳状态。
        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            # 如果连续若干 epoch 没有改善，则触发早停。
            patience_count += 1
            if patience_count >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid model state.")

    # 保存最佳模型权重。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_PATH)

    # 将模型恢复到验证集表现最好的参数。
    model.load_state_dict(best_state)

    # 在测试集上做最终评估。
    test_metrics = evaluate(model, test_loader, device)
    print(
        "Test | "
        f"MSE {test_metrics['mse']:.6f} | MAE {test_metrics['mae']:.6f} | "
        f"Corr {test_metrics['corr']:.6f} | Delta MSE {test_metrics['delta_mse']:.6f}"
    )
    print(f"Saved best model to {MODEL_PATH}")
    return model


@torch.no_grad()
def predict_dynamic_fc_matrices(
    model: DynamicSCToFCModel,
    adj: torch.Tensor,
    features: torch.Tensor,
    num_windows: int,
    device: torch.device,
) -> np.ndarray:
    # 切换到评估模式，用训练好的模型生成动态 FC。
    model.eval()

    # 这里传入单个被试，因此先 unsqueeze(0) 增加 batch 维度。
    predicted_edges = model(
        adj.unsqueeze(0).to(device),
        features.unsqueeze(0).to(device),
        num_windows=num_windows,
    )

    # 把上三角边向量还原为完整对称 FC 矩阵序列。
    matrices = edges_to_symmetric_matrices(predicted_edges, model.num_nodes)
    return matrices.squeeze(0).cpu().numpy().astype(np.float32)


def main() -> None:
    config = Config()

    # 固定 NumPy 和 PyTorch 的随机种子，增强结果可复现性。
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # 构建数据集，会自动扫描并匹配 SC/FC 文件。
    dataset = SCFCDataset(SC_DIR, FC_DIR, config)

    # 如果有 CUDA GPU 则使用 GPU，否则使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印数据和运行环境信息。
    print(f"Device: {device}")
    print(f"Subjects: {len(dataset)}, Regions: {dataset.num_nodes}")
    print(f"Dynamic FC windows: {dataset.num_windows}, Predicted edges: {dataset.num_edges}")
    print(f"SC dir: {SC_DIR}")
    print(f"FC dir: {FC_DIR}")

    # 启动训练、验证、测试，并保存最佳模型。
    train_model(dataset=dataset, config=config, device=device)


if __name__ == "__main__":
    main()
