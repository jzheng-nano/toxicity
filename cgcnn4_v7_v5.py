import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm, GINEConv
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import warnings
import argparse
import sys
import traceback
import logging
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch.utils.data import TensorDataset, DataLoader
import json
from pathlib import Path
import torch.multiprocessing as mp
from functools import partial


# 设置日志 - 修复编码问题
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 文件处理器使用UTF-8编码
    file_handler = logging.FileHandler("cgcnn_enhanced.log", encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


logger = setup_logging()

# 配置参数
MATERIALS_API_KEY = "API_Key"
BATCH_SIZE = 32
EPOCHS = 3000
LEARNING_RATE = 0.001
CRYSTAL_DATA_CACHE = "crystal_data_enhanced.pkl"
EMBEDDING_CACHE = "crystal_embeddings_enhanced.pkl"
CONFIG_FILE = "model_config_enhanced.pkl"
MAX_CONCURRENT_MODELS = 3  # 最大并行训练的模型数量


# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='晶体毒性预测模型 - 增强版')
    parser.add_argument('--download', action='store_true', help='仅下载晶体数据（不训练）')
    parser.add_argument('--offline', action='store_true', help='离线模式（使用缓存数据）')
    parser.add_argument('--debug', action='store_true', help='启用调试模式（详细错误信息）')
    parser.add_argument('--skip-embeddings', action='store_true', help='跳过嵌入预计算')
    parser.add_argument('--skip-plots', action='store_true', help='跳过绘图生成')
    parser.add_argument('--graph-method', type=str, default='crystalnn',
                        choices=['voronoi', 'crystalnn', 'hybrid'],
                        help='图构建方法: voronoi, crystalnn 或 hybrid')
    parser.add_argument('--ensemble', action='store_true', help='使用集成学习策略')
    parser.add_argument('--parallel', type=int, default=MAX_CONCURRENT_MODELS,
                        help=f'并行训练的模型数量 (默认: {MAX_CONCURRENT_MODELS})')
    parser.add_argument('--gpu', type=int, nargs='+', default=None,
                        help='指定使用的GPU设备ID (例如: 0 或 0,1)')
    return parser.parse_args()


# 错误处理装饰器
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} 执行成功 (耗时: {elapsed:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} 执行失败: {str(e)}")
            if args and hasattr(args[0], 'debug') and args[0].debug:
                logger.exception("详细错误信息:")
            else:
                logger.info("使用 --debug 参数查看详细错误信息")
            sys.exit(1)

    return wrapper


# 特征融合模块
class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, feat1, feat2):
        combined = torch.cat([feat1, feat2], dim=-1)
        gate = self.gate(combined)
        projected = self.proj(combined)
        return gate * feat1 + (1 - gate) * feat2 + projected


# 1. 数据准备与预处理 (带完整缓存机制)
@error_handler
def load_and_preprocess_data(args):
    logger.info("开始数据加载与预处理...")

    # 检查必要文件
    required_files = ['materials_name.csv', 'data_cleaned1B.csv']
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"缺少必要文件: {file}")

    logger.info("加载CSV数据...")
    materials_df = pd.read_csv('materials_name.csv')
    exp_df = pd.read_csv('data_cleaned1B.csv')

    # 合并数据
    merged_df = pd.merge(exp_df, materials_df, on='Materials', how='left')

    # 检查关键列是否存在
    required_columns = ['mp_id', 'Cell_Viability', 'Metal_Valence_State', 'Shape', 'Coatings', 'Cell_line_name',
                        'Assay', 'Size', 'Time', 'Concentration']
    for col in required_columns:
        if col not in merged_df.columns:
            raise ValueError(f"数据中缺少必要列: {col}")

    # 特征工程 - 浓度对数变换 (使用log1p避免过小的常数)
    merged_df['Log_Concentration'] = np.log10(merged_df['Concentration'] + 1e-8)
    merged_df['Log_Time'] = np.log10(merged_df['Time'] + 1e-8)
    merged_df['Log_Size'] = np.log10(merged_df['Size'] + 1e-8)

    # 增强特征交互
    merged_df['Size_Time'] = merged_df['Log_Size'] * merged_df['Log_Time']
    merged_df['Concentration_Time'] = merged_df['Log_Concentration'] * merged_df['Log_Time']
    merged_df['Size_Concentration'] = merged_df['Log_Size'] * merged_df['Log_Concentration']
    merged_df['Size_Concentration_Time'] = merged_df['Log_Size'] * merged_df['Log_Concentration'] * merged_df[
        'Log_Time']

    # 分类特征和数值特征
    categorical_features = ['Shape', 'Coatings', 'Cell_line_name', 'Assay']
    numerical_features = [
        'Log_Size', 'Log_Time', 'Log_Concentration',
        'Size_Time', 'Concentration_Time',
        'Size_Concentration', 'Size_Concentration_Time'
    ]

    # 创建特征交叉
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    num_poly = poly.fit_transform(merged_df[numerical_features])
    num_poly_df = pd.DataFrame(num_poly, columns=poly.get_feature_names_out(numerical_features))

    # 预处理分类特征
    cat_processor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat = cat_processor.fit_transform(merged_df[categorical_features])

    # 合并所有特征
    X_experimental = np.hstack([num_poly_df.values, X_cat])
    y = merged_df['Cell_Viability'].values

    # 获取唯一的晶体结构ID
    unique_mp_ids = merged_df['mp_id'].unique()
    logger.info(f"发现 {len(unique_mp_ids)} 个独特晶体结构")

    # 检查晶体数据缓存
    if os.path.exists(CRYSTAL_DATA_CACHE) and not args.download:
        logger.info(f"加载晶体数据缓存: {CRYSTAL_DATA_CACHE}")
        with open(CRYSTAL_DATA_CACHE, 'rb') as f:
            crystal_data_cache = pickle.load(f)
    else:
        if args.offline:
            raise RuntimeError("离线模式但找不到晶体数据缓存。请先在有网络的环境运行 --download")

        logger.info("从Materials Project下载晶体数据...")
        crystal_data_cache = {'structures': {}, 'graphs': {}}

        # 记录下载失败的材料
        failed_ids = []

        with MPRester(MATERIALS_API_KEY) as mpr:
            for mp_id in tqdm(unique_mp_ids, desc="下载晶体结构"):
                try:
                    # 忽略pymatgen的警告
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        structure = mpr.get_structure_by_material_id(mp_id)
                        structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()

                    graph = structure_to_graph(structure, args.graph_method)
                    # 添加唯一标识符
                    graph.mp_id = mp_id
                    crystal_data_cache['structures'][mp_id] = structure
                    crystal_data_cache['graphs'][mp_id] = graph
                except Exception as e:
                    logger.warning(f"下载 {mp_id} 失败: {str(e)}")
                    failed_ids.append(mp_id)
                    # 使用占位图避免中断
                    dummy = create_dummy_graph(mp_id)
                    crystal_data_cache['graphs'][mp_id] = dummy

        # 保存缓存
        with open(CRYSTAL_DATA_CACHE, 'wb') as f:
            pickle.dump(crystal_data_cache, f)
            logger.info(f"晶体数据已保存到缓存: {CRYSTAL_DATA_CACHE}")

        if failed_ids:
            logger.warning(f"以下材料ID下载失败: {', '.join(failed_ids)}")
            logger.warning(f"已为这些材料创建占位图")

        if args.download:
            logger.info("✅ 晶体数据下载完成。请将此文件复制到GPU机器:")
            logger.info(f"  - {os.path.abspath(CRYSTAL_DATA_CACHE)}")
            logger.info("然后使用 --offline 参数运行训练")
            return None, None, None, None, None, None, None

    # 为每个样本分配对应的图并添加唯一标识符
    crystal_graphs = []
    for idx, mp_id in enumerate(merged_df['mp_id']):
        if mp_id in crystal_data_cache['graphs']:
            graph = crystal_data_cache['graphs'][mp_id]
            # 确保每个图都有唯一标识
            if not hasattr(graph, 'mp_id'):
                graph.mp_id = mp_id
            crystal_graphs.append(graph)
        else:
            logger.warning(f"第 {idx} 行: {mp_id} 不在缓存中，使用占位图")
            dummy = create_dummy_graph(mp_id)
            crystal_graphs.append(dummy)

    # 保存预处理器配置
    with open(CONFIG_FILE, 'wb') as f:
        pickle.dump({
            'cat_processor': cat_processor,
            'poly': poly,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features
        }, f)
        logger.info(f"保存预处理器配置到 {CONFIG_FILE}")

    logger.info("数据预处理完成")
    return X_experimental, crystal_graphs, y, cat_processor, crystal_data_cache[
        'graphs'], merged_df  # 返回merged_df用于保存预测结果


def create_dummy_graph(mp_id="dummy"):
    """创建占位图用于缺失结构"""
    # 使用氢原子特征作为占位符 - 16维特征
    dummy_features = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return Data(
        x=dummy_features,
        edge_index=torch.tensor([[0, 0]], dtype=torch.long).t().contiguous(),  # 自环避免空图
        edge_attr=torch.tensor([[0.0]], dtype=torch.float),
        mp_id=mp_id
    )


def get_atomic_features(species):
    """获取增强的原子特征"""
    element = Element(species.symbol)
    # 基本特征
    features = [
        element.Z,  # 原子序数
        element.X,  # 电负性 (Pauling)
        element.atomic_mass,  # 原子质量
        element.group,  # 周期表族
        element.row,  # 周期
        element.atomic_radius,  # 原子半径
        element.average_ionic_radius,
        element.average_anionic_radius,
        element.average_cationic_radius,
        element.ionization_energy,  # 第一电离能
        element.electron_affinity,  # 电子亲和能
    ]

    # 周期性特征 (正弦余弦编码)
    period_sin = math.sin(2 * math.pi * element.row / 9)  # 最大周期为9
    period_cos = math.cos(2 * math.pi * element.row / 9)
    group_sin = math.sin(2 * math.pi * element.group / 18)  # 最大族为18
    group_cos = math.cos(2 * math.pi * element.group / 18)

    # 添加价电子数特征
    nvalence = element.nvalence if hasattr(element, 'nvalence') else element.group
    features.append(nvalence)

    features.extend([period_sin, period_cos, group_sin, group_cos])

    # 确保是16维特征
    assert len(features) == 16, f"原子特征维度错误: 期望16, 实际{len(features)}"
    return features


def structure_to_graph(structure, method='crystalnn', max_cutoff=15.0):
    """高效将晶体结构转换为图表示（增强特征）"""
    # 节点特征（现在包含更多原子信息）
    atomic_features = []
    for site in structure:
        features = get_atomic_features(site.specie)
        atomic_features.append(features)

    # 转换为张量
    x = torch.tensor(atomic_features, dtype=torch.float)

    # 归一化特征（每列独立归一化）
    for i in range(x.size(1)):
        col = x[:, i]
        if col.max() - col.min() > 1e-6:  # 避免除零
            x[:, i] = (col - col.min()) / (col.max() - col.min())
        else:
            x[:, i] = 0.0  # 常数列设为0

    # 构建邻接表
    edge_index = []
    edge_attr = []
    max_attempts = 3
    current_cutoff = 8.0  # 初始截断距离
    for attempt in range(max_attempts):
        try:
            if method == 'voronoi':
                # 使用 Voronoi 分析构建图
                vnn = VoronoiNN(tol=0.5, cutoff=current_cutoff)  # 增加容差
                for i in range(len(structure)):
                    neighbors = vnn.get_nn_info(structure, i)
                    for neighbor in neighbors:
                        j = neighbor['site_index']
                        if i != j:  # 排除自环
                            distance = neighbor['weight']
                            # 避免重复边
                            if [i, j] not in edge_index and [j, i] not in edge_index:
                                edge_index.append([i, j])
                                edge_attr.append([distance])

            elif method == 'crystalnn':
                # 使用 CrystalNN 构建图（修复未使用的代码）
                cnn = CrystalNN()
                for i in range(len(structure)):
                    neighbors = cnn.get_nn_info(structure, i)
                    for neighbor in neighbors:
                        j = neighbor['site_index']
                        if i != j:  # 排除自环
                            distance = neighbor['weight']
                            # 避免重复边
                            if [i, j] not in edge_index and [j, i] not in edge_index:
                                edge_index.append([i, j])
                                edge_attr.append([distance])

            elif method == 'hybrid':
                try:
                    # 先尝试 Voronoi
                    vnn = VoronoiNN(tol=0.5, cutoff=current_cutoff)
                    for i in range(len(structure)):
                        neighbors = vnn.get_nn_info(structure, i)
                        for neighbor in neighbors:
                            j = neighbor['site_index']
                            if i != j:
                                distance = neighbor['weight']
                                if [i, j] not in edge_index and [j, i] not in edge_index:
                                    edge_index.append([i, j])
                                    edge_attr.append([distance])
                except:
                    # 如果 Voronoi 失败，使用距离截断方法
                    from pymatgen.analysis.local_env import MinimumDistanceNN
                    mnn = MinimumDistanceNN(cutoff=current_cutoff)
                    for i in range(len(structure)):
                        neighbors = mnn.get_nn_info(structure, i)
                        for neighbor in neighbors:
                            j = neighbor['site_index']
                            if i != j:
                                distance = neighbor['weight']
                                if [i, j] not in edge_index and [j, i] not in edge_index:
                                    edge_index.append([i, j])
                                    edge_attr.append([distance])

            # 检查是否找到边
            if edge_index:
                break
            else:
                # 增加截断距离重试
                current_cutoff += 3.0
                logger.warning(f"尝试 {attempt + 1}: 未找到边，增加截断距离至 {current_cutoff:.1f}Å")
                if current_cutoff > max_cutoff:
                    break

        except Exception as e:
            logger.warning(f"图构建尝试 {attempt + 1} 失败: {str(e)}")
            current_cutoff += 3.0
            if current_cutoff > max_cutoff:
                break

    # 转换为张量
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        # 最终回退：创建全连接图
        logger.warning("所有尝试均未找到边，创建全连接图")
        num_nodes = len(structure)
        edge_index = []
        edge_attr = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # 计算实际距离
                distance = structure[i].distance(structure[j])
                edge_index.append([i, j])
                edge_attr.append([distance])

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            # 单原子情况
            logger.warning("单原子结构，创建自环")
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_attr = torch.tensor([[0.1]], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mp_id=getattr(structure, 'material_id', 'unknown')
    )


# 2. 模型架构 (增强版)
class EnhancedCrystalToxicityPredictor(nn.Module):
    def __init__(self, exp_feature_dim, gcn_hidden_dim=256, fc_hidden_dim=128):
        super().__init__()
        self.gcn_hidden_dim = gcn_hidden_dim
        self.dropout = nn.Dropout(0.6)  # 增加dropout比例防止过拟合

        # 晶体编码器（使用边特征）
        self.edge_encoder1 = nn.Sequential(
            nn.Linear(1, gcn_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(gcn_hidden_dim),
            self.dropout,
        )

        # 修正输入维度为16
        self.gine1 = GINEConv(
            nn.Sequential(
                nn.Linear(16, gcn_hidden_dim),  # 输入维度16
                nn.LeakyReLU(0.1),
                nn.LayerNorm(gcn_hidden_dim),
                self.dropout,
                nn.Linear(gcn_hidden_dim, gcn_hidden_dim),
            ),
            edge_dim=gcn_hidden_dim
        )
        self.bn1 = BatchNorm(gcn_hidden_dim)

        # 图注意力层
        self.gat_conv = GATConv(gcn_hidden_dim, gcn_hidden_dim // 4, heads=4, dropout=0.3)  # 增加dropout
        self.gat_norm = BatchNorm(gcn_hidden_dim)

        self.edge_encoder2 = nn.Sequential(
            nn.Linear(gcn_hidden_dim, gcn_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(gcn_hidden_dim),
            self.dropout,
        )
        self.gine2 = GINEConv(
            nn.Sequential(
                nn.Linear(gcn_hidden_dim, gcn_hidden_dim),
                nn.LeakyReLU(0.1),
                nn.LayerNorm(gcn_hidden_dim),
                self.dropout,
                nn.Linear(gcn_hidden_dim, gcn_hidden_dim),
            ),
            edge_dim=gcn_hidden_dim
        )
        self.bn2 = BatchNorm(gcn_hidden_dim)

        # 残差连接 - 输入维度16
        self.residual = nn.Sequential(
            nn.Linear(16, gcn_hidden_dim),  # 输入维度16
            nn.LayerNorm(gcn_hidden_dim)
        )

        # 实验特征处理器
        self.exp_encoder = nn.Sequential(
            nn.Linear(exp_feature_dim, 512),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(512),
            self.dropout,
            nn.Linear(512, gcn_hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(gcn_hidden_dim),
            self.dropout
        )

        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            gcn_hidden_dim, num_heads=8, dropout=0.3, batch_first=True  # 增加dropout
        )

        # 特征融合模块
        self.fusion_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * gcn_hidden_dim, 2 * gcn_hidden_dim),
                nn.LeakyReLU(0.1),
                nn.LayerNorm(2 * gcn_hidden_dim),
                self.dropout
            ) for _ in range(1)
        ])

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(2 * gcn_hidden_dim, 2 * gcn_hidden_dim),
            nn.Sigmoid()
        )

        # 联合预测器
        fusion_dim = 2 * gcn_hidden_dim
        self.final_predictor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(256),
            self.dropout,
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward_crystal(self, crystal_data):
        x, edge_index, edge_attr = crystal_data.x, crystal_data.edge_index, crystal_data.edge_attr

        # 处理空图/单原子情况
        if edge_index.size(1) == 0:
            num_nodes = x.size(0)
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
            edge_attr = torch.ones((num_nodes, 1)) * 0.1

        edge_attr = edge_attr.to(x.device)
        edge_embed1 = self.edge_encoder1(edge_attr)
        identity = self.residual(x)  # 输入维度16

        # GINE层
        x = F.leaky_relu(self.bn1(self.gine1(x, edge_index, edge_embed1)), 0.1)

        # 图注意力层
        x = F.leaky_relu(self.gat_norm(self.gat_conv(x, edge_index)), 0.1)

        edge_embed2 = self.edge_encoder2(edge_embed1)
        x = F.leaky_relu(self.bn2(self.gine2(x, edge_index, edge_embed2)) + identity, 0.1)

        # 全局平均池化
        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device))

    def forward(self, crystal_embed, exp_features):
        exp_embed = self.exp_encoder(exp_features)

        # 交叉注意力
        crystal_attn = crystal_embed.unsqueeze(1)
        exp_attn = exp_embed.unsqueeze(1)
        attn_output, _ = self.cross_attention(crystal_attn, exp_attn, exp_attn)
        attn_output = attn_output.squeeze(1)

        # 特征拼接
        combined = torch.cat([crystal_embed, exp_embed], dim=1)

        # 特征融合
        fused = combined
        for block in self.fusion_blocks:
            fused = block(fused) + fused

        # 门控机制
        gate_value = self.gate(combined)
        gated_output = gate_value * fused + (1 - gate_value) * combined

        # 最终预测
        return self.final_predictor(gated_output).squeeze()


@error_handler
def precompute_crystal_embeddings(model, crystal_graphs_dict, device):
    embeddings = {}
    model.to(device)
    model.eval()

    stats = defaultdict(int)

    with torch.no_grad():
        for mp_id, graph in tqdm(crystal_graphs_dict.items(), desc="预计算嵌入"):
            try:
                graph.x = graph.x.to(device)
                if graph.edge_index is not None:
                    graph.edge_index = graph.edge_index.to(device)
                if graph.edge_attr is not None:
                    graph.edge_attr = graph.edge_attr.to(device)

                # 添加维度检查
                if graph.x.dim() == 1:
                    graph.x = graph.x.unsqueeze(0)

                embed = model.forward_crystal(graph).cpu()

                # 确保嵌入维度正确
                if embed.dim() == 1:
                    embed = embed.unsqueeze(0)

                embeddings[mp_id] = embed
                stats['success'] += 1
            except Exception as e:
                logger.error(f"处理 {mp_id} 失败: {str(e)}")
                embeddings[mp_id] = torch.zeros(1, model.gcn_hidden_dim)
                stats['failed'] += 1

    logger.info(f"嵌入预计算完成: 成功 {stats['success']}, 失败 {stats['failed']}")
    return embeddings


@error_handler
def train_and_evaluate(X_experimental, crystal_graphs, y, crystal_graphs_dict, device, merged_df, args, model_idx=0):
    logger.info("开始训练与评估流程...")

    # 划分数据集 - 80%训练+验证, 20%测试
    indices = np.arange(len(y))
    X_temp, X_test, graph_temp, graph_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X_experimental, crystal_graphs, y, indices, test_size=0.2, random_state=42, stratify=(y > 0.5).astype(int)
    )

    # 在训练集上拟合特征选择器 (修复数据泄露问题)
    selector = SelectKBest(f_regression, k=min(1200, X_temp.shape[1]))
    X_temp_selected = selector.fit_transform(X_temp, y_temp)
    X_test_selected = selector.transform(X_test)

    logger.info(f"特征选择后维度: {X_temp_selected.shape[1]}")

    # 进一步划分训练集和验证集
    X_train, X_val, graph_train, graph_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_temp_selected, graph_temp, y_temp, idx_temp, test_size=0.125, random_state=42,
        stratify=(y_temp > 0.5).astype(int)
    )

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float)
    X_val = torch.tensor(X_val, dtype=torch.float)
    X_test = torch.tensor(X_test_selected, dtype=torch.float)  # 使用选择后的特征
    y_train = torch.tensor(y_train, dtype=torch.float)
    y_val = torch.tensor(y_val, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)

    # 保存原始索引用于结果匹配
    idx_train = torch.tensor(idx_train, dtype=torch.long)
    idx_val = torch.tensor(idx_val, dtype=torch.long)
    idx_test = torch.tensor(idx_test, dtype=torch.long)

    # 检查嵌入缓存
    if os.path.exists(EMBEDDING_CACHE) and not args.skip_embeddings:
        logger.info(f"加载预计算的嵌入缓存: {EMBEDDING_CACHE}")
        with open(EMBEDDING_CACHE, 'rb') as f:
            crystal_embeddings = pickle.load(f)
    else:
        if args.skip_embeddings:
            logger.info("跳过嵌入预计算")
        else:
            logger.info("未找到嵌入缓存，开始预计算...")

        # 初始化模型
        exp_feature_dim = X_train.shape[1]
        model = EnhancedCrystalToxicityPredictor(exp_feature_dim)

        crystal_embeddings = precompute_crystal_embeddings(model, crystal_graphs_dict, device)

        # 保存缓存
        if not args.skip_embeddings:
            with open(EMBEDDING_CACHE, 'wb') as f:
                pickle.dump(crystal_embeddings, f)
                logger.info(f"晶体嵌入已保存到缓存: {EMBEDDING_CACHE}")

    # 为数据集创建嵌入
    def create_embedding_dataset(graphs, gcn_hidden_dim=256):
        """创建嵌入数据集 - 确保维度一致"""
        embeds = []
        for graph in graphs:
            mp_id = getattr(graph, 'mp_id', None)
            embed = None

            # 尝试从预计算嵌入中获取
            if mp_id and mp_id in crystal_embeddings:
                embed = crystal_embeddings[mp_id]
            else:
                # 尝试查找匹配的图
                for known_id, known_graph in crystal_graphs_dict.items():
                    if hasattr(graph, 'mp_id') and graph.mp_id == known_id:
                        embed = crystal_embeddings[known_id]
                        break

            # 如果仍未找到，创建默认嵌入
            if embed is None:
                embed = torch.zeros(1, gcn_hidden_dim)

            # 确保嵌入是2D张量 [1, features]
            if embed.dim() == 1:
                embed = embed.unsqueeze(0)

            # 检查特征维度
            if embed.size(1) != gcn_hidden_dim:
                logger.warning(f"嵌入维度不匹配: 期望{gcn_hidden_dim}, 实际{embed.size(1)}")
                embed = torch.zeros(1, gcn_hidden_dim)

            embeds.append(embed)

        return torch.cat(embeds, dim=0)

    logger.info("为训练集创建嵌入...")
    crystal_emb_train = create_embedding_dataset(graph_train)
    logger.info("为验证集创建嵌入...")
    crystal_emb_val = create_embedding_dataset(graph_val)
    logger.info("为测试集创建嵌入...")
    crystal_emb_test = create_embedding_dataset(graph_test)

    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_dataset = TensorDataset(crystal_emb_train, X_train, y_train, idx_train)
    val_dataset = TensorDataset(crystal_emb_val, X_val, y_val, idx_val)
    test_dataset = TensorDataset(crystal_emb_test, X_test, y_test, idx_test)

    # 确保批次大小至少为2，避免BatchNorm问题
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=True)

    # 初始化模型
    logger.info(f"初始化模型 (设备: {device})...")
    exp_feature_dim = X_train.shape[1]
    model = EnhancedCrystalToxicityPredictor(exp_feature_dim).to(device)

    # 分离模型参数：主干网络和融合注意力层
    logger.info("设置优化器和学习率调度器...")
    base_params = []
    fusion_attn_params = []

    for name, param in model.named_parameters():
        if 'cross_attention' in name or 'final_predictor' in name:
            fusion_attn_params.append(param)
        else:
            base_params.append(param)

    # 增加权重衰减（L2正则化）
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': LEARNING_RATE, 'weight_decay': 5e-4},  # 增加权重衰减防止过拟合
        {'params': fusion_attn_params, 'lr': LEARNING_RATE * 0.5, 'weight_decay': 5e-4}
    ])

    # 动态学习率调度策略
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=100,
        min_lr=1e-6
    )

    # 使用加权Huber损失 - 对低活性样本给予更高权重
    def weighted_huber_loss(pred, target):
        # 为低活性样本分配更高权重
        weights = torch.where(target < 0.4, 2.5, 1.0)
        huber_loss = F.huber_loss(pred, target, reduction='none', delta=0.3)
        return (huber_loss * weights).mean()

    criterion = weighted_huber_loss

    # 训练循环
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    best_epoch = 0
    patience_counter = 0
    patience = 300  # 减少早停耐心值

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'epoch_time': [],
        'lr': []
    }

    logger.info("开始训练...")
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        batch_count = 0

        for batch_idx, (crystal_emb, exp_feats, targets, indices) in enumerate(train_loader):
            try:
                # 确保所有张量都在正确设备上
                crystal_emb = crystal_emb.to(device)
                exp_feats = exp_feats.to(device)
                targets = targets.to(device)

                # 双重检查批次大小
                if crystal_emb.size(0) < 2:
                    logger.warning(f"跳过大小为{crystal_emb.size(0)}的批次 (批次索引: {batch_idx})")
                    continue

                optimizer.zero_grad()
                preds = model(crystal_emb, exp_feats)
                loss = criterion(preds, targets)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()
                train_loss += loss.item() * len(targets)
                batch_count += 1

            except Exception as e:
                logger.error(f"训练批次失败: {str(e)}")
                if args.debug:
                    logger.exception("批次失败详情")

        # 验证
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for crystal_emb, exp_feats, targets, indices in val_loader:
                try:
                    crystal_emb = crystal_emb.to(device)
                    exp_feats = exp_feats.to(device)
                    targets = targets.to(device)

                    preds = model(crystal_emb, exp_feats)
                    loss = criterion(preds, targets)
                    val_loss += loss.item() * len(targets)

                    val_preds.append(preds.cpu())
                    val_targets.append(targets.cpu())
                except Exception as e:
                    logger.error(f"验证批次失败: {str(e)}")

        # 计算验证集R2
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_r2 = r2_score(val_targets.numpy(), val_preds.numpy())

        # 计算平均损失
        train_loss_avg = train_loss / len(train_dataset) if len(train_dataset) > 0 else float('nan')
        val_loss_avg = val_loss / len(val_dataset) if len(val_dataset) > 0 else float('nan')

        # 更新学习率
        scheduler.step(val_loss_avg)

        # 记录历史
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['val_r2'].append(val_r2)
        history['epoch_time'].append(time.time() - epoch_start)
        history['lr'].append(optimizer.param_groups[0]['lr'])  # 记录基础学习率

        # 早停和模型保存（仅基于验证集）
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0

            # 保存最佳模型
            model_name = f'best_model_{model_idx}.pth' if model_idx > 0 else 'best_model.pth'
            torch.save(model.state_dict(), model_name)
            logger.info(f"保存最佳模型 (epoch {epoch + 1}, val_loss={val_loss_avg:.4f}, val_r2={val_r2:.4f})")
        else:
            patience_counter += 1

        epoch_time = history['epoch_time'][-1]
        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | "
            f"Val R²: {val_r2:.4f} | LR: {history['lr'][-1]:.6f} | Time: {epoch_time:.1f}s")

        # 早停检查
        if patience_counter >= patience:
            logger.info(f"验证损失连续{patience}个epoch未改善，提前停止训练")
            break

    # 训练结束后评估测试集
    logger.info("训练结束，开始测试集评估...")
    model_name = f'best_model_{model_idx}.pth' if model_idx > 0 else 'best_model.pth'
    model.load_state_dict(torch.load(model_name))
    model.eval()

    # 收集所有预测结果
    all_predictions = {
        'train': {'indices': [], 'true': [], 'pred': []},
        'val': {'indices': [], 'true': [], 'pred': []},
        'test': {'indices': [], 'true': [], 'pred': []}
    }

    # 训练集预测
    logger.info("收集训练集预测结果...")
    with torch.no_grad():
        for crystal_emb, exp_feats, targets, indices in train_loader:
            crystal_emb = crystal_emb.to(device)
            exp_feats = exp_feats.to(device)
            targets = targets.to(device)

            preds = model(crystal_emb, exp_feats)
            all_predictions['train']['indices'].extend(indices.cpu().numpy())
            all_predictions['train']['true'].extend(targets.cpu().numpy())
            all_predictions['train']['pred'].extend(preds.detach().cpu().numpy())

    # 验证集预测
    logger.info("收集验证集预测结果...")
    with torch.no_grad():
        for crystal_emb, exp_feats, targets, indices in val_loader:
            crystal_emb = crystal_emb.to(device)
            exp_feats = exp_feats.to(device)
            targets = targets.to(device)

            preds = model(crystal_emb, exp_feats)
            all_predictions['val']['indices'].extend(indices.cpu().numpy())
            all_predictions['val']['true'].extend(targets.cpu().numpy())
            all_predictions['val']['pred'].extend(preds.detach().cpu().numpy())

    # 测试集预测
    logger.info("收集测试集预测结果...")
    with torch.no_grad():
        for crystal_emb, exp_feats, targets, indices in test_loader:
            crystal_emb = crystal_emb.to(device)
            exp_feats = exp_feats.to(device)
            targets = targets.to(device)

            preds = model(crystal_emb, exp_feats)
            all_predictions['test']['indices'].extend(indices.cpu().numpy())
            all_predictions['test']['true'].extend(targets.cpu().numpy())
            all_predictions['test']['pred'].extend(preds.detach().cpu().numpy())

    # 计算测试指标
    test_true = np.array(all_predictions['test']['true'])
    test_pred = np.array(all_predictions['test']['pred'])

    if len(test_true) > 0:
        test_mae = mean_absolute_error(test_true, test_pred)
        test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
        test_r2 = r2_score(test_true, test_pred)

        logger.info(f'\n最终测试结果:')
        logger.info(f'MAE: {test_mae:.4f}')
        logger.info(f'RMSE: {test_rmse:.4f}')
        logger.info(f'R^2: {test_r2:.4f}')
    else:
        logger.error("未能计算测试指标，没有有效的预测结果")

    # 保存预测结果
    save_predictions(all_predictions, merged_df, best_epoch + 1, model_idx)

    # 评估模型性能并保存指标
    evaluate_model_performance(all_predictions, best_epoch + 1, model_idx)

    # 绘图
    if not args.skip_plots:
        plot_results(all_predictions, history, best_epoch + 1, model_idx)

    return model


class CrystalToxicityEnsemble:
    def __init__(self, num_models=7, exp_feature_dim=None, device=None):  # 增加基模型数量
        self.num_models = num_models
        self.models = []
        self.exp_feature_dim = exp_feature_dim
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.histories = []  # 存储每个基模型的训练历史

        # 初始化多个基模型
        for i in range(num_models):
            model = EnhancedCrystalToxicityPredictor(exp_feature_dim).to(self.device)
            self.models.append(model)

    def _train_single_model(self, model_idx, X_train, X_val, graph_train, graph_val, y_train, y_val,
                            crystal_graphs_dict, args, gpu_id=None):
        """训练单个基模型 - 独立函数用于并行"""
        try:
            # 设置当前进程的GPU设备
            if gpu_id is not None and torch.cuda.is_available():
                device = torch.device(f'cuda:{gpu_id}')
                torch.cuda.set_device(device)
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 创建子进程专用日志
            logger = logging.getLogger(f"Model_{model_idx}")
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(f"model_{model_idx}_train.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)

            logger.info(f"开始训练基模型 {model_idx + 1} (设备: {device})")
            model = self.models[model_idx].to(device)

            # 转换为PyTorch张量
            X_train_tensor = torch.tensor(X_train, dtype=torch.float)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float)

            # 创建嵌入数据集
            def create_embedding_dataset(graphs, gcn_hidden_dim=256):
                embeds = []
                for graph in graphs:
                    mp_id = getattr(graph, 'mp_id', None)
                    embed = None

                    # 尝试从预计算嵌入中获取
                    if os.path.exists(EMBEDDING_CACHE):
                        with open(EMBEDDING_CACHE, 'rb') as f:
                            crystal_embeddings = pickle.load(f)
                        if mp_id and mp_id in crystal_embeddings:
                            embed = crystal_embeddings[mp_id]

                    # 如果仍未找到，创建默认嵌入
                    if embed is None:
                        embed = torch.zeros(1, gcn_hidden_dim)

                    # 确保嵌入维度正确
                    if embed.dim() == 1:
                        embed = embed.unsqueeze(0)
                    if embed.size(1) != gcn_hidden_dim:
                        embed = torch.zeros(1, gcn_hidden_dim)

                    embeds.append(embed)

                return torch.cat(embeds, dim=0).to(device)

            # 创建嵌入
            crystal_emb_train = create_embedding_dataset(graph_train)
            crystal_emb_val = create_embedding_dataset(graph_val)

            # 创建数据加载器
            train_dataset = TensorDataset(crystal_emb_train, X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(crystal_emb_val, X_val_tensor, y_val_tensor)

            batch_size = max(BATCH_SIZE, 2)  # 确保批次大小至少为2
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True)

            # 设置优化器和调度器
            base_params = []
            fusion_attn_params = []
            for name, param in model.named_parameters():
                if 'cross_attention' in name or 'final_predictor' in name:
                    fusion_attn_params.append(param)
                else:
                    base_params.append(param)

            optimizer = torch.optim.AdamW([
                {'params': base_params, 'lr': LEARNING_RATE, 'weight_decay': 5e-4},  # 增加权重衰减
                {'params': fusion_attn_params, 'lr': LEARNING_RATE * 0.5, 'weight_decay': 5e-4}
            ])

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.7, patience=100, min_lr=1e-6
            )

            # 损失函数
            def weighted_huber_loss(pred, target):
                weights = torch.where(target < 0.4, 2.5, 1.0)
                huber_loss = F.huber_loss(pred, target, reduction='none', delta=0.3)
                return (huber_loss * weights).mean()

            criterion = weighted_huber_loss

            # 训练循环 - 增强历史记录
            best_val_loss = float('inf')
            best_val_r2 = -float('inf')
            patience_counter = 0
            patience = 300

            history = {
                'train_loss': [],
                'val_loss': [],
                'train_r2': [],
                'val_r2': [],
                'lr': [],
                'epoch_time': []
            }

            for epoch in range(EPOCHS):
                epoch_start = time.time()
                model.train()
                train_loss = 0
                train_preds = []
                train_targets = []

                for crystal_emb, exp_feats, targets in train_loader:
                    crystal_emb = crystal_emb.to(device)
                    exp_feats = exp_feats.to(device)
                    targets = targets.to(device)

                    optimizer.zero_grad()
                    preds = model(crystal_emb, exp_feats)
                    loss = criterion(preds, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    train_loss += loss.item() * len(targets)
                    train_preds.append(preds.detach().cpu().numpy())
                    train_targets.append(targets.cpu().numpy())

                # 计算训练集R²
                train_preds = np.concatenate(train_preds)
                train_targets = np.concatenate(train_targets)
                train_r2 = r2_score(train_targets, train_preds) if len(train_targets) > 1 else 0.0

                # 验证
                model.eval()
                val_loss = 0
                val_preds = []
                val_targets = []

                with torch.no_grad():
                    for crystal_emb, exp_feats, targets in val_loader:
                        crystal_emb = crystal_emb.to(device)
                        exp_feats = exp_feats.to(device)
                        targets = targets.to(device)

                        preds = model(crystal_emb, exp_feats)
                        loss = criterion(preds, targets)
                        val_loss += loss.item() * len(targets)

                        val_preds.append(preds.cpu().numpy())
                        val_targets.append(targets.cpu().numpy())

                # 计算验证集R²
                val_preds = np.concatenate(val_preds)
                val_targets = np.concatenate(val_targets)
                val_r2 = r2_score(val_targets, val_preds) if len(val_targets) > 1 else 0.0

                # 计算指标
                train_loss_avg = train_loss / len(train_dataset)
                val_loss_avg = val_loss / len(val_dataset)

                # 更新历史
                history['train_loss'].append(train_loss_avg)
                history['val_loss'].append(val_loss_avg)
                history['train_r2'].append(train_r2)
                history['val_r2'].append(val_r2)
                history['lr'].append(optimizer.param_groups[0]['lr'])
                history['epoch_time'].append(time.time() - epoch_start)

                # 学习率调度
                scheduler.step(val_loss_avg)

                # 早停逻辑
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    best_val_r2 = val_r2
                    patience_counter = 0
                    torch.save(model.state_dict(), f'best_base_model_{model_idx}.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"基模型 {model_idx} 提前停止于epoch {epoch}")
                        break

                if (epoch + 1) % 50 == 0:
                    logger.info(f"基模型 {model_idx} Epoch {epoch + 1} | "
                                f"训练损失: {train_loss_avg:.4f} | 验证损失: {val_loss_avg:.4f} | "
                                f"训练R²: {train_r2:.4f} | 验证R²: {val_r2:.4f}")

            return {
                'model_idx': model_idx,
                'best_val_loss': best_val_loss,
                'best_val_r2': best_val_r2,
                'best_train_r2': history['train_r2'][-1] if history['train_r2'] else 0.0,
                'history': history
            }
        except Exception as e:
            logger.error(f"训练基模型 {model_idx} 失败: {str(e)}")
            if args.debug:
                logger.exception("详细错误信息:")
            return {
                'model_idx': model_idx,
                'error': str(e)
            }

    def train_base_models(self, X_experimental, crystal_graphs, y, crystal_graphs_dict, merged_df, args):
        """并行训练多个基模型并保存训练历史"""
        logger.info(f"开始并行训练集成模型，共{self.num_models}个基模型")
        logger.info(f"并行度: {args.parallel} 个模型同时训练")

        # 获取可用GPU
        available_gpus = []
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if args.gpu:
                available_gpus = [int(g) for g in args.gpu if int(g) < num_gpus]
            else:
                available_gpus = list(range(num_gpus))
            logger.info(f"可用GPU设备: {available_gpus}")
        else:
            logger.info("没有可用的GPU设备，使用CPU训练")

        # 使用K折交叉验证生成不同的训练集
        kf = KFold(n_splits=self.num_models, shuffle=True, random_state=42)
        fold_data = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_experimental)):
            # 划分当前折的训练集和验证集
            X_train, X_val = X_experimental[train_idx], X_experimental[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            graph_train = [crystal_graphs[i] for i in train_idx]
            graph_val = [crystal_graphs[i] for i in val_idx]

            fold_data.append({
                'fold_idx': fold_idx,
                'X_train': X_train,
                'X_val': X_val,
                'graph_train': graph_train,
                'graph_val': graph_val,
                'y_train': y_train,
                'y_val': y_val
            })

        # 使用多进程并行训练
        results = []
        start_time = time.time()

        # 创建进程池
        processes = []
        manager = mp.Manager()
        result_queue = manager.Queue()

        # 为每个模型分配GPU
        gpu_assignments = []
        for i in range(self.num_models):
            if available_gpus:
                gpu_id = available_gpus[i % len(available_gpus)]
            else:
                gpu_id = None
            gpu_assignments.append(gpu_id)

        # 启动训练进程
        for i in range(self.num_models):
            gpu_id = gpu_assignments[i]
            fold = fold_data[i]
            p = mp.Process(
                target=self._run_training,
                args=(
                    i,
                    fold['X_train'],
                    fold['X_val'],
                    fold['graph_train'],
                    fold['graph_val'],
                    fold['y_train'],
                    fold['y_val'],
                    crystal_graphs_dict,
                    args,
                    gpu_id,
                    result_queue
                )
            )
            p.start()
            processes.append(p)

            # 控制并行度
            if len(processes) >= args.parallel:
                for p in processes:
                    p.join()
                processes = []

        # 等待剩余进程完成
        for p in processes:
            p.join()

        # 收集结果
        while not result_queue.empty():
            results.append(result_queue.get())

        elapsed = time.time() - start_time
        logger.info(f"所有基模型训练完成 | 总耗时: {elapsed:.2f}秒 | 平均每个模型: {elapsed / self.num_models:.2f}秒")

        # 保存训练历史
        fold_results = []
        for result in results:
            if 'error' not in result:
                fold_results.append({
                    'best_val_loss': result['best_val_loss'],
                    'best_val_r2': result['best_val_r2'],
                    'best_train_r2': result['best_train_r2']
                })
                self.histories.append(result['history'])

                # 保存训练历史到JSON文件
                history_filename = f'base_model_{result["model_idx"]}_history.json'
                with open(history_filename, 'w') as f:
                    json.dump(result['history'], f, indent=4)
                logger.info(f"基模型 {result['model_idx']} 训练历史已保存到: {history_filename}")

        return fold_results

    def _run_training(self, model_idx, X_train, X_val, graph_train, graph_val, y_train, y_val,
                      crystal_graphs_dict, args, gpu_id, result_queue):
        """在单独进程中运行训练"""
        try:
            # 设置GPU设备
            if gpu_id is not None and torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                device = torch.device(f'cuda:{gpu_id}')
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 训练模型
            result = self._train_single_model(
                model_idx, X_train, X_val, graph_train, graph_val,
                y_train, y_val, crystal_graphs_dict, args, gpu_id
            )
            result_queue.put(result)
        except Exception as e:
            result_queue.put({
                'model_idx': model_idx,
                'error': str(e)
            })

    def evaluate_model(self, model_idx, dataset_loader, dataset_name):
        """评估单个基模型在指定数据集上的性能"""
        model = self.models[model_idx]
        model.load_state_dict(torch.load(f'best_base_model_{model_idx}.pth'))
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for crystal_emb, exp_feats, targets in dataset_loader:
                crystal_emb = crystal_emb.to(self.device)
                exp_feats = exp_feats.to(self.device)
                targets = targets.to(self.device)

                preds = model(crystal_emb, exp_feats)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # 计算指标
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        r2 = r2_score(all_targets, all_preds)

        return {
            'dataset': dataset_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': all_preds,
            'targets': all_targets
        }

    def save_ensemble_predictions(self, base_preds, ensemble_pred, y_test, merged_df, indices, base_train_preds=None,
                                  y_train=None, train_indices=None):
        """保存集成模型的预测结果 - 增加训练集预测"""
        try:
            # 创建结果目录
            results_dir = "ensemble_results"
            os.makedirs(results_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # 保存每个基模型的预测结果
            base_predictions_df = pd.DataFrame()
            for i, preds in enumerate(base_preds):
                model_name = f'Base_Model_{i + 1}'
                base_predictions_df[model_name] = preds

                # 保存单个模型的详细预测
                model_df = merged_df.iloc[indices].copy()
                model_df['True_Cell_Viability'] = y_test
                model_df['Predicted_Cell_Viability'] = preds
                model_df['Model'] = model_name
                model_df.to_csv(os.path.join(results_dir, f"{model_name}_test_predictions_{timestamp}.csv"),
                                index=False)

                # 保存训练集预测
                if base_train_preds and i < len(base_train_preds):
                    train_preds = base_train_preds[i]
                    train_df = merged_df.iloc[train_indices].copy()
                    train_df['True_Cell_Viability'] = y_train
                    train_df['Predicted_Cell_Viability'] = train_preds
                    train_df['Model'] = model_name
                    train_df.to_csv(os.path.join(results_dir, f"{model_name}_train_predictions_{timestamp}.csv"),
                                    index=False)

            # 保存集成模型的预测结果
            ensemble_df = merged_df.iloc[indices].copy()
            ensemble_df['True_Cell_Viability'] = y_test
            ensemble_df['Predicted_Cell_Viability'] = ensemble_pred
            ensemble_df['Model'] = 'Ensemble'
            ensemble_df.to_csv(os.path.join(results_dir, f"Ensemble_test_predictions_{timestamp}.csv"), index=False)

            # 保存所有预测结果
            all_predictions_df = base_predictions_df.copy()
            all_predictions_df['Ensemble'] = ensemble_pred
            all_predictions_df['True_Cell_Viability'] = y_test
            all_predictions_df.to_csv(os.path.join(results_dir, f"All_Test_Predictions_{timestamp}.csv"), index=False)

            logger.info(f"所有预测结果已保存到目录: {results_dir}")

            return os.path.abspath(results_dir)
        except Exception as e:
            logger.error(f"保存集成预测结果失败: {str(e)}")
            return None

    def save_ensemble_metrics(self, base_metrics, ensemble_metrics):
        """保存所有模型的性能指标"""
        try:
            # 准备基模型指标
            metrics_data = []

            for i, metrics in enumerate(base_metrics):
                for dataset_metrics in metrics:
                    metrics_data.append({
                        'model_type': 'Base',
                        'model_index': i + 1,
                        'dataset': dataset_metrics['dataset'],
                        'mae': dataset_metrics['mae'],
                        'rmse': dataset_metrics['rmse'],
                        'r2': dataset_metrics['r2']
                    })

            # 添加集成模型指标
            for dataset_metrics in ensemble_metrics:
                metrics_data.append({
                    'model_type': 'Ensemble',
                    'model_index': 'N/A',
                    'dataset': dataset_metrics['dataset'],
                    'mae': dataset_metrics['mae'],
                    'rmse': dataset_metrics['rmse'],
                    'r2': dataset_metrics['r2']
                })

            # 创建DataFrame
            metrics_df = pd.DataFrame(metrics_data)

            # 保存到CSV
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ensemble_performance_metrics_{timestamp}.csv"
            metrics_df.to_csv(filename, index=False)

            # 打印摘要
            self.print_performance_summary(base_metrics, ensemble_metrics)

            return metrics_df
        except Exception as e:
            logger.error(f"保存性能指标失败: {str(e)}")
            return None

    def print_performance_summary(self, base_metrics, ensemble_metrics):
        """打印详细的性能摘要"""
        logger.info("\n" + "=" * 50)
        logger.info("集成模型性能摘要")
        logger.info("=" * 50)

        # 打印基模型性能
        logger.info("\n基模型性能:")
        for i, model_metrics in enumerate(base_metrics):
            logger.info(f"\n模型 {i + 1}:")
            for dataset_metrics in model_metrics:
                logger.info(f"  {dataset_metrics['dataset']}集: "
                            f"MAE={dataset_metrics['mae']:.4f}, "
                            f"RMSE={dataset_metrics['rmse']:.4f}, "
                            f"R²={dataset_metrics['r2']:.4f}")

        # 打印集成模型性能
        logger.info("\n集成模型性能:")
        for dataset_metrics in ensemble_metrics:
            logger.info(f"  {dataset_metrics['dataset']}集: "
                        f"MAE={dataset_metrics['mae']:.4f}, "
                        f"RMSE={dataset_metrics['rmse']:.4f}, "
                        f"R²={dataset_metrics['r2']:.4f}")

        # 计算平均改进
        base_test_r2 = np.mean([m[2]['r2'] for m in base_metrics])  # 测试集是第三个
        ensemble_test_r2 = ensemble_metrics[2]['r2']  # 测试集是第三个
        improvement = (ensemble_test_r2 - base_test_r2) * 100

        logger.info("\n" + "-" * 50)
        logger.info(f"测试集平均R²改进: {improvement:.2f}%")
        logger.info(f"基模型平均测试R²: {base_test_r2:.4f}")
        logger.info(f"集成模型测试R²: {ensemble_test_r2:.4f}")
        logger.info("=" * 50 + "\n")

    def create_embedding_dataset(self, graphs, gcn_hidden_dim=256):
        """创建嵌入数据集 - 供内部使用"""
        embeds = []
        for graph in graphs:
            mp_id = getattr(graph, 'mp_id', None)
            embed = None

            # 尝试从预计算嵌入中获取
            if os.path.exists(EMBEDDING_CACHE):
                with open(EMBEDDING_CACHE, 'rb') as f:
                    crystal_embeddings = pickle.load(f)
                if mp_id and mp_id in crystal_embeddings:
                    embed = crystal_embeddings[mp_id]

            # 如果仍未找到，创建默认嵌入
            if embed is None:
                embed = torch.zeros(1, gcn_hidden_dim)

            # 确保嵌入维度正确
            if embed.dim() == 1:
                embed = embed.unsqueeze(0)
            if embed.size(1) != gcn_hidden_dim:
                embed = torch.zeros(1, gcn_hidden_dim)

            embeds.append(embed)

        return torch.cat(embeds, dim=0)


# 修改主训练函数以支持集成学习
@error_handler
def main_training_flow(args, X_experimental, crystal_graphs, y, crystal_graphs_dict, merged_df, device):
    if args.ensemble:
        logger.info("使用集成学习模式")
        # 划分数据集 - 80%训练+验证, 20%测试
        indices = np.arange(len(y))
        X_train, X_test, graph_train, graph_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_experimental, crystal_graphs, y, indices, test_size=0.2, random_state=42, stratify=(y > 0.5).astype(int))

        # 在训练集上拟合特征选择器 (修复数据泄露问题)
        selector = SelectKBest(f_regression, k=min(1200, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        logger.info(f"特征选择后维度: {X_train_selected.shape[1]}")

        # 初始化集成模型
        exp_feature_dim = X_train_selected.shape[1]
        ensemble = CrystalToxicityEnsemble(
            num_models=7,  # 增加基模型数量
            exp_feature_dim=exp_feature_dim,
            device=device
        )

        # 并行训练基模型
        fold_results = ensemble.train_base_models(
            X_train_selected, graph_train, y_train,
            crystal_graphs_dict, merged_df, args
        )

        # 创建数据集加载器
        def create_dataset_loader(X, graphs, y, batch_size=BATCH_SIZE):
            """创建数据加载器"""
            # 创建嵌入
            crystal_emb = ensemble.create_embedding_dataset(graphs)
            X_tensor = torch.tensor(X, dtype=torch.float)
            y_tensor = torch.tensor(y, dtype=torch.float)
            dataset = TensorDataset(crystal_emb, X_tensor, y_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # 训练集和测试集加载器
        train_loader = create_dataset_loader(X_train_selected, graph_train, y_train)
        test_loader = create_dataset_loader(X_test_selected, graph_test, y_test)

        # 评估每个基模型在所有数据集上的性能
        base_metrics_all = []
        base_preds_test = []
        base_preds_train = []

        # 顺序评估模型（避免并行评估时的GPU冲突）
        for model_idx in range(ensemble.num_models):
            logger.info(f"\n评估基模型 {model_idx + 1} 在所有数据集上的性能...")
            model_metrics = []

            # 评估训练集
            train_metrics = ensemble.evaluate_model(model_idx, train_loader, "train")
            model_metrics.append(train_metrics)
            base_preds_train.append(train_metrics['predictions'])

            # 评估测试集
            test_metrics = ensemble.evaluate_model(model_idx, test_loader, "test")
            model_metrics.append(test_metrics)
            base_preds_test.append(test_metrics['predictions'])

            base_metrics_all.append(model_metrics)

            logger.info(f"基模型 {model_idx + 1} 测试集结果: "
                        f"MAE={test_metrics['mae']:.4f}, "
                        f"RMSE={test_metrics['rmse']:.4f}, "
                        f"R²={test_metrics['r2']:.4f}")

        # 集成预测 - 获取每个基模型的预测结果
        logger.info("进行集成预测...")
        ensemble_pred_train = np.mean(base_preds_train, axis=0)
        ensemble_pred_test = np.mean(base_preds_test, axis=0)

        # 评估集成模型在所有数据集上的性能
        ensemble_metrics_all = []

        # 训练集评估
        ensemble_metrics_all.append({
            'dataset': 'train',
            'mae': mean_absolute_error(y_train, ensemble_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, ensemble_pred_train)),
            'r2': r2_score(y_train, ensemble_pred_train)
        })

        # 测试集评估
        ensemble_metrics_all.append({
            'dataset': 'test',
            'mae': mean_absolute_error(y_test, ensemble_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred_test)),
            'r2': r2_score(y_test, ensemble_pred_test)
        })

        # 保存性能指标
        metrics_df = ensemble.save_ensemble_metrics(base_metrics_all, ensemble_metrics_all)

        # 保存预测结果 (包括训练集预测)
        results_dir = ensemble.save_ensemble_predictions(
            base_preds_test, ensemble_pred_test, y_test, merged_df, idx_test,
            base_train_preds=base_preds_train, y_train=y_train, train_indices=idx_train
        )

        logger.info("\n===== 集成模型最终测试结果 =====")
        logger.info(f"MAE: {ensemble_metrics_all[1]['mae']:.4f}")
        logger.info(f"RMSE: {ensemble_metrics_all[1]['rmse']:.4f}")
        logger.info(f"R²: {ensemble_metrics_all[1]['r2']:.4f}")
        logger.info(f"详细结果保存到: {results_dir}")

        return {
            'metrics': metrics_df,
            'ensemble_pred': ensemble_pred_test,
            'base_preds': base_preds_test,
            'y_test': y_test
        }
    else:
        # 原有单模型训练逻辑
        return train_and_evaluate(
            X_experimental, crystal_graphs, y,
            crystal_graphs_dict, device, merged_df, args
        )


def save_predictions(predictions, merged_df, epoch, model_idx=0):
    """保存预测结果到CSV文件"""
    try:
        # 创建结果DataFrame
        results = []

        # 确定文件名
        if isinstance(epoch, str):  # 集成模型
            filename = f"ensemble_predictions.csv"
            full_filename = f"full_ensemble_predictions.csv"
        else:
            if model_idx > 0:
                filename = f"predictions_model{model_idx}_epoch{epoch}.csv"
                full_filename = f"full_predictions_model{model_idx}_epoch{epoch}.csv"
            else:
                filename = f"predictions_epoch{epoch}.csv"
                full_filename = f"full_predictions_epoch{epoch}.csv"

        # 为每个数据集创建结果
        for dataset in ['train', 'val', 'test']:
            indices = predictions[dataset].get('indices', [])
            true_vals = predictions[dataset].get('true', [])
            pred_vals = predictions[dataset].get('pred', [])

            # 确保长度匹配
            n_samples = len(true_vals)
            if n_samples == 0:
                logger.warning(f"{dataset} 数据集没有预测结果")
                continue

            # 创建该数据集的DataFrame
            df = pd.DataFrame({
                'Dataset': [dataset] * n_samples,
                'True_Cell_Viability': true_vals,
                'Predicted_Cell_Viability': pred_vals
            })
            results.append(df)

        # 合并所有结果
        if results:
            results_df = pd.concat(results, ignore_index=True)

            # 保存到CSV
            results_df.to_csv(filename, index=False)
            logger.info(f"预测结果已保存到: {filename}")

            # 保存完整结果（包含原始特征）
            full_results = merged_df.copy()
            full_results['Dataset'] = ''
            full_results['Predicted_Cell_Viability'] = np.nan

            # 使用索引匹配预测结果到原始数据
            for dataset in ['train', 'val', 'test']:
                indices = predictions[dataset].get('indices', [])
                pred_vals = predictions[dataset].get('pred', [])

                if len(indices) > 0:
                    # 通过索引直接赋值
                    for idx, pred_val in zip(indices, pred_vals):
                        if idx < len(full_results):
                            full_results.at[idx, 'Dataset'] = dataset
                            full_results.at[idx, 'Predicted_Cell_Viability'] = pred_val
                        else:
                            logger.warning(f"索引 {idx} 超出范围 (数据集大小: {len(full_results)})")

            full_results.to_csv(full_filename, index=False)
            logger.info(f"完整预测结果（含原始特征）已保存到: {full_filename}")

            # 额外保存绘图专用数据
            plot_df = pd.DataFrame({
                'True': np.concatenate([
                    predictions['train']['true'],
                    predictions['val']['true'],
                    predictions['test']['true']
                ]),
                'Predicted': np.concatenate([
                    predictions['train']['pred'],
                    predictions['val']['pred'],
                    predictions['test']['pred']
                ]),
                'Dataset': ['train'] * len(predictions['train']['true']) +
                           ['val'] * len(predictions['val']['true']) +
                           ['test'] * len(predictions['test']['true'])
            })
            plot_filename = f"plot_data_{filename}"
            plot_df.to_csv(plot_filename, index=False)
            logger.info(f"绘图专用数据已保存到: {plot_filename}")
    except Exception as e:
        logger.error(f"保存预测结果失败: {str(e)}")


def evaluate_model_performance(predictions, epoch, model_idx=0):
    """评估模型性能并保存指标 - 修复版本"""
    try:
        performance = []
        for dataset in ['train', 'val', 'test']:
            true = np.array(predictions[dataset].get('true', []))
            pred = np.array(predictions[dataset].get('pred', []))

            if len(true) == 0:
                continue

            mae = mean_absolute_error(true, pred)
            rmse = np.sqrt(mean_squared_error(true, pred))
            r2 = r2_score(true, pred)

            performance.append({
                'Dataset': dataset,
                'Samples': len(true),
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            })

        if performance:
            perf_df = pd.DataFrame(performance)
            timestamp = time.strftime("%Y%m%d-%H%M%S")

            # 添加模型索引到文件名
            if model_idx > 0:
                filename = f"performance_metrics_model{model_idx}_epoch{epoch}_{timestamp}.csv"
            else:
                filename = f"performance_metrics_epoch{epoch}_{timestamp}.csv"

            perf_df.to_csv(filename, index=False)
            logger.info(f"模型性能指标已保存到: {filename}")

            # 打印性能摘要
            logger.info("\n模型性能摘要:")
            for row in performance:
                logger.info(f"{row['Dataset']}集 (样本数: {row['Samples']}):")
                logger.info(f"  MAE: {row['MAE']:.4f}, RMSE: {row['RMSE']:.4f}, R^2: {row['R2']:.4f}")
    except Exception as e:
        logger.error(f"评估模型性能失败: {str(e)}")


def plot_results(predictions, history, best_epoch, model_idx=0):
    """生成多种可视化图表"""
    try:
        logger.info("生成可视化图表...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plot_dir = f"plots_epoch{best_epoch}_{timestamp}"
        os.makedirs(plot_dir, exist_ok=True)

        # 1. 损失曲线
        loss_curve_name = f"loss_curve_{model_idx}.png"
        plt.figure(figsize=(12, 8))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Huber)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, loss_curve_name), dpi=300)
        plt.close()

        # 2. 学习率曲线
        learning_rate_name = f"learning_rate_{model_idx}.png"
        plt.figure(figsize=(12, 6))
        plt.plot(history['lr'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, learning_rate_name), dpi=300)
        plt.close()

        # 3. 真实值 vs 预测值散点图
        plt.figure(figsize=(10, 8))
        all_true = []
        all_pred = []

        for dataset, color, marker in zip(['train', 'val', 'test'],
                                          ['blue', 'green', 'red'],
                                          ['o', 's', '^']):
            true = predictions[dataset].get('true', [])
            pred = predictions[dataset].get('pred', [])

            if true and pred:
                plt.scatter(true, pred, alpha=0.6, label=dataset, c=color, marker=marker)
                all_true.extend(true)
                all_pred.extend(pred)

        if all_true and all_pred:
            # 计算全局最小值和最大值
            min_val = min(min(all_true), min(all_pred))
            max_val = max(max(all_true), max(all_pred))
            margin = 0.05 * (max_val - min_val)

            # 添加对角线
            plt.plot([min_val - margin, max_val + margin],
                     [min_val - margin, max_val + margin],
                     'k--', alpha=0.5)
            true_vs_predicted_name = f"true_vs_predicted_{model_idx}.png"
            plt.xlabel('True Cell Viability')
            plt.ylabel('Predicted Cell Viability')
            plt.title('True vs Predicted Values')
            plt.legend()
            plt.grid(True)
            plt.xlim(min_val - margin, max_val + margin)
            plt.ylim(min_val - margin, max_val + margin)
            plt.savefig(os.path.join(plot_dir, true_vs_predicted_name), dpi=300)
            plt.close()
        else:
            logger.warning("没有足够的数据绘制真实值 vs 预测值散点图")

        # 4. 残差分布图
        plt.figure(figsize=(10, 6))
        for dataset, color in zip(['train', 'val', 'test'], ['blue', 'green', 'red']):
            true = np.array(predictions[dataset].get('true', []))
            pred = np.array(predictions[dataset].get('pred', []))
            if len(true) > 0 and len(pred) > 0:
                residuals = true - pred
                sns.histplot(residuals, label=dataset, kde=True, alpha=0.3, color=color)
        residual_distribution_name = f"residual_distribution_{model_idx}.png"
        plt.xlabel('Residuals (True - Predicted)')
        plt.ylabel('Density')
        plt.title('Residual Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, residual_distribution_name), dpi=300)
        plt.close()

        # 5. 残差 vs 预测值
        plt.figure(figsize=(10, 8))
        for dataset, color in zip(['train', 'val', 'test'], ['blue', 'green', 'red']):
            true = np.array(predictions[dataset].get('true', []))
            pred = np.array(predictions[dataset].get('pred', []))
            if len(true) > 0 and len(pred) > 0:
                residuals = true - pred
                plt.scatter(pred, residuals, alpha=0.5, label=dataset, c=color)

        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Predicted Cell Viability')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'residuals_vs_predicted.png'), dpi=300)
        plt.close()

        # 6. 按数据集划分的预测误差分布
        plt.figure(figsize=(10, 6))
        errors = []
        labels = []
        for dataset in ['train', 'val', 'test']:
            true = np.array(predictions[dataset].get('true', []))
            pred = np.array(predictions[dataset].get('pred', []))
            if len(true) > 0 and len(pred) > 0:
                errors.append(np.abs(true - pred))
                labels.append(dataset)

        if errors:
            # 修复Matplotlib警告
            plt.boxplot(errors)
            plt.xticks(range(1, len(labels) + 1), labels)
            plt.ylabel('Absolute Error')
            plt.title('Prediction Error Distribution by Dataset')
            plt.grid(True)
            plt.savefig(os.path.join(plot_dir, 'error_distribution.png'), dpi=300)
            plt.close()
        else:
            logger.warning("没有足够的数据绘制误差分布箱线图")

        logger.info(f"所有图表已保存到目录: {plot_dir}")

    except Exception as e:
        logger.error(f"生成图表失败: {str(e)}")
        if os.path.exists(plot_dir):
            logger.info(f"部分图表可能已保存到: {plot_dir}")


def main():
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)

    args = parse_args()

    logger.info("=" * 50)
    logger.info("晶体毒性预测模型 - 增强版")
    logger.info("=" * 50)

    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"图构建方法: {args.graph_method}")
    if args.ensemble:
        logger.info("使用集成学习策略")

    # 下载模式
    if args.download:
        logger.info("运行下载模式...")
        load_and_preprocess_data(args)
        logger.info("下载完成，准备离线训练")
        sys.exit(0)

    # 离线模式检查
    if args.offline:
        logger.info("运行离线模式 (使用缓存数据)")
        if not os.path.exists(CRYSTAL_DATA_CACHE):
            logger.error(f"找不到晶体缓存文件: {CRYSTAL_DATA_CACHE}")
            sys.exit(1)

    try:
        # 加载并预处理数据
        logger.info("加载数据...")
        result = load_and_preprocess_data(args)
        if result is None:
            sys.exit(0)  # 仅下载模式已退出

        X_experimental, crystal_graphs, y, cat_processor, crystal_graphs_dict, merged_df = result

        # 训练模型（支持集成）
        if args.ensemble:
            logger.info("使用集成学习策略训练模型...")
            results = main_training_flow(args, X_experimental, crystal_graphs, y, crystal_graphs_dict, merged_df,
                                         device)

            logger.info("集成模型已保存")
        else:
            logger.info("训练单个模型...")
            model = train_and_evaluate(
                X_experimental, crystal_graphs, y,
                crystal_graphs_dict, device, merged_df, args
            )

            # 保存完整模型
            logger.info("保存完整模型...")
            torch.save({
                'model_state_dict': model.state_dict(),
                'cat_processor': cat_processor,
                'config': {
                    'exp_feature_dim': X_experimental.shape[1],
                    'gcn_hidden_dim': 256
                }
            }, 'crystal_toxicity_model_enhanced.pt')

        logger.info("训练完成!")
    except Exception as e:
        logger.error(f"主程序执行失败: {str(e)}")
        if args.debug:
            logger.exception("完整错误信息:")
        sys.exit(1)


# 4. 主执行流程
if __name__ == "__main__":
    main()