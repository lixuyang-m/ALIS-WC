# This file is derived from the HeteroMRTA codebase by Dai et al.
# (IEEE RA-L 2025), originally released under the Apache-2.0 License.
# Original source: https://github.com/marmotlab/HeteroMRTA
# Modifications: LTL encoding variants, sleep-wake masking integration,
#   multi-head PPO policy factorisation, cost-quantile critic head.

import torch
import torch.nn as nn
import math
import numpy as np
from parameters import *


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.sum(dim=2).size()
    batch_size, len_k = seq_k.sum(dim=2).size()
    # eq(zero) is PAD token
    pad_attn_mask_k = (
        seq_q.eq(0).all(2).data.eq(1).unsqueeze(1)
    )  # batch_size x 1 x len_q, one is masking
    pad_attn_mask_q = (
        seq_k.eq(0).all(2).data.eq(1).unsqueeze(1)
    )  # batch_size x 1 x len_k, one is masking
    pad_attn_mask_k = pad_attn_mask_k.expand(batch_size, len_k, len_q).permute(0, 2, 1)
    pad_attn_mask_q = pad_attn_mask_q.expand(batch_size, len_q, len_k)
    return ~torch.logical_and(
        ~pad_attn_mask_k, ~pad_attn_mask_q
    )  # batch_size x len_q x len_k


class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        # 【新增】为输入 Query 和 Key/Value 添加独立的层归一化
        self.q_layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.kv_layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q

        # 【新增】在进行任何计算之前，对输入进行归一化
        q_norm = self.q_layer_norm(q)
        h_norm = self.kv_layer_norm(h)

        batch_size, target_size, input_dim = h_norm.size()
        n_query = q_norm.size(1)

        h_flat = h_norm.reshape(-1, input_dim)
        q_flat = q_norm.reshape(-1, input_dim)

        # batch_size, target_size, input_dim = h.size()
        # n_query = q.size(1)
        #
        # h_flat = h.reshape(-1, input_dim)
        # q_flat = q.reshape(-1, input_dim)

        shape_k = (batch_size, target_size, -1)
        shape_q = (batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(h_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            mask = mask.view(batch_size, -1, target_size).expand_as(U)
            U[mask.bool()] = -1e9

        attention = torch.softmax(U, dim=-1)
        logp_list = torch.log_softmax(U, dim=-1)

        # return attention, logp_list

        return attention, U


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.key_dim)
        )
        self.w_key = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.key_dim)
        )
        self.w_value = nn.Parameter(
            torch.Tensor(self.n_heads, self.input_dim, self.value_dim)
        )
        self.w_out = nn.Parameter(
            torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim)
        )

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1.0 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)

        h_flat = h.contiguous().view(-1, input_dim)
        q_flat = q.contiguous().view(-1, input_dim)
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(h_flat, self.w_key).view(shape_k)
        V = torch.matmul(h_flat, self.w_value).view(shape_v)

        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)
            U[mask.bool()] = -np.inf
        attention = torch.softmax(U, dim=-1)

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc

        heads = torch.matmul(attention, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            self.w_out.view(-1, self.embedding_dim),
        ).view(batch_size, n_query, self.embedding_dim)

        return out


class GateFFNDense(nn.Module):
    def __init__(self, model_dim, hidden_unit=512):
        super(GateFFNDense, self).__init__()
        self.W = nn.Linear(model_dim, hidden_unit, bias=False)
        self.V = nn.Linear(model_dim, hidden_unit, bias=False)
        self.W2 = nn.Linear(hidden_unit, model_dim, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, hidden_states):
        hidden_act = self.act(self.W(hidden_states))
        hidden_linear = self.V(hidden_states)
        hidden_states = hidden_act * hidden_linear
        hidden_states = self.W2(hidden_states)
        return hidden_states


class GateFFNLayer(nn.Module):
    def __init__(self, model_dim):
        super(GateFFNLayer, self).__init__()
        self.DenseReluDense = GateFFNDense(model_dim)
        self.layer_norm = Normalization(model_dim)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        return forwarded_states


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = GateFFNLayer(embedding_dim)

    def forward(self, src, mask=None):
        h0 = src
        h = self.normalization1(src)
        h = self.multiHeadAttention(q=h, mask=mask)
        h = h + h0
        h1 = h
        h = self.feedForward(h)
        h = h + h1
        return h


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(embedding_dim, n_head)
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.feedForward = GateFFNLayer(embedding_dim)
        self.normalization1 = Normalization(embedding_dim)
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, dec_self_attn_mask, dec_enc_attn_mask):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization2(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=dec_enc_attn_mask)
        h = h + h0
        h1 = h
        h = self.feedForward(h)
        h = h + h1
        return h


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4, n_layer=2):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(embedding_dim, n_head) for i in range(n_layer)
        )

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=4, n_layer=2):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(embedding_dim, n_head) for i in range(n_layer)]
        )

    def forward(self, tgt, memory, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, dec_self_attn_mask, dec_enc_attn_mask)
        return tgt


class AttentionNet(nn.Module):
    def __init__(self, agent_input_dim, task_input_dim, embedding_dim):
        super(AttentionNet, self).__init__()
        self.agent_embedding = nn.Linear(agent_input_dim, embedding_dim)
        self.task_embedding = nn.Linear(task_input_dim, embedding_dim)
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

        self.taskEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.crossDecoder1 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.crossDecoder2 = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)
        self.agentEncoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.globalDecoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=2)

        self.pointer = SingleHeadAttention(embedding_dim)

        # Multi-head output layers
        self.action_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 3),  # 3 actions: MOVE, LOAD, UNLOAD
        )

        self.cargo_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, EnvParams.TRAIT_DIM),
        )

        # 分离的Critic头：奖励期望 + 成本分布
        self.reward_critic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )
        self.cost_critic = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, TrainParams.NUM_QUANTILES),
        )

        # ===== 【方案一：添加GAT模块处理任务依赖图】 =====
        if TrainParams.LTL_ENABLED:
            from torch_geometric.nn import GATConv

            # 2层GAT，每层4个attention头
            self.task_dependency_gat = nn.ModuleList(
                [
                    GATConv(
                        embedding_dim,
                        embedding_dim,
                        heads=4,
                        concat=False,
                        add_self_loops=True,
                    )
                    for _ in range(2)
                ]
            )
            # Layer normalization for GAT
            self.gat_layer_norms = nn.ModuleList(
                [nn.LayerNorm(embedding_dim) for _ in range(2)]
            )

        if TrainParams.LTL_ENABLED:
            if TrainParams.LTL_ENCODING_TYPE == "A":
                # 【方案A：ID-specific编码 + 智能体状态】
                # 稀疏编码格式（扩展版本）：
                # 前4维：[type, param1_norm, param2_norm, state_norm]
                # - type ∈ {0,1}：约束类型（0=SAFETY, 1=SEQUENTIAL）
                # - param1_norm, param2_norm ∈ [0,1]：归一化的任务/智能体ID
                # - state_norm ∈ [0,1]：归一化的FSA状态（支持4个状态）
                # 后max_agents*3维：每个智能体的三状态one-hot编码
                # - 每3个连续位表示一个智能体：[active, temp_sleeping, inactive]

                # 计算输入维度：4 + max_agents*3
                max_agents = (
                    EnvParams.SPECIES_RANGE[1] * EnvParams.SPECIES_AGENTS_RANGE[1]
                )
                ltl_input_dim = 4 + max_agents * 3

                # 嵌入层：将扩展的稀疏向量映射到embedding维度
                self.ltl_embedding = nn.Linear(ltl_input_dim, embedding_dim)

                # 保留交叉注意力模块用于融合LTL信息和智能体状态
                self.ltl_fusion_attention = SingleHeadAttention(embedding_dim)

            elif TrainParams.LTL_ENCODING_TYPE == "B":
                # 【方案B：Task feasibility编码 - 增强版】
                # 输入格式：[max_agents, max_tasks]，0=可行，1=不可行
                # 【修改】保留完整矩阵，让当前agent看到所有agent的LTL约束

                # 1. 任务可行性嵌入：将0/1标量映射到embedding维度
                self.task_feasibility_embedding = nn.Linear(1, embedding_dim)

                # 2. Agent-level LTL aggregation：聚合每个agent的所有任务约束
                # 输入：每个agent的task feasibility embeddings [max_tasks, embedding_dim]
                # 输出：该agent的LTL约束汇总 [1, embedding_dim]
                self.agent_ltl_aggregation = MultiHeadAttention(
                    embedding_dim, n_heads=4
                )

                # 3. Cross-agent LTL attention：让当前agent关注所有agent的LTL约束
                # 输入：所有agent的LTL汇总 [max_agents, embedding_dim]
                # 输出：考虑全局约束的LTL表示 [1, embedding_dim]
                self.cross_agent_ltl_attention = MultiHeadAttention(
                    embedding_dim, n_heads=4
                )

                # 4. 融合层：将attention后的LTL信息融入agent状态
                self.ltl_fusion = nn.Sequential(
                    nn.Linear(embedding_dim * 2, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )

            elif TrainParams.LTL_ENCODING_TYPE == "C":
                # 【方案C：Task feasibility + Dependency graph（连续边权重）】
                # 输入格式：
                # - feasibility矩阵: [max_agents, max_tasks]
                # - edge_index: [2, E] 稀疏边列表
                # - edge_attr: [E, 1] 连续边权重（阻塞度 ∈ [0,1]）

                # 1. Task feasibility部分（与B相同）
                self.task_feasibility_embedding = nn.Linear(1, embedding_dim)
                self.agent_ltl_aggregation = MultiHeadAttention(
                    embedding_dim, n_heads=4
                )
                self.cross_agent_ltl_attention = MultiHeadAttention(
                    embedding_dim, n_heads=4
                )

                # 2. Dependency graph部分（新增GAT层）
                # 边特征嵌入：将连续权重映射到edge embedding
                self.edge_attr_embedding = nn.Linear(1, embedding_dim)

                # GAT层：利用edge_index和edge_attr进行图传播
                # 注意：这里使用简化的GAT，在forward中手动实现
                # 因为edge_attr需要动态处理
                self.gat_key = nn.Linear(embedding_dim, embedding_dim)
                self.gat_query = nn.Linear(embedding_dim, embedding_dim)
                self.gat_value = nn.Linear(embedding_dim, embedding_dim)
                self.gat_edge = nn.Linear(embedding_dim, embedding_dim)  # 边特征变换

                # 3. 融合层：结合feasibility和dependency两部分信息
                # 【修复】更新维度以匹配三部分信息拼接：current_state + feasibility + dependency
                self.ltl_fusion = nn.Sequential(
                    nn.Linear(embedding_dim * 3, embedding_dim),
                    nn.ReLU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
            else:
                raise ValueError(
                    f"Unknown LTL_ENCODING_TYPE: {TrainParams.LTL_ENCODING_TYPE}"
                )

    def encoding_tasks(self, task_inputs, dependency_adjacency=None, mask=None):
        """
        【方案一+二：融合任务特征编码和依赖图传播】

        Args:
            task_inputs: [B, N, D] 任务输入特征（已包含LTL状态特征）
            dependency_adjacency: [B, N, N] or [N, N] 任务依赖邻接矩阵（可选）
            mask: 填充mask

        Returns:
            aggregated_tasks: [B, 1, embedding_dim]
            task_encoding: [B, N, embedding_dim] (已融合依赖信息)
        """
        # 基础任务嵌入
        task_embedding = self.task_embedding(task_inputs)  # [B, N, embedding_dim]

        # ===== 【方案一关键】：GAT传播依赖信息 =====
        if TrainParams.LTL_ENABLED and dependency_adjacency is not None:
            task_embedding_with_deps = self._apply_gat(
                task_embedding, dependency_adjacency
            )
        else:
            task_embedding_with_deps = task_embedding

        # 原有的Self-Attention编码
        task_encoding = self.taskEncoder(task_embedding_with_deps, mask)

        # 聚合
        embedding_dim = task_encoding.size(-1)
        mean_mask = mask[:, 0, :].unsqueeze(2).repeat(1, 1, embedding_dim)
        # 【修复】使用task_encoding而不是task_embedding，确保聚合的特征包含GAT信息
        compressed_task = torch.where(mean_mask, torch.nan, task_encoding)
        aggregated_tasks = torch.nanmean(compressed_task, dim=1).unsqueeze(1)

        ## FIX: Replace any potential NaNs with 0.0 to ensure numerical stability.
        aggregated_tasks = torch.nan_to_num(aggregated_tasks, nan=0.0)

        return aggregated_tasks, task_encoding

    def _apply_gat(self, task_embedding, dependency_adjacency):
        """
        【方案一核心】：使用GAT在任务依赖图上传播信息

        Args:
            task_embedding: [B, N, embedding_dim] 任务嵌入
            dependency_adjacency: [B, N, N] or [N, N] 邻接矩阵

        Returns:
            task_embedding_with_deps: [B, N, embedding_dim] 融合依赖信息后的嵌入
        """
        batch_size, num_tasks, embedding_dim = task_embedding.shape

        # 处理邻接矩阵维度
        if dependency_adjacency.dim() == 2:
            # [N, N] -> [B, N, N]
            dependency_adjacency = dependency_adjacency.unsqueeze(0).expand(
                batch_size, -1, -1
            )

        # 转换邻接矩阵为edge_index格式 (PyG要求)
        # 【修复】：传递实际的num_tasks（来自task_embedding，包含padding）
        edge_index, edge_weight = self._adjacency_to_edge_index(
            dependency_adjacency, num_tasks
        )

        # 将batch展平: [B, N, D] -> [B*N, D]
        x = task_embedding.view(batch_size * num_tasks, embedding_dim)

        # GAT传播（2层）
        for i, (gat_layer, layer_norm) in enumerate(
            zip(self.task_dependency_gat, self.gat_layer_norms)
        ):
            x_residual = x

            # 【修复】Pre-LN：先归一化，再GAT，最后残差
            x = layer_norm(x)
            x = gat_layer(x, edge_index, edge_attr=edge_weight)
            x = torch.nn.functional.elu(x)

            # 残差连接（降低系数从0.5到0.1，减少数值不稳定）
            x = x + 0.1 * x_residual

        # 恢复batch维度: [B*N, D] -> [B, N, D]
        task_embedding_with_deps = x.view(batch_size, num_tasks, embedding_dim)

        return task_embedding_with_deps

    def _adjacency_to_edge_index(self, adjacency, actual_num_nodes):
        """
        将邻接矩阵转换为PyG的edge_index格式

        Args:
            adjacency: [B, N_adj, N_adj] 邻接矩阵（可能小于实际节点数）
            actual_num_nodes: int, 实际的节点数（来自task_embedding维度，包含padding）

        Returns:
            edge_index: [2, E] 边索引
            edge_weight: [E] 边权重（全为1.0）
        """
        batch_size, adj_size, _ = adjacency.shape

        # 收集所有批次的边
        edge_indices = []
        edge_weights = []

        for b in range(batch_size):
            # 找到非零元素（有边的位置）
            sources, targets = torch.where(adjacency[b] > 0.5)

            # 【修复】：使用实际节点数计算offset，而不是邻接矩阵大小
            # 邻接矩阵大小 = 100 (只有任务)
            # 实际节点数 = 106 (任务 + 仓库)
            offset = b * actual_num_nodes
            sources = sources + offset
            targets = targets + offset

            edge_indices.append(torch.stack([sources, targets], dim=0))
            edge_weights.append(torch.ones(sources.shape[0], device=adjacency.device))

        # 合并所有批次
        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_weight = torch.cat(edge_weights, dim=0)
        else:
            # 没有边的情况
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=adjacency.device)
            edge_weight = torch.zeros(0, device=adjacency.device)

        return edge_index, edge_weight

    def _dynamic_gat_processing(self, task_encoding, edge_index, edge_attr, device):
        """
        【方案2B：Dynamic GAT Implementation + Solution B for Variable Edge Counts】

        完全动态的GAT处理，支持两种输入模式：
        1. 单一tensor模式（推理时）: edge_index=[2,E], edge_attr=[E,1]
        2. List模式（训练批处理时）: edge_index=List[[2,E_i]], edge_attr=List[[E_i,1]]

        设计特点：
        - 零边情况优雅处理
        - 任意边数支持(1-5个Sequential约束)
        - 向量化操作，高效批处理
        - 与现有topology potential完全兼容
        - 解决driver.py中torch.stack的变长tensor问题

        Args:
            task_encoding: [B, N_tasks, embedding_dim] 任务特征
            edge_index: [2, E] 边索引 OR List of [2, E_i] 边索引（numpy或tensor）
            edge_attr: [E, 1] 边权重 OR List of [E_i, 1] 边权重（阻塞度）
            device: 目标设备

        Returns:
            dependency_attended: [B, 1, embedding_dim] 依赖信息摘要
        """
        # 【方案B】检测输入类型：List模式 vs 单一tensor模式
        if isinstance(edge_index, list) and isinstance(edge_attr, list):
            # List模式：来自driver.py的rollout buffer批处理
            return self._dynamic_gat_processing_list_mode(
                task_encoding, edge_index, edge_attr, device
            )
        else:
            # 单一tensor模式：来自单步推理
            return self._dynamic_gat_processing_tensor_mode(
                task_encoding, edge_index, edge_attr, device
            )

    def _dynamic_gat_processing_tensor_mode(
        self, task_encoding, edge_index, edge_attr, device
    ):
        """
        单一tensor模式的GAT处理（原始实现）
        """
        # 转换输入到tensor格式
        if isinstance(edge_index, np.ndarray):
            edge_index = torch.from_numpy(edge_index).long()
        if isinstance(edge_attr, np.ndarray):
            edge_attr = torch.from_numpy(edge_attr).float()

        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)

        # 零边情况：直接返回零向量（类似topology potential的has_seq_edge检查）
        if edge_index.size(1) == 0:
            B = task_encoding.size(0)
            embedding_dim = task_encoding.size(2)
            return torch.zeros(B, 1, embedding_dim, device=device)

        B, N_tasks, embedding_dim = task_encoding.shape
        E = edge_index.size(1)

        # 边界检查：确保索引有效（防御性编程）
        source_indices = edge_index[0]  # [E]
        target_indices = edge_index[1]  # [E]

        if (
            source_indices.numel() == 0
            or source_indices.max() >= N_tasks
            or target_indices.max() >= N_tasks
        ):
            # 索引超出范围，返回零向量
            return torch.zeros(B, 1, embedding_dim, device=device)

        # 嵌入边权重：[E, 1] -> [E, embedding_dim]
        edge_embedded = self.edge_attr_embedding(edge_attr)  # [E, embedding_dim]
        edge_features = self.gat_edge(edge_embedded)  # [E, embedding_dim]

        # 批量化节点特征：[B, N_tasks, embedding_dim] -> [B*N_tasks, embedding_dim]
        task_encoding_flat = task_encoding.view(B * N_tasks, embedding_dim)

        # 为每个batch创建对应的边索引偏移
        # 原始边索引：[2, E] -> 批量边索引：[2, B*E]
        batch_edge_index = []
        batch_edge_features = []

        for batch_idx in range(B):
            # 为当前batch的边索引添加偏移
            offset = batch_idx * N_tasks
            batch_edges = edge_index + offset  # [2, E]
            batch_edge_index.append(batch_edges)
            batch_edge_features.append(edge_features)  # 复用相同的边特征

        # 合并所有batch的边：[2, B*E], [B*E, embedding_dim]
        full_edge_index = torch.cat(batch_edge_index, dim=1)  # [2, B*E]
        full_edge_features = torch.cat(
            batch_edge_features, dim=0
        )  # [B*E, embedding_dim]

        # 提取source和target特征
        source_features = task_encoding_flat[full_edge_index[0]]  # [B*E, embedding_dim]
        target_features = task_encoding_flat[full_edge_index[1]]  # [B*E, embedding_dim]

        # GAT attention计算
        keys = self.gat_key(source_features)  # [B*E, embedding_dim]
        queries = self.gat_query(target_features)  # [B*E, embedding_dim]
        values = self.gat_value(source_features)  # [B*E, embedding_dim]

        # Attention分数：query · (key + edge_feature)
        attention_logits = torch.sum(
            queries * (keys + full_edge_features), dim=-1
        )  # [B*E]

        # Softmax attention（使用scatter_softmax进行高效聚合）
        attention_weights = self._scatter_softmax(
            attention_logits, full_edge_index[1], B * N_tasks
        )

        # 计算消息并聚合
        messages = attention_weights.unsqueeze(-1) * values  # [B*E, embedding_dim]

        # Scatter聚合：将消息聚合到target节点
        aggregated_messages = torch.zeros(B * N_tasks, embedding_dim, device=device)
        aggregated_messages.index_add_(0, full_edge_index[1], messages)

        # 残差连接并reshape回原始形状
        task_encoding_updated = task_encoding_flat + aggregated_messages
        task_encoding_updated = task_encoding_updated.view(B, N_tasks, embedding_dim)

        # 全局依赖信息摘要：使用mean pooling
        dependency_attended = task_encoding_updated.mean(
            dim=1, keepdim=True
        )  # [B, 1, embedding_dim]

        return dependency_attended

    def _dynamic_gat_processing_list_mode(
        self, task_encoding, edge_index_list, edge_attr_list, device
    ):
        """
        【方案B核心实现】List模式的GAT处理，解决torch.stack变长tensor问题

        处理来自driver.py rollout buffer的List输入：
        - edge_index_list: [edge_index_0, edge_index_1, ...] 每个形状[2, E_i]
        - edge_attr_list: [edge_attr_0, edge_attr_1, ...] 每个形状[E_i, 1]

        每个batch元素可能有不同的边数，避免torch.stack错误
        """
        B, N_tasks, embedding_dim = task_encoding.shape

        # 验证输入list长度
        if len(edge_index_list) != B or len(edge_attr_list) != B:
            raise ValueError(
                f"List长度不匹配: edge_index_list={len(edge_index_list)}, edge_attr_list={len(edge_attr_list)}, batch_size={B}"
            )

        batch_results = []

        for batch_idx in range(B):
            # 提取当前batch的数据
            edge_index = edge_index_list[batch_idx]  # [2, E_i]
            edge_attr = edge_attr_list[batch_idx]  # [E_i, 1]
            task_encoding_single = task_encoding[
                batch_idx : batch_idx + 1
            ]  # [1, N_tasks, embedding_dim]

            # 单batch GAT处理（复用tensor模式逻辑）
            result = self._dynamic_gat_processing_tensor_mode(
                task_encoding_single, edge_index, edge_attr, device
            )  # [1, 1, embedding_dim]

            batch_results.append(result)

        # 合并所有batch结果
        dependency_attended = torch.cat(batch_results, dim=0)  # [B, 1, embedding_dim]

        return dependency_attended

    def _scatter_softmax(self, src, index, num_nodes):
        """
        高效的scatter softmax实现

        Args:
            src: [E] attention logits
            index: [E] target node indices
            num_nodes: int 节点总数

        Returns:
            out: [E] softmax attention weights
        """
        # 为数值稳定性减去最大值
        src_max = torch.full(
            (num_nodes,), float("-inf"), device=src.device, dtype=src.dtype
        )
        src_max.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
        src_max = src_max[index]
        src = src - src_max

        # 计算exp
        src_exp = torch.exp(src)

        # 计算分母（每个target节点的exp和）
        src_exp_sum = torch.zeros(num_nodes, device=src.device, dtype=src.dtype)
        src_exp_sum.index_add_(0, index, src_exp)
        src_exp_sum = src_exp_sum[index]

        # 避免除零
        src_exp_sum = torch.clamp(src_exp_sum, min=1e-8)

        return src_exp / src_exp_sum

    def evaluate_actions(
        self,
        tasks,
        agents,
        global_mask,
        index,
        actions,
        cargo_mask=None,
        action_type_mask=None,
        quantity_mask=None,
        ltl_info=None,
        dependency_graph=None,
    ):

        policy_logits, reward_value, cost_quantiles = self.forward(
            tasks, agents, global_mask, index, ltl_info, dependency_graph
        )

        # 【形状诊断】检查value的形状
        # value应该是[B, 1]，但在返回时需要确保形状正确

        # 解析动作 (现在包含4个维度: type, destination, cargo, quantity)
        stored_action_types = actions[:, 0]
        stored_destinations = actions[:, 1]
        stored_cargos = actions[:, 2]
        stored_quantities = (
            actions[:, 3]
            if actions.size(1) > 3
            else torch.zeros_like(stored_action_types)
        )

        action_type_logits = policy_logits["action_type"]
        if action_type_mask is not None:
            # 确保掩码形状匹配 (B, num_actions)
            if action_type_mask.dim() == 1:
                action_type_mask = action_type_mask.unsqueeze(0)
            action_type_logits = action_type_logits.masked_fill(
                action_type_mask.bool(), -1e9
            )

        action_type_dist = torch.distributions.Categorical(logits=action_type_logits)
        # action_type_dist = torch.distributions.Categorical(logits=policy_logits['action_type'])
        log_prob_type = action_type_dist.log_prob(stored_action_types)

        # MOVE 子集：destination
        log_prob_dest = torch.zeros_like(log_prob_type)
        entropy_dest_full = torch.zeros_like(log_prob_type, dtype=torch.float)

        move_mask = stored_action_types == 0  # 假定 0 对应 MOVE
        if move_mask.any():
            dest_logits_sub = policy_logits["destination"][move_mask]  # [N_move, D]

            # 【修复】应用global_mask（与worker采样时保持一致）
            if global_mask is not None:
                # 提取对应MOVE动作的mask
                global_mask_sub = global_mask[move_mask]
                dest_logits_sub = dest_logits_sub.masked_fill(
                    global_mask_sub.bool(), -1e9
                )

            # 行有效性：避免全 -inf 行
            valid_rows = torch.any(dest_logits_sub > -1e8, dim=-1)
            if valid_rows.any():
                dist_dest = torch.distributions.Categorical(
                    logits=dest_logits_sub[valid_rows]
                )
                # log prob 只对有效行计算
                log_prob_dest_sub = dist_dest.log_prob(
                    stored_destinations[move_mask][valid_rows]
                )
                idx = move_mask.nonzero(as_tuple=False).squeeze(1)[valid_rows]
                log_prob_dest[idx] = log_prob_dest_sub
                # 熵
                entropy_dest = torch.zeros(
                    dest_logits_sub.size(0), device=dest_logits_sub.device
                )
                entropy_dest[valid_rows] = dist_dest.entropy()
                entropy_dest_full[move_mask] = entropy_dest

        # LOAD 子集：cargo（需要 cargo_mask 一致化）
        log_prob_cargo = torch.zeros_like(log_prob_type)
        entropy_cargo_full = torch.zeros_like(log_prob_type, dtype=torch.float)

        load_mask = stored_action_types == 1  # 假定 1 对应 LOAD
        if load_mask.any():
            cargo_logits_sub = policy_logits["cargo"][load_mask]  # [N_load, K]
            if cargo_mask is not None:
                # 兼容 [T,K] / [B,K]
                if cargo_mask.dim() == 2 and cargo_mask.size(0) == actions.size(0):
                    cargo_mask_sub = cargo_mask[load_mask]
                else:
                    cargo_mask_sub = cargo_mask
                cargo_logits_sub = cargo_logits_sub.masked_fill(
                    cargo_mask_sub.bool(), -1e9
                )

            # 有效行判定
            valid_rows_c = torch.any(cargo_logits_sub > -1e8, dim=-1)
            if valid_rows_c.any():
                dist_cargo = torch.distributions.Categorical(
                    logits=cargo_logits_sub[valid_rows_c]
                )
                # log prob
                log_prob_cargo_sub = dist_cargo.log_prob(
                    stored_cargos[load_mask][valid_rows_c]
                )
                idx = load_mask.nonzero(as_tuple=False).squeeze(1)[valid_rows_c]
                log_prob_cargo[idx] = log_prob_cargo_sub
                # 熵
                entropy_c = torch.zeros(
                    cargo_logits_sub.size(0), device=cargo_logits_sub.device
                )
                entropy_c[valid_rows_c] = dist_cargo.entropy()
                entropy_cargo_full[load_mask] = entropy_c

        log_probs = log_prob_type + log_prob_dest + log_prob_cargo

        # 【方案A：加权归一化熵】
        # 获取各头的动作空间大小
        num_action_types = policy_logits["action_type"].size(-1)
        num_destinations = policy_logits["destination"].size(-1)
        num_cargos = policy_logits["cargo"].size(-1)

        # 计算最大熵（避免除以0）
        max_ent_type = math.log(max(num_action_types, 2))
        max_ent_dest = math.log(max(num_destinations, 2))
        max_ent_cargo = math.log(max(num_cargos, 2))

        # 归一化熵：每个头的熵除以其最大熵，然后求和
        # 范围：[0, 3]（3个头，每个头贡献[0,1]）
        entropy_type = action_type_dist.entropy()
        entropy = (
            entropy_type / max_ent_type
            + entropy_dest_full / max_ent_dest
            + entropy_cargo_full / max_ent_cargo
        )

        # 【诊断】计算各头的平均熵（用于监控）
        # 存储为类属性，供driver.py访问
        self.last_entropy_diagnostics = {
            "entropy_type_mean": entropy_type.mean().item(),
            "entropy_dest_mean": entropy_dest_full.mean().item(),
            "entropy_cargo_mean": entropy_cargo_full.mean().item(),
            "max_ent_type": max_ent_type,
            "max_ent_dest": max_ent_dest,
            "max_ent_cargo": max_ent_cargo,
        }

        # 【形状和数值诊断】确保返回的tensor形状和数值正确
        # log_probs: [B], entropy: [B], reward_value: [B, 1], cost_quantiles: [B, N]
        try:
            assert log_probs.dim() == 1, (
                f"log_probs should be 1D, got {log_probs.shape}"
            )
            assert entropy.dim() == 1, f"entropy should be 1D, got {entropy.shape}"
            assert reward_value.dim() == 2 and reward_value.size(1) == 1, (
                f"reward_value should be [B, 1], got {reward_value.shape}"
            )
            assert cost_quantiles.dim() == 2, (
                f"cost_quantiles should be [B, N], got {cost_quantiles.shape}"
            )

            # 检查是否有NaN或Inf
            assert not torch.isnan(log_probs).any(), "log_probs contains NaN!"
            assert not torch.isinf(log_probs).any(), "log_probs contains Inf!"
            assert not torch.isnan(entropy).any(), "entropy contains NaN!"
            assert not torch.isinf(entropy).any(), "entropy contains Inf!"
            assert not torch.isnan(reward_value).any(), "reward_value contains NaN!"
            assert not torch.isinf(reward_value).any(), "reward_value contains Inf!"
            assert not torch.isnan(cost_quantiles).any(), "cost_quantiles contains NaN!"
            assert not torch.isinf(cost_quantiles).any(), "cost_quantiles contains Inf!"
        except AssertionError as e:
            print(f"\n❌ [ATTENTION.PY ERROR] {e}")
            print(
                f"  log_probs shape: {log_probs.shape}, min: {log_probs.min().item():.4f}, max: {log_probs.max().item():.4f}"
            )
            print(
                f"  entropy shape: {entropy.shape}, min: {entropy.min().item():.4f}, max: {entropy.max().item():.4f}"
            )
            print(
                f"  reward_value shape: {reward_value.shape}, min: {reward_value.min().item():.4f}, max: {reward_value.max().item():.4f}"
            )
            print(
                f"  cost_quantiles shape: {cost_quantiles.shape}, min: {cost_quantiles.min().item():.4f}, max: {cost_quantiles.max().item():.4f}"
            )
            raise

        return log_probs, entropy, reward_value, cost_quantiles

    def evaluate_actions_split(
        self,
        tasks,
        agents,
        global_mask,
        index,
        actions,
        cargo_mask=None,
        action_type_mask=None,
        quantity_mask=None,
        ltl_info=None,
        dependency_graph=None,
    ):
        """
        返回一个 dict，包含：
          - 'logp_type', 'logp_dest', 'logp_cargo', 'logp_quantity' : [T] 的逐步 log_prob（不适用的步为 0）
          - 'ent_type', 'ent_dest', 'ent_cargo', 'ent_quantity'    : [T] 的逐步 entropy（不适用的步为 0）
          - 'reward_value'                          : [T] 的逐步状态价值
          - 'cost_quantiles'                        : [T, N] 的成本分布
          - 'logp_all'                              : [T] 组合后的总 log_prob（保持兼容）
        """
        policy_logits, reward_value, cost_quantiles = self.forward(
            tasks, agents, global_mask, index, ltl_info, dependency_graph
        )

        stored_action_types = actions[:, 0]
        stored_destinations = actions[:, 1]
        stored_cargos = actions[:, 2]
        stored_quantities = (
            actions[:, 3]
            if actions.size(1) > 3
            else torch.zeros_like(stored_action_types)
        )

        # --- action_type ---
        action_type_logits = policy_logits["action_type"]
        if action_type_mask is not None:
            if action_type_mask.dim() == 1:
                action_type_mask = action_type_mask.unsqueeze(0)
            action_type_logits = action_type_logits.masked_fill(
                action_type_mask.bool(), -1e9
            )
        dist_type = torch.distributions.Categorical(logits=action_type_logits)
        logp_type = dist_type.log_prob(stored_action_types)
        ent_type = dist_type.entropy()

        # --- destination (only when MOVE=0) ---
        move_mask = stored_action_types == 0
        logp_dest = torch.zeros_like(logp_type)
        ent_dest = torch.zeros_like(ent_type, dtype=torch.float)

        if move_mask.any():
            dest_logits_sub = policy_logits["destination"][move_mask]

            # 【修复】应用global_mask（与worker采样时保持一致）
            if global_mask is not None:
                # 提取对应MOVE动作的mask
                global_mask_sub = global_mask[move_mask]
                dest_logits_sub = dest_logits_sub.masked_fill(
                    global_mask_sub.bool(), -1e9
                )

            valid_rows = torch.any(dest_logits_sub > -1e8, dim=-1)
            if valid_rows.any():
                dist_dest = torch.distributions.Categorical(
                    logits=dest_logits_sub[valid_rows]
                )
                lp = dist_dest.log_prob(stored_destinations[move_mask][valid_rows])
                ent = dist_dest.entropy()
                idx = move_mask.nonzero(as_tuple=False).squeeze(1)[valid_rows]
                logp_dest[idx] = lp
                # 对齐 entropy 到原 T 轴
                ent_full = torch.zeros(
                    dest_logits_sub.size(0), device=dest_logits_sub.device
                )
                ent_full[valid_rows] = ent
                ent_dest[move_mask] = ent_full

        # --- cargo (only when LOAD=1) ---
        load_mask = stored_action_types == 1
        logp_cargo = torch.zeros_like(logp_type)
        ent_cargo = torch.zeros_like(ent_type, dtype=torch.float)

        if load_mask.any():
            cargo_logits_sub = policy_logits["cargo"][load_mask]
            if cargo_mask is not None:
                cm = cargo_mask[load_mask]
                cargo_logits_sub = cargo_logits_sub.masked_fill(cm.bool(), -1e9)
            valid_rows = torch.any(cargo_logits_sub > -1e8, dim=-1)
            if valid_rows.any():
                dist_cargo = torch.distributions.Categorical(
                    logits=cargo_logits_sub[valid_rows]
                )
                lp = dist_cargo.log_prob(stored_cargos[load_mask][valid_rows])
                ent = dist_cargo.entropy()
                idx = load_mask.nonzero(as_tuple=False).squeeze(1)[valid_rows]
                logp_cargo[idx] = lp
                ent_full = torch.zeros(
                    cargo_logits_sub.size(0), device=cargo_logits_sub.device
                )
                ent_full[valid_rows] = ent
                ent_cargo[load_mask] = ent_full

        # 组合 logp（与旧 evaluate_actions 等价）
        logp_all = logp_type + logp_dest + logp_cargo

        return {
            "logp_all": logp_all,
            "logp_type": logp_type,
            "logp_dest": logp_dest,
            "logp_cargo": logp_cargo,
            "ent_type": ent_type,
            "ent_dest": ent_dest,
            "ent_cargo": ent_cargo,
            "reward_value": reward_value.squeeze(-1)
            if reward_value.dim() > 1
            else reward_value,
            "cost_quantiles": cost_quantiles,  # [B, NUM_QUANTILES]
        }

    def encoding_agents(self, agents_inputs, mask=None):
        agents_embedding = self.agent_embedding(agents_inputs)
        agents_encoding = self.agentEncoder(agents_embedding, mask)
        embedding_dim = agents_encoding.size(-1)
        mean_mask = mask[:, 0, :].unsqueeze(2).repeat(1, 1, embedding_dim)
        compressed_task = torch.where(mean_mask, torch.nan, agents_embedding)
        aggregated_agents = torch.nanmean(compressed_task, dim=1).unsqueeze(1)

        ## FIX: Replace any potential NaNs with 0.0 to ensure numerical stability.
        aggregated_agents = torch.nan_to_num(aggregated_agents, nan=0.0)

        return aggregated_agents, agents_encoding

    def forward(
        self, tasks, agents, global_mask, index, ltl_info=None, dependency_graph=None
    ):
        """
        【方案一+二：forward方法现在接受dependency_graph参数】

        Args:
            tasks: [B, N_tasks, D_task] 任务特征（已包含5维LTL状态特征）
            agents: [B, N_agents, D_agent] agent特征
            global_mask: destination mask
            index: 当前agent索引
            ltl_info: 旧的LTL编码（方案A/B，保留兼容性）
            dependency_graph: [B, N_tasks, N_tasks] or [N_tasks, N_tasks] 任务依赖邻接矩阵（新增）
        """
        task_mask = get_attn_pad_mask(tasks, tasks)
        agent_mask = get_attn_pad_mask(agents, agents)
        task_agent_mask = get_attn_pad_mask(tasks, agents)
        agent_task_mask = get_attn_pad_mask(agents, tasks)

        # ===== 【方案一关键调用】：传入dependency_graph =====
        aggregated_task, task_encoding = self.encoding_tasks(
            tasks, dependency_adjacency=dependency_graph, mask=task_mask
        )
        aggregated_agents, agents_encoding = self.encoding_agents(
            agents, mask=agent_mask
        )

        task_agent_feature = self.crossDecoder1(
            task_encoding, agents_encoding, None, task_agent_mask
        )
        agent_task_feature = self.crossDecoder2(
            agents_encoding, task_encoding, None, agent_task_mask
        )

        current_state1 = torch.gather(
            agent_task_feature, 1, index.repeat(1, 1, agent_task_feature.size(2))
        )
        current_state = self.fusion(
            torch.cat((current_state1, aggregated_task, aggregated_agents), dim=-1)
        )

        current_state_prime = self.globalDecoder(
            current_state, task_agent_feature, None, global_mask
        )

        if TrainParams.LTL_ENABLED and ltl_info is not None:
            if TrainParams.LTL_ENCODING_TYPE == "A":
                # 【方案A：ID-specific编码处理流程】
                # 输入格式：[C, 4] 或 [B, C, 4]
                # 其中C=max_clauses, 4=[type, param1_norm, param2_norm, state_norm]

                # 1) 规整 ltl_info 维度到 [B, C, 4]
                if ltl_info.dim() == 2:
                    # [C, 4] -> [1, C, 4]
                    ltl_info = ltl_info.unsqueeze(0)
                elif ltl_info.dim() == 1:
                    # 异常情况：[4] -> [1, 1, 4]（单个约束）
                    ltl_info = ltl_info.view(1, 1, -1)

                ltl_info = ltl_info.to(current_state.device)  # 确保设备一致

                # 2) 嵌入：[B, C, 4] -> [B, C, embedding_dim]
                ltl_embedded = self.ltl_embedding(ltl_info.float())

                # 3) 聚合：[B, C, embedding_dim] -> [B, 1, embedding_dim]
                ltl_query = torch.mean(ltl_embedded, dim=1, keepdim=True)

                # 4) 交叉注意力融合
                ltl_fused_context, _ = self.pointer(q=ltl_query, h=current_state_prime)

                # 5) 残差连接
                current_state_prime = current_state_prime + ltl_fused_context

            elif TrainParams.LTL_ENCODING_TYPE == "B":
                # 【方案B：Task feasibility编码处理流程 - 增强版】
                # 输入格式：[max_agents, max_tasks] 或 [B, max_agents, max_tasks]
                # 0=可行，1=不可行
                # 【关键修改】保留完整矩阵，让当前agent看到所有agent的LTL约束

                # 1) 规整维度到 [B, max_agents, max_tasks]
                if ltl_info.dim() == 2:
                    # [max_agents, max_tasks] -> [1, max_agents, max_tasks]
                    ltl_info = ltl_info.unsqueeze(0)

                ltl_info = ltl_info.to(current_state.device)
                B = ltl_info.size(0)
                max_agents_in_matrix = ltl_info.size(1)
                max_tasks = ltl_info.size(2)

                # 2) 嵌入所有(agent, task)组合的可行性
                # ltl_info: [B, max_agents, max_tasks]
                # 展开为 [B, max_agents, max_tasks, 1] 以便输入Linear层
                ltl_info_expanded = ltl_info.unsqueeze(
                    -1
                )  # [B, max_agents, max_tasks, 1]

                # 嵌入：[B, max_agents, max_tasks, 1] -> [B, max_agents, max_tasks, embedding_dim]
                all_feasibility_embedded = self.task_feasibility_embedding(
                    ltl_info_expanded.float()
                )

                # 3) Agent-level aggregation：为每个agent聚合其所有任务的约束
                # 对于每个agent，使用attention聚合其max_tasks个嵌入向量
                agent_ltl_representations = []
                for agent_idx in range(max_agents_in_matrix):
                    # 提取该agent的所有任务可行性嵌入
                    agent_task_embeddings = all_feasibility_embedded[
                        :, agent_idx, :, :
                    ]  # [B, max_tasks, embedding_dim]

                    # 使用mean作为query，对所有任务进行attention
                    agent_query = agent_task_embeddings.mean(
                        dim=1, keepdim=True
                    )  # [B, 1, embedding_dim]

                    # Attention聚合：学习关注哪些被屏蔽的任务
                    agent_ltl_summary = self.agent_ltl_aggregation(
                        q=agent_query,  # [B, 1, embedding_dim]
                        h=agent_task_embeddings,  # [B, max_tasks, embedding_dim]
                    )  # [B, 1, embedding_dim]

                    agent_ltl_representations.append(agent_ltl_summary)

                # 堆叠所有agent的LTL表示：[B, max_agents, embedding_dim]
                all_agents_ltl = torch.cat(
                    agent_ltl_representations, dim=1
                )  # [B, max_agents, embedding_dim]

                # 4) Cross-agent attention：让当前agent关注所有agent的LTL约束
                # Query: 当前agent的状态
                # Key/Value: 所有agent的LTL约束汇总
                current_agent_query = current_state_prime  # [B, 1, embedding_dim]

                global_ltl_attended = self.cross_agent_ltl_attention(
                    q=current_agent_query,  # [B, 1, embedding_dim] - 当前agent状态
                    h=all_agents_ltl,  # [B, max_agents, embedding_dim] - 所有agent的LTL约束
                )  # [B, 1, embedding_dim]

                # 5) 融合：将全局LTL信息融入agent状态
                combined = torch.cat(
                    [current_state_prime, global_ltl_attended], dim=-1
                )  # [B, 1, 2*embedding_dim]
                ltl_fused = self.ltl_fusion(combined)  # [B, 1, embedding_dim]

                # 6) 残差连接
                current_state_prime = current_state_prime + ltl_fused

            elif TrainParams.LTL_ENCODING_TYPE == "C":
                # 【方案C：Task feasibility + Dependency graph处理流程】
                # 输入格式：字典 {'feasibility': [max_agents, max_tasks],
                #                 'edge_index': [2, E], 'edge_attr': [E, 1]}

                # ===== Part 1: 处理feasibility矩阵（与模式B相同） =====
                feasibility_matrix = ltl_info["feasibility"]
                edge_index = ltl_info["edge_index"]
                edge_attr = ltl_info["edge_attr"]

                # 1) 规整feasibility维度到 [B, max_agents, max_tasks]
                if feasibility_matrix.dim() == 2:
                    feasibility_matrix = feasibility_matrix.unsqueeze(0)

                feasibility_matrix = feasibility_matrix.to(current_state.device)
                B = feasibility_matrix.size(0)
                max_agents_in_matrix = feasibility_matrix.size(1)
                max_tasks_in_matrix = feasibility_matrix.size(2)

                # 2) 嵌入feasibility
                feasibility_expanded = feasibility_matrix.unsqueeze(
                    -1
                )  # [B, max_agents, max_tasks, 1]
                all_feasibility_embedded = self.task_feasibility_embedding(
                    feasibility_expanded.float()
                )

                # 3) Agent-level aggregation
                agent_ltl_representations = []
                for agent_idx in range(max_agents_in_matrix):
                    agent_task_embeddings = all_feasibility_embedded[
                        :, agent_idx, :, :
                    ]  # [B, max_tasks, embedding_dim]
                    agent_query = agent_task_embeddings.mean(
                        dim=1, keepdim=True
                    )  # [B, 1, embedding_dim]
                    agent_ltl_summary = self.agent_ltl_aggregation(
                        q=agent_query, h=agent_task_embeddings
                    )  # [B, 1, embedding_dim]
                    agent_ltl_representations.append(agent_ltl_summary)

                all_agents_ltl = torch.cat(
                    agent_ltl_representations, dim=1
                )  # [B, max_agents, embedding_dim]

                # 4) Cross-agent attention
                current_agent_query = current_state_prime  # [B, 1, embedding_dim]
                global_ltl_attended = self.cross_agent_ltl_attention(
                    q=current_agent_query, h=all_agents_ltl
                )  # [B, 1, embedding_dim]

                # ===== Part 2: 处理dependency graph（方案2B：Dynamic GAT） =====
                # 使用完全动态的scatter-based GAT，灵感来自topology potential的成功经验
                # 支持任意数量边(1-5)，无需batch循环，完全向量化处理

                dependency_attended = self._dynamic_gat_processing(
                    task_encoding, edge_index, edge_attr, current_state.device
                )

                # ===== Part 3: 融合feasibility和dependency两部分信息 =====
                # 【修复】分别处理两种信息，避免直接相加导致的信息混淆
                combined = torch.cat(
                    [current_state_prime, global_ltl_attended, dependency_attended],
                    dim=-1,
                )
                ltl_fused = self.ltl_fusion(combined)  # [B, 1, embedding_dim]

                # 残差连接
                current_state_prime = current_state_prime + ltl_fused
            else:
                raise ValueError(
                    f"Unknown LTL_ENCODING_TYPE: {TrainParams.LTL_ENCODING_TYPE}"
                )

        _, destination_logps = self.pointer(
            current_state_prime, task_agent_feature, mask=global_mask
        )
        destination_logits = destination_logps.squeeze(1)

        squeezed_context = current_state_prime.squeeze(1)

        action_type_logits = self.action_head(squeezed_context)
        cargo_logits = self.cargo_head(squeezed_context)

        # 分离的Critic输出
        reward_value = self.reward_critic(squeezed_context)  # [B, 1]
        cost_quantiles = self.cost_critic(squeezed_context)  # [B, NUM_QUANTILES]

        policy_logits = {
            "action_type": action_type_logits,
            "cargo": cargo_logits,
            "destination": destination_logits,
        }

        return policy_logits, reward_value, cost_quantiles
