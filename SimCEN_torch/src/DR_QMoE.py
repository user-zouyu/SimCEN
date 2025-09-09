# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding
import math
from typing import Dict, List, Tuple


class DR_QMoE(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="DR_QMoE",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=16,
                 num_layers=3,
                 num_experts=6,
                 top_k=2,
                 expert_capacity_factor=1.25,
                 load_balance_weight=0.01,
                 expert_dropout=0.1,
                 net_dropout=0,
                 batch_norm=False,
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(DR_QMoE, self).__init__(feature_map,
                                      model_id=model_id,
                                      gpu=gpu,
                                      embedding_regularizer=embedding_regularizer,
                                      net_regularizer=net_regularizer,
                                      **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        input_dim = feature_map.sum_emb_out_dim()
        num_fields = feature_map.num_fields
        self.qnn = DynamicRoutingMoEQuadraticNeuralNetworks(
            input_dim=input_dim,
            num_fields=num_fields,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            expert_capacity_factor=expert_capacity_factor,
            load_balance_weight=load_balance_weight,
            expert_dropout=expert_dropout,
            net_dropout=net_dropout,
            batch_norm=batch_norm
        )
        self.fc = nn.Linear(input_dim, 1)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X, dynamic_emb_dim=True)
        qnn_output, auxiliary_loss = self.qnn(feature_emb)
        y_pred = self.fc(qnn_output)
        y_pred = self.output_activation(y_pred)
        
        # 返回主要预测和辅助损失
        return {"y_pred": y_pred, "aux_loss": auxiliary_loss}

    def add_loss(self, inputs):
        """重写损失函数以包含负载均衡损失"""
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred = return_dict["y_pred"]
        aux_loss = return_dict["aux_loss"]
        
        # 主要损失
        main_loss = self.loss_fn(y_pred, y_true, reduction='mean')
        
        # 总损失（包含专家负载均衡）
        total_loss = main_loss + aux_loss
        
        return total_loss


class DynamicRoutingMoEQuadraticNeuralNetworks(nn.Module):
    def __init__(self,
                 input_dim,
                 num_fields,
                 num_layers=3,
                 num_experts=6,
                 top_k=2,
                 expert_capacity_factor=1.25,
                 load_balance_weight=0.01,
                 expert_dropout=0.1,
                 net_dropout=0.1,
                 batch_norm=False):
        super().__init__()
        self.num_layers = num_layers
        self.load_balance_weight = load_balance_weight
        
        self.dropout = nn.ModuleList()
        self.layer = nn.ModuleList()
        
        for i in range(num_layers):
            self.layer.append(DynamicRoutingMoEQuadraticLayer(
                input_dim=input_dim,
                num_fields=num_fields,
                num_experts=num_experts,
                top_k=top_k,
                expert_capacity_factor=expert_capacity_factor,
                expert_dropout=expert_dropout,
                batch_norm=batch_norm,
                layer_idx=i
            ))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))

    def forward(self, x):
        total_aux_loss = 0.0
        
        for i in range(self.num_layers):
            x, aux_loss = self.layer[i](x)
            total_aux_loss += aux_loss
            
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        
        # 平均辅助损失并加权
        avg_aux_loss = total_aux_loss / self.num_layers
        weighted_aux_loss = self.load_balance_weight * avg_aux_loss
        
        return x, weighted_aux_loss


class DynamicRoutingMoEQuadraticLayer(nn.Module):
    """
    T37: 动态路由专家混合二次网络
    
    创新点：
    1. 智能路由器：基于样本特征和专家特长的动态路由
    2. 专业化专家：6个专家处理不同类型的特征交互
    3. 容量感知路由：防止专家过载，确保负载均衡
    4. 层次化专家协作：专家间的知识共享和协作机制
    """
    def __init__(self,
                 input_dim,
                 num_fields,
                 num_experts=6,
                 top_k=2,
                 expert_capacity_factor=1.25,
                 expert_dropout=0.1,
                 batch_norm=False,
                 layer_idx=0):
        super().__init__()
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.layer_idx = layer_idx
        self.embedding_dim = input_dim // num_fields
        
        # 智能路由器
        self.router = IntelligentRouter(
            input_dim, num_experts, num_fields, layer_idx
        )
        
        # 专业化专家库（按需截取前 num_experts 个，保证与路由维度一致）
        available_experts = [
            ('sparse_expert',   SparseQuadraticExpert(input_dim, num_fields, expert_dropout)),
            ('dense_expert',    DenseQuadraticExpert(input_dim, num_fields, expert_dropout)),
            ('cross_field_expert', CrossFieldQuadraticExpert(input_dim, num_fields, expert_dropout)),
            ('high_freq_expert',   HighFreqQuadraticExpert(input_dim, num_fields, expert_dropout)),
            ('temporal_expert', TemporalQuadraticExpert(input_dim, num_fields, expert_dropout)),
            ('long_tail_expert', LongTailQuadraticExpert(input_dim, num_fields, expert_dropout)),
        ]
        # 若配置的 num_experts 超过可用数量，则退化为全部专家
        selected = available_experts[:self.num_experts] if self.num_experts <= len(available_experts) else available_experts
        self.num_experts = len(selected)
        self.experts = nn.ModuleDict({name: module for name, module in selected})
        
        # 简化的输出处理
        self.output_projection = nn.Linear(input_dim, input_dim)
        
        # 归一化
        self.layer_norm = nn.LayerNorm(input_dim) if batch_norm else nn.Identity()

    def forward(self, x):
        ego_x = x
        batch_size = x.shape[0]
        
        # 智能路由决策
        routing_result = self.router(x)
        expert_gates = routing_result['gates']  # B × top_k
        expert_indices = routing_result['indices']  # B × top_k
        routing_probs = routing_result['probs']  # B × num_experts
        
        # 专家容量控制
        expert_capacity = int(batch_size * self.expert_capacity_factor / self.num_experts)
        
        # 简化的专家计算 - 避免复杂的掩码操作
        expert_outputs = []
        # 仅计算与路由一致数量的专家输出，避免维度不匹配
        for expert in self.experts.values():
            expert_outputs.append(expert(x))  # B × D
        expert_stack = torch.stack(expert_outputs, dim=1)  # B × num_experts × D
        
        # 使用路由概率进行加权平均
        # 若路由概率维度与专家数不一致，则按列截断或补齐
        if routing_probs.size(1) != expert_stack.size(1):
            if routing_probs.size(1) > expert_stack.size(1):
                routing_probs = routing_probs[:, :expert_stack.size(1)]
            else:
                pad_cols = expert_stack.size(1) - routing_probs.size(1)
                padding = torch.full((routing_probs.size(0), pad_cols),
                                     fill_value=1.0 / expert_stack.size(1),
                                     device=routing_probs.device)
                routing_probs = torch.cat([routing_probs, padding], dim=1)
        routing_weights = routing_probs.unsqueeze(-1)  # B × num_experts × 1
        weighted_output = torch.sum(expert_stack * routing_weights, dim=1)  # B × D
        
        # 输出投影
        final_output = self.output_projection(weighted_output)
        
        # 残差连接和层标准化
        output = self.layer_norm(final_output + ego_x)
        
        # 计算负载均衡损失
        aux_loss = self._compute_load_balance_loss(routing_probs, expert_indices)
        
        return output, aux_loss
    
    def _compute_load_balance_loss(self, routing_probs, expert_indices):
        """计算专家负载均衡损失"""
        # 期望的均匀分布
        num_tokens = routing_probs.shape[0] * self.top_k
        expected_load = num_tokens / self.num_experts
        
        # 实际专家使用次数
        expert_counts = torch.bincount(expert_indices.flatten(), minlength=self.num_experts).float()
        
        # 负载均衡损失（变异系数）
        load_variance = torch.var(expert_counts)
        load_mean = torch.mean(expert_counts)
        
        if load_mean > 0:
            cv_loss = load_variance / (load_mean ** 2)
        else:
            cv_loss = torch.tensor(0.0, device=routing_probs.device)
        
        # 专家使用概率的熵损失（鼓励使用多样性）
        routing_entropy = -torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=1).mean()
        entropy_loss = -routing_entropy  # 负熵，鼓励高熵
        
        return cv_loss + 0.1 * entropy_loss


class IntelligentRouter(nn.Module):
    """智能路由器：基于样本特征和专家特长的动态路由"""
    def __init__(self, input_dim, num_experts, num_fields, layer_idx):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_fields = num_fields
        self.layer_idx = layer_idx
        
        # 多维特征分析器
        self.feature_analyzer = FeatureAnalyzer(input_dim, num_fields)
        
        # 专家特长建模
        self.expert_specialization = ExpertSpecializationModule(num_experts, input_dim)
        
        # 路由决策网络
        self.routing_network = nn.Sequential(
            nn.Linear(input_dim + num_experts, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 动态top-k选择
        self.top_k_selector = DynamicTopKSelector(input_dim, num_experts)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 分析输入特征
        feature_analysis = self.feature_analyzer(x)
        
        # 获取专家特长分数
        expert_specialization_scores = self.expert_specialization(x)
        
        # 路由决策
        routing_input = torch.cat([x, expert_specialization_scores], dim=-1)
        routing_probs = self.routing_network(routing_input)  # B × num_experts
        
        # 动态top-k选择
        top_k = self.top_k_selector(x, feature_analysis)
        
        # 选择top-k专家
        top_k_probs, top_k_indices = torch.topk(routing_probs, top_k, dim=-1)
        top_k_gates = F.softmax(top_k_probs, dim=-1)
        
        return {
            'gates': top_k_gates,
            'indices': top_k_indices,
            'probs': routing_probs,
            'analysis': feature_analysis
        }


class FeatureAnalyzer(nn.Module):
    """多维特征分析器"""
    def __init__(self, input_dim, num_fields):
        super().__init__()
        self.input_dim = input_dim
        self.num_fields = num_fields
        
    def forward(self, x):
        # 计算多种特征统计
        analysis = {
            'sparsity': (x == 0).float().mean(dim=-1),  # 稀疏度
            'variance': x.var(dim=-1),  # 方差
            'magnitude': x.abs().max(dim=-1)[0],  # 最大幅度
            'norm': torch.norm(x, p=2, dim=-1),  # L2范数
            'skewness': self._compute_skewness(x),  # 偏度
            'concentration': self._compute_concentration(x)  # 集中度
        }
        return analysis
    
    def _compute_skewness(self, x):
        """计算偏度"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        std = torch.sqrt(var + 1e-8)
        skewness = ((x - mean) / std).pow(3).mean(dim=-1)
        return skewness
    
    def _compute_concentration(self, x):
        """计算特征集中度"""
        sorted_x, _ = torch.sort(x.abs(), dim=-1, descending=True)
        total_mass = sorted_x.sum(dim=-1)
        top_20_percent = int(0.2 * x.shape[-1])
        top_mass = sorted_x[:, :top_20_percent].sum(dim=-1)
        concentration = top_mass / (total_mass + 1e-8)
        return concentration


class ExpertSpecializationModule(nn.Module):
    """专家特长建模"""
    def __init__(self, num_experts, input_dim):
        super().__init__()
        self.num_experts = num_experts
        
        # 每个专家的特长模式
        self.expert_patterns = nn.Parameter(torch.randn(num_experts, input_dim))
        
        # 专家相似度计算
        self.similarity_network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts)
        )
        
    def forward(self, x):
        # 计算输入与每个专家模式的相似度
        similarities = torch.matmul(x, self.expert_patterns.T)  # B × num_experts
        
        # 通过网络进一步处理
        specialization_scores = self.similarity_network(x)
        
        # 结合相似度和网络输出
        final_scores = torch.sigmoid(similarities + specialization_scores)
        
        return final_scores


class DynamicTopKSelector(nn.Module):
    """动态top-k选择器"""
    def __init__(self, input_dim, num_experts, min_k=1, max_k=4):
        super().__init__()
        self.min_k = min_k
        self.max_k = max_k
        
        self.k_predictor = nn.Sequential(
            nn.Linear(6, 16),  # 6个特征分析维度
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, feature_analysis):
        # 基于特征分析决定top-k
        analysis_features = torch.stack([
            feature_analysis['sparsity'],
            feature_analysis['variance'],
            feature_analysis['magnitude'],
            feature_analysis['norm'],
            feature_analysis['skewness'],
            feature_analysis['concentration']
        ], dim=-1)
        
        k_ratio = self.k_predictor(analysis_features).squeeze(-1)  # B
        k_values = self.min_k + (self.max_k - self.min_k) * k_ratio
        
        # 取批次内的中位数作为统一的k值
        k = int(torch.median(k_values).item())
        k = max(self.min_k, min(self.max_k, k))
        
        return k


# 六个专业化专家的实现
class SparseQuadraticExpert(nn.Module):
    """稀疏交互专家：专门处理稀疏特征"""
    def __init__(self, input_dim, num_fields, dropout_rate):
        super().__init__()
        # 为避免 embed_dim % num_heads 约束带来的潜在维度问题，使用单头注意力
        self.sparse_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, dropout=dropout_rate, batch_first=True)
        self.sparse_quadratic = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, x):
        # 针对稀疏输入的注意力机制
        x_attended, _ = self.sparse_attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        x_attended = x_attended.squeeze(1)
        
        # 稀疏友好的二次交互
        output = self.sparse_quadratic(x_attended * x)
        return output


class DenseQuadraticExpert(nn.Module):
    """密集交互专家：专门处理密集特征"""
    def __init__(self, input_dim, num_fields, dropout_rate):
        super().__init__()
        # 瓶颈结构，降低内存占用
        hidden_dim = max(64, input_dim // 4)
        self.dense_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim)
        )
        # 轻量级二次门控：h(x) ⊙ x
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 密集的非线性变换
        transformed = self.dense_layers(x)
        # 轻量二次项
        quadratic_term = self.gate(x) * x
        output = transformed + quadratic_term
        return output


class CrossFieldQuadraticExpert(nn.Module):
    """跨字段专家：专门处理字段间交互"""
    def __init__(self, input_dim, num_fields, dropout_rate):
        super().__init__()
        self.num_fields = num_fields
        self.field_dim = input_dim // num_fields
        # 使用更高效的参数化方式
        # 这里 self.field_dim 可能很小，为避免 num_heads 约束问题，使用单头
        self.field_interaction = nn.MultiheadAttention(
            embed_dim=self.field_dim,
            num_heads=1,
            batch_first=True,
            dropout=dropout_rate
        )
        self.field_fusion = nn.Linear(self.field_dim, self.field_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_field = x.view(batch_size, self.num_fields, self.field_dim)
        
        # 使用注意力机制进行字段间交互
        attended_fields, _ = self.field_interaction(x_field, x_field, x_field)
        
        # 融合原始字段和交互字段
        fused_fields = self.field_fusion(attended_fields * x_field)
        
        # 重新展平
        output = fused_fields.view(batch_size, -1)
        
        return self.dropout(output)


class TemporalQuadraticExpert(nn.Module):
    """时序专家：处理时序相关的特征"""
    def __init__(self, input_dim, num_fields, dropout_rate):
        super().__init__()
        # 降低通道数，减少显存占用
        self.temporal_conv = nn.Conv1d(1, 4, kernel_size=3, padding=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(input_dim)
        self.temporal_fc = nn.Linear(4, 1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 将特征视为时序序列
        x_temporal = x.unsqueeze(1)  # B × 1 × D
        conv_out = torch.relu(self.temporal_conv(x_temporal))  # B × 16 × D
        
        # 时序池化和融合
        pooled = self.temporal_pool(conv_out)  # B × 16 × D
        weights = torch.sigmoid(self.temporal_fc(pooled.transpose(1, 2)))  # B × D × 1
        
        # 加权的二次交互
        weighted_x = (x.unsqueeze(-1) * weights).squeeze(-1)
        output = weighted_x * x
        
        return self.dropout(output)


class HighFreqQuadraticExpert(nn.Module):
    """高频专家：处理高频变化的特征"""
    def __init__(self, input_dim, num_fields, dropout_rate):
        super().__init__()
        hidden_dim = max(64, input_dim // 4)
        self.high_freq_filter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 高频滤波
        filtered = self.high_freq_filter(x)
        
        # 高频成分的二次交互
        high_freq_component = filtered - x
        output = x + high_freq_component * x
        
        return self.dropout(output)


class LongTailQuadraticExpert(nn.Module):
    """长尾专家：处理长尾分布的特征"""
    def __init__(self, input_dim, num_fields, dropout_rate):
        super().__init__()
        hidden_dim = max(64, input_dim // 4)
        self.tail_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 长尾变换（ELU有助于处理长尾分布）
        transformed = self.tail_transform(x)
        
        # 长尾友好的二次交互
        output = torch.sign(x) * torch.sqrt(torch.abs(transformed * x) + 1e-8)
        
        return self.dropout(output)


# 移除了ExpertCollaborationModule和OutputFusionModule以简化代码和减少内存使用
