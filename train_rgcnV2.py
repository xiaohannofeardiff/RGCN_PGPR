import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import negative_sampling

# 定义设备，严禁在 CPU 上跑图神经网络
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print("警告: 未检测到 GPU！在 CPU 上跑 GNN 极度缓慢，请确认你的 CUDA 环境。")


class RGCN_LinkPrediction(nn.Module):
    def __init__(self, num_nodes, num_relations, hidden_channels):
        super(RGCN_LinkPrediction, self).__init__()
        # 初始的节点特征，等价于 TransE 的随机初始化
        self.node_emb = nn.Embedding(num_nodes, hidden_channels)
        # 新增：关系 Embedding (用于 DistMult 解码)
        self.rel_emb = nn.Embedding(num_relations, hidden_channels)

        # 两层 RGCN 卷积
        # 注意: 这里的 num_relations 必须准确，否则底层权重矩阵分配会错乱
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)

    def encode(self, edge_index, edge_type):
        """编码器：聚合邻居信息，生成带有结构感知的最终向量"""
        x = self.node_emb.weight
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)

        # 加入 L2 归一化
        x = F.normalize(x, p=2, dim=-1)
        return x


    def decode(self, z, edge_index, edge_type):
        """
        定制版解码器：完美适配 PGPR 的 (h + r) * t 点积逻辑
        """
        h = z[edge_index[0]]
        t = z[edge_index[1]]

        # 获取关系向量。PGPR 原版没有对向量进行强行 L2 归一化，
        # 而是依赖较小的初始化和 L2 正则。
        # 为了保证 GNN 聚合后数值不爆炸，这里建议做缩放或归一化。
        r = self.rel_emb(edge_type)

        # 【核心魔法】：(h + r) 和 t 进行逐元素相乘后求和，等价于点积 (Dot Product)
        # 这与原版 torch.bmm(pos_vec, example_vec) 在数学上完全等价！
        score = ((h + r) * t).sum(dim=-1)

        return score


def train():
    dataset_name = 'beauty'  # 根据你的实际情况修改
    base_dir = f'./tmp/{dataset_name}'
    graph_path = os.path.join(base_dir, 'pyg_graph.pt')
    ckpt_dir = os.path.join(base_dir, 'checkpoints')  # 新增：检查点专属目录

    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"找不到 {graph_path}，请确认上一步提取脚本是否成功。")
    os.makedirs(ckpt_dir, exist_ok=True)
    print("加载图数据...")
    graph_data = torch.load(graph_path)
    edge_index = graph_data['edge_index'].to(device)
    edge_type = graph_data['edge_type'].to(device)

    # 动态推断节点数和关系数
    num_nodes = int(edge_index.max()) + 1
    num_relations = int(edge_type.max()) + 1

    print(f"检测到全图节点数: {num_nodes}")
    print(f"检测到图谱关系数: {num_relations}")

    # 超参数设置 (这里必须与你后续 RL Agent 的 hidden_size 严格对齐)
    # PGPR 默认 embedding size 似乎是 100 或 64，请查阅你的 config
    #hidden_channels = 64
    hidden_channels = 100
    epochs = 100
    learning_rate = 0.01
    save_interval = 10

    model = RGCN_LinkPrediction(num_nodes, num_relations, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\n开始预训练 RGCN...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 1. 编码：获取全图节点的结构化表征
        z = model.encode(edge_index, edge_type)

        # 2. 正样本：计算真实存在的边的得分
        pos_out = model.decode(z, edge_index, edge_type)
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))

        # 3. 负样本生成 (Entity Corruption: 实体破坏机制)
        # 生成与 edge_index 形状完全一样的负样本索引
        head, tail = edge_index[0], edge_index[1]

        # 抛硬币：50% 概率替换头实体，50% 概率替换尾实体
        mask = torch.rand(edge_index.size(1), device=device) < 0.5

        # 生成全局随机节点作为干扰项
        random_nodes = torch.randint(0, num_nodes, (edge_index.size(1),), device=device)

        # 根据 mask 执行替换
        neg_head = torch.where(mask, random_nodes, head)
        neg_tail = torch.where(~mask, random_nodes, tail)

        # 拼接成最终的负样本边 (形状 [2, num_edges]，与正样本完全对齐)
        neg_edge_index = torch.stack([neg_head, neg_tail], dim=0)
        # 3. 负样本：随机生成不存在的边并计算得分
        # 注意: 这一步是防止模型退化的绝对核心
        # neg_edge_index = negative_sampling(
        #     edge_index=edge_index,
        #     num_nodes=num_nodes,
        #     num_neg_samples=edge_index.size(1),
        #     method='sparse'
        # )
        neg_out = model.decode(z, neg_edge_index, edge_type)
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))

        # 4. 汇总损失并反向传播
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        # 5. 打印与规范化保存 (Checkpointing)
        if epoch % save_interval == 0 or epoch == epochs:
            print(f"Epoch: {epoch:03d}/{epochs}, Loss: {loss.item():.4f}")

            # 保存模型权重 (用于后续恢复训练或检查)
            ckpt_path = os.path.join(ckpt_dir, f'rgcn_model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)
        #打印每轮的训练数据
        print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}")

    # ---------------------------------------------------------
    # 训练完毕，导出“弹药”
    # ---------------------------------------------------------
    print("\n训练完成，正在导出结构化 Embedding...")
    model.eval()
    with torch.no_grad():
        # 清空无用显存，防止 OOM
        torch.cuda.empty_cache()
        final_entity_embeds = model.encode(edge_index, edge_type)
        final_rel_embeds = model.rel_emb.weight.detach()

    # 保存到文件，供 PGPR 的 RL 环境直接查表使用
    out_path = f'./tmp/{dataset_name}/gnn_entity_embeds.pt'
    torch.save(final_entity_embeds.cpu(), f'./tmp/{dataset_name}/gnn_entity_embeds.pt')
    torch.save(final_rel_embeds.cpu(), f'./tmp/{dataset_name}/gnn_rel_embeds.pt')
    print("实体与关系表征均已保存！")


if __name__ == '__main__':
    train()