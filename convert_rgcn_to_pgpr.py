# # convert_rgcn_to_pgpr.py
# import torch
# import numpy as np
# import pickle
# import os
#
# # ====== 1. 配置路径 ======
# output_dir = '/kaggle/working'
# dataset_dir = './tmp/Amazon_Beauty' # 对应 PGPR 的 TMP_DIR
# rgcn_ent_path = os.path.join(output_dir, 'gnn_entity_embeds.pt')
# rgcn_rel_path = os.path.join(output_dir, 'gnn_rel_embeds.pt')
#
# # ====== 2. 加载 RGCN 向量 (转换为 Numpy) ======
# print("加载 RGCN 向量...")
# ent_embeds = torch.load(rgcn_ent_path).numpy()
# rel_embeds = torch.load(rgcn_rel_path).numpy()
#
# # (强烈建议) L2 归一化，防止 RL 训练时梯度爆炸或 Reward 不稳定
# ent_embeds = ent_embeds / np.linalg.norm(ent_embeds, axis=-1, keepdims=True)
# rel_embeds = rel_embeds / np.linalg.norm(rel_embeds, axis=-1, keepdims=True)
#
# embed_size = ent_embeds.shape[1]
# pgpr_embed_dict = {}
#
# # ====== 3. 切分实体向量 (⚠️这里需要你填入构建图时的真实数量) ======
# # 假设你在构建 pyg_graph 时，节点的堆叠顺序是：user -> product -> word -> brand -> category -> related_product
# # 你需要从 dataset 中获取真实的 vocab_size（可以通过读取 dataset.pkl 获知）
# # 这里以假设的数量为例，请务必替换为你真实的节点数！
# num_users = 22363     # 替换为你的真实 user 数量
# num_products = 12101  # 替换为你的真实 product 数量
# num_words = 22564      # 替换为你的真实 word 数量
# num_brands = 2077      # 替换为你的真实 brand 数量
# num_categories = 248  # 替换为你的真实 category 数量
# num_rproducts = 164721  # 替换为你的真实 related_product 数量
#
# start = 0
# pgpr_embed_dict['user'] = ent_embeds[start : start + num_users]
# start += num_users
#
# pgpr_embed_dict['product'] = ent_embeds[start : start + num_products]
# start += num_products
#
# pgpr_embed_dict['word'] = ent_embeds[start : start + num_words]
# start += num_words
#
# pgpr_embed_dict['brand'] = ent_embeds[start : start + num_brands]
# start += num_brands
#
# pgpr_embed_dict['category'] = ent_embeds[start : start + num_categories]
# start += num_categories
#
# pgpr_embed_dict['related_product'] = ent_embeds[start : start + num_rproducts]
#
# # ====== 4. 组装关系向量 ======
# # PGPR 代码在 kg_env.py 中读取关系的格式是元组： (关系向量矩阵, 偏置bias矩阵)
# # 例如： last_relation_embed, _ = self.embeds[last_relation]
# # RGCN 通常没有 TransE 那样的 bias，所以 bias 我们填充为 0。
#
# # ⚠️同样，这里的索引 (0, 1, 2...) 必须对应你在构建 pyg_graph 的 edge_type 映射！
# relation_mapping = {
#     'purchase': 0,
#     'mentions': 1,
#     'described_as': 2,
#     'produced_by': 3,
#     'belongs_to': 4,
#     'also_bought': 5,
#     'also_viewed': 6,
#     'bought_together': 7
# }
#
# for rel_name, rel_idx in relation_mapping.items():
#     # 注意：PGPR 预期关系向量的 shape 是一维数组 (embed_size, )
#     rel_vec = rel_embeds[rel_idx]
#     zero_bias = np.zeros(embed_size)
#     pgpr_embed_dict[rel_name] = (rel_vec, zero_bias)
#
# # ====== 5. 保存为 PGPR 可直接读取的 Pickle 格式 ======
# save_path = os.path.join(dataset_dir, 'rgcn_embed.pkl')
# with open(save_path, 'wb') as f:
#     pickle.dump(pgpr_embed_dict, f)
# print(f"转换成功！RGCN 字典已保存至 {save_path}")

import torch
import numpy as np
import pickle
import os

# 导入 PGPR 的常量，保持和你生成图时一致
from utils import USER, PRODUCT, WORD, RPRODUCT, BRAND, CATEGORY
from utils import PURCHASE, MENTION, DESCRIBED_AS, PRODUCED_BY, BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER


def convert_embeddings(dataset_name='beauty'):
    # ========== 1. 路径配置 ==========
    output_dir = './tmp/beauty'
    dataset_dir = f'./tmp/{dataset_name}'
    kg_path = './kg.pkl'  # 或者指向具体的 tmp 目录下的 kg.pkl

    rgcn_ent_path = os.path.join(output_dir, 'gnn_entity_embeds.pt')
    rgcn_rel_path = os.path.join(output_dir, 'gnn_rel_embeds.pt')

    # ========== 2. 加载数据 ==========
    print("加载原始 KG 获取实体数量...")
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)

    print("加载 RGCN 训练好的 Tensor...")
    ent_embeds = torch.load(rgcn_ent_path).numpy()
    rel_embeds = torch.load(rgcn_rel_path).numpy()

    # (强烈建议) L2 归一化，RL 智能体对 scale 很敏感
    ent_embeds = ent_embeds / np.linalg.norm(ent_embeds, axis=-1, keepdims=True)
    rel_embeds = rel_embeds / np.linalg.norm(rel_embeds, axis=-1, keepdims=True)
    embed_size = ent_embeds.shape[1]

    # ========== 3. 按照 extract_triplets.py 的顺序精准切分实体 ==========
    entity_types = [USER, PRODUCT, WORD, RPRODUCT, BRAND, CATEGORY]
    pgpr_embed_dict = {}
    current_offset = 0

    print("开始切分实体向量...")
    for etype in entity_types:
        if etype in kg.G:
            num_nodes = len(kg.G[etype])
            # 切片取回该实体的专属向量
            pgpr_embed_dict[etype] = ent_embeds[current_offset: current_offset + num_nodes]
            print(f"实体 [{etype}]: 提取 {num_nodes} 个向量, 形状: {pgpr_embed_dict[etype].shape}")
            current_offset += num_nodes

    # ========== 4. 按照 extract_triplets.py 的顺序精准映射关系 ==========
    relation_types = [
        PURCHASE, MENTION, DESCRIBED_AS, PRODUCED_BY,
        BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER
    ]

    print("开始组装关系向量...")
    for idx, rel_name in enumerate(relation_types):
        # PGPR 的 relation 是 (向量, bias) 的元组，我们把 bias 设为 0
        rel_vec = rel_embeds[idx]
        zero_bias = np.zeros(embed_size)
        pgpr_embed_dict[rel_name] = (rel_vec, zero_bias)
        print(f"关系 [{rel_name}]: 映射到索引 {idx}")

    # ========== 5. 保存结果 ==========
    save_path = os.path.join(dataset_dir, 'rgcn_embed.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(pgpr_embed_dict, f)

    print(f"\n✅ 转换圆满成功！无缝兼容 PGPR 的字典已保存至: {save_path}")


if __name__ == '__main__':
    convert_embeddings('beauty')