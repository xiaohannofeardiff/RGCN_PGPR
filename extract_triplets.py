import pickle
import os
import pandas as pd
import torch

# 导入 PGPR 原有的工具类，以获取常量和关系映射
from utils import *;


def extract_gnn_triplets(dataset_name=BEAUTY):
    kg_path = './kg.pkl'
    if not os.path.exists(kg_path):
        raise FileNotFoundError(f"找不到 {kg_path}，请先运行 preprocess.py")

    print(f"正在加载 {dataset_name} 的知识图谱...")
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)

    # ---------------------------------------------------------
    # 步骤 1: 构建 Global ID 偏移字典 (极其关键)
    # ---------------------------------------------------------
    entity_offsets = {}
    current_offset = 0

    # 强制固定实体的遍历顺序，确保每次运行映射一致
    entity_types = [USER, PRODUCT, WORD, RPRODUCT, BRAND, CATEGORY]

    for etype in entity_types:
        if etype in kg.G:
            num_nodes = len(kg.G[etype])
            entity_offsets[etype] = current_offset
            print(f"实体 [{etype}]: 数量 {num_nodes}, 全局起始 ID: {current_offset}")
            current_offset += num_nodes

    print(f"全图总节点数 (GNN 的 num_nodes): {current_offset}")

    # ---------------------------------------------------------
    # 步骤 2: 构建关系 ID 映射
    # ---------------------------------------------------------
    relation_types = [
        PURCHASE, MENTION, DESCRIBED_AS, PRODUCED_BY,
        BELONG_TO, ALSO_BOUGHT, ALSO_VIEWED, BOUGHT_TOGETHER
    ]
    rel_mapping = {rel: idx for idx, rel in enumerate(relation_types)}
    print(f"关系映射字典: {rel_mapping}")

    # ---------------------------------------------------------
    # 步骤 3: 遍历 kg.G，展开为全局三元组
    # ---------------------------------------------------------
    triplets = []

    for head_type in kg.G:
        for head_id in kg.G[head_type]:
            global_head_id = entity_offsets[head_type] + head_id

            for relation in kg.G[head_type][head_id]:
                rel_id = rel_mapping[relation]

                # 从 utils.py 的 KG_RELATION 查出尾实体类型
                tail_type = KG_RELATION[head_type][relation]

                for tail_id in kg.G[head_type][head_id][relation]:
                    global_tail_id = entity_offsets[tail_type] + tail_id
                    triplets.append([global_head_id, rel_id, global_tail_id])

    # ---------------------------------------------------------
    # 步骤 4: 保存为 CSV 和 PyG 直接可用的 Tensor
    # ---------------------------------------------------------
    df = pd.DataFrame(triplets, columns=['head', 'relation', 'tail'])

    output_csv = f'./tmp/{dataset_name}/gnn_triplets.csv'
    # 确保目录存在
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df.to_csv(output_csv, index=False)
    print(f"成功导出 {len(df)} 条三元组至: {output_csv}")

    # 为了方便你后续直接写 PyG 模型，这里顺手帮你存成 edge_index 和 edge_type
    edge_index = torch.tensor([df['head'].tolist(), df['tail'].tolist()], dtype=torch.long)
    edge_type = torch.tensor(df['relation'].tolist(), dtype=torch.long)

    torch.save({'edge_index': edge_index, 'edge_type': edge_type}, f'./tmp/{dataset_name}/pyg_graph.pt')
    print("已生成 PyG 可用的 pyg_graph.pt！")


if __name__ == '__main__':
    extract_gnn_triplets(BEAUTY)