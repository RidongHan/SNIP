import os
import time
import glob
import json
import argparse
import numpy as np
import networkx as nx
import os.path as osp
from scipy.stats import rankdata

import torch
import torch.nn.functional as F

from model import Model
from utils import MMD, evaluate, CitationDataset, TwitchDataset, CitationDatasetSource


def rank_normalization(diff_dict):
    # 获取差异值的列表
    diff_values = list(diff_dict.values())
    # 使用rankdata进行升序排名
    ranks = rankdata(diff_values, method='ordinal')
    ranks = len(ranks) - ranks + 1  # 反转排名
    # 将排名映射到 [0, 1] 范围
    normalized = (ranks - 1) / (len(ranks) - 1)
    # 将归一化后的值返回到一个新的字典中，key 为节点，value 为归一化后的值
    normalized_dict = dict(zip(diff_dict.keys(), normalized))
    return normalized_dict

def centrality_pruning(source_data, target_data, args):
    # 创建保存目录
    save_dir = "./nodeDiff"
    os.makedirs(save_dir, exist_ok=True)

    # 构建文件名
    filename = f"{args.source}_{args.target}_node_differences.json"
    filepath = os.path.join(save_dir, filename)

    # 检查文件是否存在
    if os.path.exists(filepath):
        print(f"Load Data From: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)

        degree_diff = {int(k): v for k, v in data['degree_diff'].items()}
        betweenness_diff = {int(k): v for k, v in data['betweenness_diff'].items()}
        closeness_diff = {int(k): v for k, v in data['closeness_diff'].items()}
        eigenvector_diff = {int(k): v for k, v in data['eigenvector_diff'].items()}

        # 创建源图用于获取节点列表
        source_G = nx.from_edgelist(source_data.edge_index.t().cpu().numpy())

    else:
        print(f"Compute Centrality Measures and Save It To: {filepath}")
        # 将PyG数据转换为NetworkX图
        source_G = nx.from_edgelist(source_data.edge_index.t().cpu().numpy())
        target_G = nx.from_edgelist(target_data.edge_index.t().cpu().numpy())

        # 计算源图的中心性指标
        degree_centrality = nx.degree_centrality(source_G)  # 衡量节点直接连接的邻居数量（度数）
        betweenness_centrality = nx.betweenness_centrality(source_G)  # 衡量节点作为"桥梁"或"中介"的重要性
        closeness_centrality = nx.closeness_centrality(source_G)  # 衡量节点到其他所有节点的平均距离的倒数
        eigenvector_centrality = nx.eigenvector_centrality(source_G, max_iter=1000)  # 衡量节点的重要性，不仅看邻居数量，还看邻居的质量

        # 计算目标图的平均中心性
        target_degree = np.mean(list(nx.degree_centrality(target_G).values()))
        target_betweenness = np.mean(list(nx.betweenness_centrality(target_G).values()))
        target_closeness = np.mean(list(nx.closeness_centrality(target_G).values()))
        target_eigenvector = np.mean(list(nx.eigenvector_centrality(target_G, max_iter=1000).values()))

        # 计算每个节点的差异
        degree_diff = {}
        betweenness_diff = {}
        closeness_diff = {}
        eigenvector_diff = {}

        for node in source_G.nodes():
            degree_diff[node] = abs(degree_centrality[node] - target_degree)
            betweenness_diff[node] = abs(betweenness_centrality[node] - target_betweenness)
            closeness_diff[node] = abs(closeness_centrality[node] - target_closeness)
            eigenvector_diff[node] = abs(eigenvector_centrality[node] - target_eigenvector)

        # 保存到JSON文件
        data_to_save = {
            'degree_diff': {str(k): v for k, v in degree_diff.items()},
            'betweenness_diff': {str(k): v for k, v in betweenness_diff.items()},
            'closeness_diff': {str(k): v for k, v in closeness_diff.items()},
            'eigenvector_diff': {str(k): v for k, v in eigenvector_diff.items()}
        }

        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Save Data To: {filepath}")

    # 对四个指标分别进行rank归一化
    degree_normalized = rank_normalization(degree_diff)
    betweenness_normalized = rank_normalization(betweenness_diff)
    closeness_normalized = rank_normalization(closeness_diff)
    eigenvector_normalized = rank_normalization(eigenvector_diff)

    # 计算每个节点的综合得分
    node_scores = {}
    for node in source_G.nodes():# 输出node是什么
        score = (args.weight_degree * degree_normalized[node] +
                 args.weight_betweenness * betweenness_normalized[node] +
                 args.weight_closeness * closeness_normalized[node] +
                 args.weight_eigenvector * eigenvector_normalized[node])
        node_scores[node] = score
        
    # 按得分降序排序，移除得分小的节点
    sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

    # 根据比例决定要移除多少节点
    num_to_remove = int(len(sorted_nodes) * args.remove_ratio)
    nodes_to_remove = [node for node, _ in sorted_nodes[-num_to_remove:]]

    print(f"Number of Nodes to be Removed: {num_to_remove}")
    return nodes_to_remove

def get_nodes_to_be_removed_structural_pruning(args):
    if args.remove_ratio == 0:
        return []
    else:
        path_source = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.source)
        source_dataset = CitationDataset(path_source, args.source)
        path_target = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.target)
        target_dataset = CitationDataset(path_target, args.target)
        source_data = source_dataset[0].to(args.device)
        target_data = target_dataset[0].to(args.device)

        # 调用函数寻找最优剪枝策略
        return centrality_pruning(source_data, target_data, args)
    
def train(args, source_data, target_data):
    min_loss = 1e10
    patience_cnt = 0
    loss_values = []
    best_epoch = 0

    # --- 初始化最大 F1 变量 ---
    max_macro_f1 = 0.0
    max_micro_f1 = 0.0

    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    t = time.time()
    for epoch in range(args.epochs):
        model.train()
        # 1. 前向传播与 Loss 计算 (代码保持不变)
        output = model(source_data.x, source_data.edge_index, args.source_pnum)
        train_loss = F.nll_loss(F.log_softmax(output, dim=1), source_data.y)

        source_feature = model.feat_bottleneck(source_data.x, source_data.edge_index, args.source_pnum)
        target_feature = model.feat_bottleneck(target_data.x, target_data.edge_index, args.target_pnum)
        mmd_loss = MMD(source_feature, target_feature)
        loss = train_loss + args.weight * mmd_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 2. 评估
        with torch.no_grad():
            acc, _, _, _ = evaluate(source_data, model)
            _, macro_f1, micro_f1, test_loss = evaluate(target_data, model, args.target_pnum)

            print('Epoch: {:04d}'.format(epoch + 1),
                  'train_loss: {:.6f}'.format(loss),
                  'test_loss: {:.6f}'.format(test_loss),
                  'train_acc: {:.6f}'.format(acc),
                  'macro_f1: {:.6f}'.format(macro_f1),
                  'micro_f1: {:.6f}'.format(micro_f1))

        # 3. 核心修改：以 Macro F1 为标准保存模型
        torch.save(model.state_dict(), args.save_path + '{}.pth'.format(epoch))

        if macro_f1 > max_macro_f1:
            max_macro_f1 = macro_f1
            max_micro_f1 = micro_f1
            best_epoch = epoch  # 更新最佳轮次索引
            patience_cnt = 0  # 重置早停计数器
        else:
            patience_cnt += 1

        # 4. 早停判定
        if patience_cnt == args.patience:
            break

        # 5. 清理历史权重 (只保留当前最佳 epoch 的文件)
        files = glob.glob(args.save_path + '*.pth')
        for f in files:
            try:
                epoch_nb = int(f[len(args.save_path):].split('.')[0])
                if epoch_nb < best_epoch:
                    os.remove(f)
            except:
                pass

    # 6. 训练完全结束后，清理比 best_epoch 更晚产生的文件
    files = glob.glob(args.save_path + '*.pth')
    for f in files:
        try:
            epoch_nb = int(f[len(args.save_path):].split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(f)
        except:
            pass

    time_use = time.time() - t

    return model, best_epoch, time_use, max_macro_f1, max_micro_f1


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=200, help='random seed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--run_times', type=int, default=3, help='run times')
parser.add_argument('--epochs', type=int, default=250, help='maximum number of epochs')
# Citation Networks Datasets：DBLPv7, ACMv9, Citationv1
# Social Networks Datasets：EN, DE
parser.add_argument('--source', type=str, default='ACMv9', help='source domain data')
parser.add_argument('--target', type=str, default='DBLPv7', help='target domain data')
parser.add_argument('--weight', type=float, default=10, help='trade-off parameter of MMD')
parser.add_argument('--source_pnum', type=int, default=0, help='the number of propagation layers on the source graph')
parser.add_argument('--target_pnum', type=int, default=10, help='the number of propagation layers on the target graph')

parser.add_argument('--remove_ratio', type=float, default=0.2, help='prune ratio')
parser.add_argument('--weight_degree', type=float, default=0.6, help='trade-off weight of degree centrality')
parser.add_argument('--weight_betweenness', type=float, default=0.1, help='trade-off weight of betweenness centrality')
parser.add_argument('--weight_closeness', type=float, default=0.1, help='trade-off weight of closeness centrality')
parser.add_argument('--weight_eigenvector', type=float, default=0.2, help='trade-off weight of eigenvector centrality')
        
args = parser.parse_args()
print(json.dumps(vars(args), indent=4))

# 定义要删除的节点索引列表
nodes_to_remove = get_nodes_to_be_removed_structural_pruning(args)

if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.source)
    # source_dataset = CitationDataset(path, args.source)
    source_dataset = CitationDatasetSource(path, args.source, nodes_to_remove=nodes_to_remove)
if args.source in {'EN', 'DE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data', args.source)

    source_dataset = TwitchDataset(path, args.source)
if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data',  args.target)
    target_dataset = CitationDataset(path, args.target)
if args.target in {'EN', 'DE'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data',  args.target)
    target_dataset = TwitchDataset(path, args.target)

source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)
print("Source Data: ", source_data)
print("Target Data: ", target_data)

args.num_classes = len(np.unique(source_dataset[0].y.numpy()))    
args.num_features = source_data.x.size(1)
args.save_path = './'   
if not osp.exists('./output'):
    os.mkdir('./output') 

# Run Experiment
macro_f1_dict = []
micro_f1_dict = []
all_dict = []
time_dict = []

for i in range(args.run_times):
    model, best_model, time_use, max_macro_f1, max_micro_f1 = train(args, source_data, target_data)
    acc, _, _, _ = evaluate(source_data, model)

    print('{} -> {}  source acc = {:.6f}, macro_f1 = {:.6f}, micro_f1 = {:.6f}'.format(args.source, args.target, acc, max_macro_f1, max_micro_f1))

    macro_f1_dict.append(max_macro_f1)
    micro_f1_dict.append(max_micro_f1)
    
    out_file2 = './output/{}-{}.txt'.format(args.source, args.target)
    if osp.exists(out_file2):
        with open(out_file2, 'a') as f:
            f.write('\n' + '='*50 + '\n')
            f.write(f'Experiment: {args.source} -> {args.target}\n')
            f.write(f'Macro F1 Max: {max_macro_f1:.4f}\n')
            f.write(f'Micro F1 Max: {max_micro_f1:.4f}\n')
            f.write(f'Args: {vars(args)}\n')
    else:
        with open(out_file2, 'w') as f:
            f.write('\n' + '='*50 + '\n')
            f.write(f'Experiment: {args.source} -> {args.target}\n')
            f.write(f'Macro F1 Max: {max_macro_f1:.4f}\n')
            f.write(f'Micro F1 Max: {max_micro_f1:.4f}\n')
            f.write(f'Args: {vars(args)}\n')
    
    time_dict.append(time_use)

macro_f1_dict_print = [float('{:.6f}'.format(i)) for i in macro_f1_dict]
micro_f1_dict_print = [float('{:.6f}'.format(i)) for i in micro_f1_dict]
all_dict_print = [float('{:.6f}'.format(i)) for i in all_dict]

print('mAcro:', macro_f1_dict_print,
      'mean {:.4f}'.format(np.mean(macro_f1_dict)),
      ' std {:.4f}'.format(np.std(macro_f1_dict)))
print('mIcro:', micro_f1_dict_print,
      'mean {:.4f}'.format(np.mean(micro_f1_dict)),
      ' std {:.4f}'.format(np.std(micro_f1_dict)))

print('Time Use Mean {:.4f}'.format(np.mean(time_dict)), ' Std {:.4f}'.format(np.std(time_dict)))
