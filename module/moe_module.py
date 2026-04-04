import torch
import torch.nn as nn
import torch.nn.functional as F

from module.ffn_module import TransformerFFN

'''
    Router：
        路由层：self.router通过训练来判断输入适合路由到哪些专家进行处理，本质上是一个可学习的门控层
        辅助损失：计算aux_loss
'''
class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 模型配置
        self.h_dim = None
        self.s_len = None
        self.b_size = None
        self.device = None
        self.config = config
        # 专家模型数量（选择前K个专家模型）
        self.top_k = config.num_experts_per_tok
        # 路由专家模型数量
        self.n_routed_experts = config.n_routed_experts
        # 打分函数
        self.scoring_func = config.scoring_func
        # aux_loss权重
        self.alpha = config.aux_loss_alpha
        # 是否使用sequence级aux_loss
        self.seq_aux = config.seq_aux
        # 是否对前k个专家模型的分数归一化
        self.norm_topk_weight = config.norm_topk_weight
        # 隐藏层维度
        self.hidden_dim = config.hidden_size
        self.linear = nn.Linear(self.hidden_dim, self.n_routed_experts, bias=False)
        self.topk_prob_norm_eps = config.topk_prob_norm_eps

    def forward(self, x):
        self.device = x.device
        # 获取 input shape
        # input shape: [batch_size, sequence_length, hidden_dim]
        self.b_size, self.s_len, self.h_dim = x.shape
        # 重组 input
        # input shape: [batch_size * sequence_length, hidden_dim]
        x = x.view(-1, self.h_dim)

        # 计算 logits
        # logits shape: [batch_size * sequence_length, n_routed_experts]
        logits = self.linear(x)

        # 获取 打分函数
        # 目前仅支持softmax打分
        if self.scoring_func == "softmax":
            # scores shape: [batch_size * sequence_length, n_routed_experts]
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        # 获取每个token前k个专家的权重和索引
        # topk_weight / topk_idx shape: [batch_size * sequence_length, top_k]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 归一化前K个专家权重
        if self.top_k > 1 and self.norm_topk_weight:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + self.topk_prob_norm_eps)

        if self.training and self.alpha > 0.0:
            if self.seq_aux:
                aux_loss = self.compute_seq_aux_loss(scores, topk_idx)
            else:
                aux_loss = self.compute_batch_aux_loss(scores, topk_idx)
        else:
            aux_loss = 0

        return topk_idx, topk_weight, aux_loss

    # 统计粒度Sequence级：先统计每个 Sequence 内部使用的专家，再按 Batch 平均
    def compute_seq_aux_loss(self, scores, topk_idx):
        # 获取 bsz, seq_len
        b_size, s_len = self.b_size, self.s_len
        # topk_idx shape: [batch_size, sequence_length * top_k]
        topk_idx = topk_idx.view(b_size, -1)
        # scores shape: [batch_size, sequence_length, n_routed_experts]
        scores = scores.view(b_size, s_len, -1)

        # 计算各个专家的使用数量
        # ce shape: [batch_size, n_routed_experts]
        ce = torch.zeros(b_size, self.n_routed_experts, device=self.device)
        ce.scatter_add_(
            1,
            topk_idx,
            torch.ones(b_size, s_len * self.top_k, device=self.device),
        )
        # 计算各个专家的使用率
        ce = ce.div(s_len * self.top_k / self.n_routed_experts)
        # scores.mean(dim=1): 计算专家平均得分
        # ce * scores.mean(dim=1): 专家使用率 * 专家平均得分 -> 计算每个token的aux_loss
        # (ce * scores.mean(dim=1)).sum(dim=1): 计算每条sentence的aux_loss
        # (ce * scores.mean(dim=1)).sum(dim=1).mean(): 计算每批样本的平均aux_loss
        # self.alpha为aux_loss的权重
        aux_loss = (ce * scores.mean(dim=1)).sum(dim=1).mean() * self.alpha
        return aux_loss

    # 统计粒度单Batch级：统计整个 Batch 中所有 Token 使用的专家模型
    def compute_batch_aux_loss(self, scores, topk_idx):
        mask_ce = F.one_hot(topk_idx.view(-1), num_classes=self.n_routed_experts)
        ce = mask_ce.float().mean(0)
        Pi = scores.mean(0)
        fi = ce * self.n_routed_experts
        aux_loss = (Pi * fi).sum() * self.alpha
        return aux_loss

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.orig_shape = None
        self.aux_loss = None
        self.config = config
        self.top_k = config.num_experts_per_tok
        # 专家层
        # 本质上就是num_experts个FFN
        self.experts = nn.ModuleList([TransformerFFN(config) for _ in range(config.num_experts)])

        # 共享专家层
        # 本质上就是n_shared_experts个FFN
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([TransformerFFN(config) for _ in range(config.n_shared_experts)])

        # 路由层
        # 决定选取
        self.router = Router(config)

    def forward(self, x):
        # x作为专家层的输入
        # shared_x作为共享专家层的输入
        shared_x = x
        self.orig_shape = x.shape
        bsz, seq_len, h_dim = self.orig_shape

        # 使用门控机制选择专家
        # topk_idx/topk_weight shape: [batch_size * sequence_length, top_k]
        topk_idx, topk_weight, aux_loss = self.router(x)

        # 展开topk_idx
        # topk_idx shape: [batch_size * sequence_length * top_k]
        topk_idx = topk_idx.view(-1)

        # 展开 x 以便于后续处理
        # x shape [batch_size * sequence_length, hidden_dim]
        x = x.view(-1, h_dim)

        # Routed Experts
        if self.training:
            # input shape: [batch_size * sequence_length, hidden_dim]
            # topk_idx shape: [batch_size * sequence_length * top_k]
            # topk_weight shape: [batch_size * sequence_length, top_k]
            output = self.moe_train(x, topk_idx, topk_weight)
        else:
            # input shape: [batch_size * sequence_length, hidden_dim]
            # topk_idx shape: [batch_size * sequence_length * top_k]
            # topk_weight shape: [batch_size * sequence_length * top_k, 1]
            output = self.moe_infer(x, topk_idx, topk_weight.view(-1, 1)).view(*self.orig_shape)

        # Shared Experts
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                output += expert(shared_x)
        self.aux_loss = aux_loss

        return output

    def moe_train(self, x, topk_idx, topk_weight):
        # 按照定义的top_k重复输入token, 每个token安排top_k个专家处理
        # 假设 input: [1,2,3,4]
        # 那么 input.repeat_interleave(2): [1,1,2,2,3,3,4,4]
        # input shape [batch_size * sequence_length * top_k, hidden_dim]
        x = x.repeat_interleave(self.top_k, dim=0)

        # output是空张量，和input形状相同
        # output shape [batch_size * sequence_length * top_k, hidden_dim]
        output = torch.empty_like(x, dtype=x.dtype)

        # 遍历所有专家(遍历所有FFN)
        #     找到所有指向专家i的token
        #     然后将这些token输入专家i进行处理
        #     最后将结果放回output对应位置
        for i, expert in enumerate(self.experts):
            # i: 第i个专家的索引
            # expert: 专家层
            # topk_idx: 每个token对应的专家层索引。
            # 比如第 N 个token的 topk_idx 为i，则将该token传入第i个专家层进行处理
            # input[topk_idx == i]: 取出input中所有被路由到第i个专家的token
            # 注：专家层中的FFN可能有M个，TopK选择的专家只有K个, M>>K
            expert_output = expert(x[topk_idx == i])
            # 将expert_output按topk_idx传回output
            output[topk_idx == i] = expert_output.to(output.dtype)

        # 加权求和
        # 最后的output意义是每个token经过专家处理后的加权结果
        # output shape: [batch_size * sequence_length, top_k, hidden_dim]
        output = output.view(*topk_weight.shape, -1)
        # topk_weight shape: [batch_size * sequence_length, top_k, 1]
        topk_weight = topk_weight.unsqueeze(-1)
        # output * topk_weight: torch的广播机制，会将topk_weight的最后一个维度复制hidden_dim份
        # (output * topk_weight).sum(dim=1) shape: [batch_size * sequence_length, hidden_dim]
        # output shape: [batch_size, sequence_length, hidden_dim]
        output = (output * topk_weight).sum(dim=1).view(*self.orig_shape)
        return output

    # MoE推理方法
    @torch.no_grad()
    def moe_infer(self, input, topk_idx, topk_weight):
        # 创建一个和input形状相同的零张量output
        output = torch.zeros_like(input)
        # 对专家索引进行排序
        # 示例：
        #   num_experts_per_tok = 2
        #   最原始的topk_idx形式: [[0,1],[2,3],[0,1],[2,3]]
        #   传入进来的拉平topk_idx形式: [0, 1, 2, 3, 0, 1, 2, 3]
        #   argsort：返回对数组从小到大排序后原始数组的索引值
        #   idx = topk_idx.argsort(): [0, 4, 1, 5, 2, 6, 3, 7]
        #   topk_idx[idx]: [0, 0, 1, 1, 2, 2, 3, 3]
        idxs = topk_idx.argsort()

        # 统计每个专家被分配到的token数量
        # bincount()：核心是统计每个专家被分配的 token 数量
        # cumsum(0)：计算频次的累计和，得到 “前 N 个专家的总 token 数”
        # 示例
        # tokens_per_expert: [2,4,6,8]
        # 第一个元素：代表第0个专家被分配了2个token
        # 第二个元素：代表第0个专家+第1个专家被分配了4个token
        # 第三个元素：代表第0个专家+第1个专家+第2个专家被分配了6个token
        # 第四个元素：代表第0个专家+第1个专家+第2个专家+第3个专家被分配了8个token
        tokens_per_expert = topk_idx.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        # token_idxs: [0, 2, 0, 2, 1, 3, 1, 3]
        # token_idxs shape: [batch_size * sequence_length]
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            # exp_token_idx shape: [end_idx-start_idx+1]
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            # expert_tokens shape: [end_idx-start_idx+1, hidden_dim]
            expert_tokens = input[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            # expert_out shape: [end_idx-start_idx+1, hidden_dim]
            expert_out = expert(expert_tokens).to(output.dtype)
            # 加权
            # 推理时才能用mul_这种原地操作
            # expert_out shape: [end_idx-start_idx+1, hidden_dim]
            expert_out.mul_(topk_weight[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            # exp_token_idx.view(-1, 1) shape: [end_idx-start_idx+1, 1]
            # exp_token_idx.view(-1, 1).repeat(1, input.shape[-1]) shape: [end_idx-start_idx+1, hidden_dim]
            # 示例：
            # 要输出的值：
            # output = tensor([[0., 0., 0.],
            #                  [0., 0., 0.],
            #                  [0., 0., 0.],
            #                  [0., 0., 0.]])
            # 要添加的值:
            # src = tensor([[0.1000, 0.2000, 0.3000],
            #               [0.4000, 0.5000, 0.6000]])
            # 索引:
            # idx = tensor([[0, 0, 0],
            #               [2, 2, 2]])
            # output.scatter_add_(0, idx, src) :
            # output = tensor([[0.1000, 0.2000, 0.3000],
            #                  [0.0000, 0.0000, 0.0000],
            #                  [0.4000, 0.5000, 0.6000],
            #                  [0.0000, 0.0000, 0.0000]])

            output.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, input.shape[-1]), expert_out)

        return output