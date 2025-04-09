import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np

from src.Config import Config


def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def model_map(args):
    vocab_size = {
    "Qwen2.5-0.5B-Instruct": 151936,
    "Qwen2.5-1.5B-Instruct": 151936,
    "Qwen2.5-3B-Instruct": 151936,
    "Qwen2.5-7B-Instruct": 151936,
    "Llama-3.2-1B-Instruct":128256,
    "Llama-3.2-3B-Instruct":128256,
    "Llama-3.1-8B-Instruct":128256,
    "Llama-3.1-70B-Instruct":128256,

    }
    # model_dir_map = {
    #     "Qwen2.5-1.5B-Instruct": "D:\\model\\Qwen2.5-1.5B-Instruct",
    #     "Qwen2.5-0.5B-Instruct": "D:\\model\\Qwen2.5-0.5B-Instruct",
    # }
    model_dir_map = {
        "Qwen2.5-7B-Instruct": f"{Config.MODEL_DIR}/Qwen2.5-7B-Instruct",
        "Qwen2.5-1.5B-Instruct": f"{Config.MODEL_DIR}/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct": f"{Config.MODEL_DIR}/Qwen2.5-3B-Instruct",
        "Qwen2.5-0.5B-Instruct": f"{Config.MODEL_DIR}/Qwen2.5-0.5B-Instruct",
        "Llama-3.2-1B-Instruct": f"{Config.MODEL_DIR}/Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct": f"{Config.MODEL_DIR}/Llama-3.2-3B-Instruct",
        "Llama-3.1-8B-Instruct": f"{Config.MODEL_DIR}/Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct": f"{Config.MODEL_DIR}/Llama-3.1-70B-Instruct",
    }
    # set vocab size and
    # caution: all the models' vocab size should be the same
    args.vocab_size = vocab_size[args.draft_models[0]]
    args.draft_models_dir = [model_dir_map[model_name] for model_name in args.draft_models]
    args.target_model_dir = model_dir_map[args.target_model]
    if args.model_name is not None and args.model_name != "":
        # 作为 target model 进行测试
        print(f"args.model is {args.model_name}")
        args.target_model_dir = model_dir_map[args.model_name]


def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    parser.add_argument('--data_path', type=str, default="../data")

    parser.add_argument('--draft_models', type=str, nargs='+', default=["Llama-3.2-1B-Instruct"])
    parser.add_argument('--target_model', type=str, default="Llama-3.1-8B-Instruct")

    parser.add_argument('--exp_name', '-e', type=str, default="test", help='folder name for storing results.')
    # 实验相关
    # todo add more eval mode
    parser.add_argument('--eval_mode', type=str, default="default",
                        choices=["default","single_model", "sd", "para_sd",], help='eval mode.')

    parser.add_argument("--model_name", type=str, default="",
                        help="when '--eval_mode' is single_model ,use this to specify the model name.")

    parser.add_argument('--num_samples_per_task', '-n', type=int, default=1,
                        help='num_samples for a task (prompt) in humaneval dataset.')
    parser.add_argument('--seed', '-s', type=int, default=1234,
                        help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=1024, help='max token number generated.')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=0, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    # 框架设置相关
    # todo delete gamma later
    parser.add_argument('--gamma', type=int, default=4, help='guess time.')
    parser.add_argument('--branch_num', type=int, default=4, help='branch number for drafters.')
    parser.add_argument("--branch-prediction-num", type=int, default=2, help="branch prediction number for smallest drafter.")

    args = parser.parse_args()
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_map(args)
    # if args.eval_mode == "para_sd":
    args.rank = int(os.environ["RANK"])  # 自动从环境变量获取
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    return args



def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab_size)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (batch, vocab_size)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  vocab_size)
    """
    assert logits.dim() == 2
    if temperature == 0:
        idx = logits.argmax(dim=1)
        new_logits = torch.zeros_like(logits, device=logits.device)
        new_logits[:, idx] = 1
        return new_logits.float()
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs

def sample(probs : torch.Tensor, num_samples: int = 1):
    """
    从概率分布中采样
    :param probs:   shape (batch, vocab_size)
    :param num_samples: 采样个数
    :return:
    """
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next

def greedy_sample(probs: torch.Tensor, num_samples: int = 1):
    """
    从概率分布中进行贪心采样
    :param probs:   shape (batch, vocab_size)
    :param num_samples: 采样个数
    :return:
    """
    # 获取每个批次中概率最大的元素的索引
    _, idx_next = torch.topk(probs, k=num_samples, dim=-1)
    return idx_next

def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True)
    return x_max / x_max_sum


def batch_norm_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """ 批量处理版概率归一化 (支持3D输入)

    Args:
        logits (torch.Tensor): shape (batch_size, seq_len, vocab_size)
        temperature (float): 温度系数
        top_k (int): top-k采样
        top_p (float): top-p采样

    Returns:
        torch.Tensor: 处理后的概率分布，shape与输入一致
    """
    # 保存原始shape
    original_shape = logits.shape
    # 合并batch和sequence维度
    logits = logits.view(-1, original_shape[-1])

    if temperature == 0:
        # 贪婪采样模式
        indices = logits.argmax(dim=-1)
        probs = torch.zeros_like(logits)
        probs.scatter_(1, indices.unsqueeze(1), 1.0)
    else:
        # 应用温度系数
        logits = logits / temperature
        # 并行进行top-k和top-p过滤
        logits = batch_top_k_top_p_filter(logits, top_k, top_p)
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)

    # 恢复原始shape
    return probs.view(original_shape)


def batch_top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """ 批量版 top-k + top-p 过滤 (支持3D输入),对第二维度进行过滤

    Args:
        logits (torch.Tensor): shape (batch_size, seq_len, vocab_size) 或 (batch_size*seq_len, vocab_size)
        top_k (int): 保留概率最高的k个token
        top_p (float): 保留累积概率达到p的最小token集合

    Returns:
        torch.Tensor: 过滤后的 logits
    """
    # Top-k过滤（并行化实现）
    if top_k > 0:
        vocab_size = logits.size(-1)
        k = min(top_k, vocab_size)

        # 获取每个位置的第k大值
        topk_values = torch.topk(logits, k, dim=-1, sorted=True)[0][..., -1:]  # [..., k]->[..., -1]

        # 创建mask并过滤
        mask = logits < topk_values
        logits = logits.masked_fill(mask, float('-inf'))

    # Top-p过滤（并行化实现）
    if top_p > 0.0:
        # 按概率降序排列
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)  # [..., vocab]

        # 计算累积概率
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)  # [..., vocab]

        # 创建动态阈值mask
        mask = cumulative_probs > top_p

        # 保留第一个超过阈值的位置
        mask[..., 1:] |= mask[..., :-1].clone()  # 右移传播
        mask[..., 0] = False  # 确保至少选择一个token

        # 逆排序恢复原始索引
        inverse_mask = mask.scatter(-1, sorted_indices, mask)

        # 应用过滤
        logits = logits.masked_fill(inverse_mask, float('-inf'))

    return logits


if __name__ == '__main__':
    # 输入 probs (batch=2, vocab_size=1，仅用于演示，实际应为 (batch, vocab_size))
    probs = torch.tensor([[0.9], [0.8]])
    print(probs.shape)

    # 采样 num_samples=1
    idx_next = sample(probs, num_samples=1)
    print(idx_next)
    # 输出类似: tensor([[0], [0]])  # 因为概率最高类别是索引 0

    # 采样 num_samples=3
    idx_next = sample(probs, num_samples=3)
    print(idx_next)
    # 输出类似: tensor([[0, 0, 0], [0, 0, 0]])