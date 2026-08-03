"""解碼時共用的取樣工具（greedy / multinomial）"""

import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    sampling_mode: str = 'greedy',
    temperature: float = 1.0
) -> torch.Tensor:
    """
    根據 logits 選出下一個 token

    Args:
        logits: [batch_size, 1, vocab_size] 或 [batch_size, vocab_size]
        sampling_mode: 'greedy'（原本的行為，取機率最大值）或
                       'multinomial'（依機率分布隨機採樣，RL 訓練需要真正的隨機性時使用）
        temperature: multinomial 模式下的取樣溫度

    Returns:
        next_token: [batch_size, 1]
    """
    if logits.dim() == 3:
        logits = logits.squeeze(1)

    if sampling_mode == 'greedy':
        next_token = logits.argmax(dim=-1, keepdim=True)
    elif sampling_mode == 'multinomial':
        probs = F.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        raise ValueError(f"不支援的 sampling_mode: {sampling_mode}，請用 'greedy' 或 'multinomial'")

    return next_token
