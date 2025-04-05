import torch

from .Config import Config
from .util import norm_logits, sample, batch_norm_logits, greedy_sample
import torch.nn as nn
from time import perf_counter


class KVCacheModel(nn.Module):
    def __init__(self, model: torch.nn.Module, temperature: float = 1, top_k: int = 0, top_p: float = 0,
                 vocab_size: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model
        self._past_key_values = None
        # 保存seq中每一个tokens的logits 用于后续的验证
        # 这里的 prob_history 采用预先分配的方式
        self.prob_history = torch.empty((1, Config.BUFFER_SIZE, vocab_size),
                               device=model.device, dtype=torch.float32)
        self.current_verified_len = 0

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._vocab_size = vocab_size
        self.sum = 0

    @torch.no_grad()
    def generate(self, input: torch.Tensor, branch_prediction_num: int) -> torch.Tensor:
        tokens_id = self._generate_some_tokens_with_kvcache(input, branch_prediction_num)

        return tokens_id

    def _generate_some_tokens_with_kvcache(self,
                               prefix: torch.Tensor,
                               branch_prediction_num: int
                               , sample_method='greedy') -> torch.Tensor:
        """

        :param prefix (torch.Tensor): the prefix
        :param branch_prediction_num: how many times drafter guesses
        :return:Torch.Tensor: prefix+generated tokens
        """
        tokens_id = prefix

        for _ in range(branch_prediction_num):
            last_q = self._forward_with_kvcache(tokens_id)
            if sample_method=='greedy':
                next_token_id = greedy_sample(last_q)
            elif sample_method=='sample':
                next_token_id = sample(last_q)
            tokens_id = torch.cat([tokens_id, next_token_id], dim=1)
        return tokens_id

    @torch.no_grad()
    def rollback(self, end_pos: int):
        assert self._past_key_values is not None, "past_key_values is None"

        self._past_key_values = [
            (k[..., :end_pos, :], v[..., :end_pos, :])
            for k, v in self._past_key_values
        ]

        if self.prob_history is not None:
            # self.prob_history = self.prob_history[:, :end_pos, :]
            self.current_verified_len = end_pos


    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 第一次推理没有保存kvcache ，此时调用forward
        if self._past_key_values is None:
            outputs = self._model(input_ids)

            # logit shape is (batch_size, sequence_length, vocab_size)
            # todo 等价于 self.prob_history = outputs.logits
            # self.prob_history = outputs.logits[:, :, :self._vocab_size]
            seq_len = outputs.logits.size(1)
            self.prob_history[:, :seq_len, :] = outputs.logits
            # 对每个token的概率进行归一化 todo 效率太低，可以重写函数进行优化
            # for i in range(self.prob_history.shape[-2]):
            #     self.prob_history[:, i, :] = norm_logits(self.prob_history[:, i, :],
            #                                              self._temperature,
            #                                              self._top_k,
            #                                              self._top_p)
            self.prob_history[:, :seq_len, :] = batch_norm_logits(self.prob_history[:, :seq_len, :],
                                                  self._temperature,
                                                  self._top_k,
                                                  self._top_p)
            # 记录kvcache
            self._past_key_values = outputs.past_key_values
            # todo 只要最后一个为什么前面对所有token 进行归一化？
            # last_q = self.prob_history[:, -1, :]
            last_q = self.prob_history[:, seq_len-1, :]
            self.current_verified_len = seq_len
        else:
            # 有kvcache 进行的推理
            # return the last token's logits
            # 注意： 这里的 seq_len 不是input_ids 的len 是含有kvcache的 seq len 不包含上次cat 的tokens
            seq_len = self._past_key_values[0][0].shape[2]
            # caution 这里获取 seq_len 并不是脱裤子放屁，很重要的一步操作，因为seq_len 后面有一些tokens被接受了，但是没有kvcache
            last_input_id = input_ids[:, seq_len:]
            # 保证 input_id.dim() == 2
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            # 进行推理，传入当前 token_id 和 past_key_values ,使用use_cache 进行推理
            outputs = self._model(input_ids = last_input_id,
                                  past_key_values=self._past_key_values,
                                  use_cache=True)

            not_cached_q = outputs.logits[:,:,:self._vocab_size]

            if not_cached_q.dim() == 2:
                not_cached_q = torch.torch.unsqueeze(not_cached_q, 0)

            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:,i,:] = norm_logits(not_cached_q[:,i,:],
                                                  self._temperature,
                                                  self._top_k,
                                                  self._top_p)

            cur_len = self.current_verified_len
            self.current_verified_len += outputs.logits.size(1)

            # start = perf_counter()
            # self.prob_history = torch.cat([self.prob_history, not_cached_q], dim=1)
            self.prob_history[:, cur_len:self.current_verified_len, :] = not_cached_q[:,:,:]
            # self.sum += perf_counter() - start

            last_q = not_cached_q[:,-1 ,:]
            self._past_key_values = outputs.past_key_values

        return last_q
