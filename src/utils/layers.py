# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import numpy as np
import math
import copy
from typing import Tuple, Optional
from torch.nn import functional as F
from torch import Tensor
import logging

from utils.constants import *


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask, kv_cache=None):
        query = self.query_layer(input_tensor)
        key = self.key_layer(input_tensor)
        value = self.value_layer(input_tensor)

        if kv_cache is not None:
            # print("shape of kv_cache['key']", kv_cache["key"].shape)
            # print("shape of key", key.shape)
            key = torch.cat([kv_cache["key"], key], dim=1)
            value = torch.cat([kv_cache["value"], value], dim=1)
        else:
            kv_cache = {}

        kv_cache["key"] = key
        kv_cache["value"] = value

        query = self.transpose_for_scores(query).permute(0, 2, 1, 3)
        key = self.transpose_for_scores(key).permute(0, 2, 3, 1)
        value = self.transpose_for_scores(value).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query, key)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states, kv_cache


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask, kv_cache=None):
        attention_output, new_kv_cache = self.multi_head_attention(
            hidden_states, attention_mask, kv_cache
        )
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output, new_kv_cache


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=False,
        kv_caches=None,
    ):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
            kv_caches (list): a list of key and value caches for each transformer layer
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        present_kv_caches = []

        for i, layer_module in enumerate(self.layer):
            layer_kv_cache = kv_caches[i] if kv_caches[i] is not None else None
            if layer_kv_cache:
                layer_outputs = layer_module(
                    hidden_states[:, -1:, :], attention_mask, layer_kv_cache
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, layer_kv_cache
                )
            hidden_states = layer_outputs[0]
            present_kv_caches.append(layer_outputs[1])
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, present_kv_caches


class AutoRegressiveWrapper(nn.Module):
    """
    AutoRegressiveWrapper is a wrapper for transformer model to generate auto-regressive sequence.

    Args:
        transformer (nn.Module): the transformer model
        reason_steps (int): the number of steps to generate auto-regressive sequence
        hidden_size (int): the hidden size of the transformer model

    Returns:
        all_outputs (torch.Tensor): the output of the auto-regressive sequence
    """

    def __init__(self, transformer, hidden_size, reason_step=0, layer_norm_eps=1e-12):
        super(AutoRegressiveWrapper, self).__init__()
        self.transformer = transformer
        self.n_layers = len(transformer.layer)
        self.reason_step = reason_step
        self.hidden_size = hidden_size
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(p=0.2)

    def _prepare_attention_mask(self, batch_size, seq_len, device, padding_mask):
        mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device)
        mask = torch.tril(mask)
        mask = mask.masked_fill(mask == 0, -1e10).masked_fill(mask == 1, 0.0)
        mask = mask.masked_fill(padding_mask, -1e10)
        return mask

    def _prepare_padding_mask(self, input_lens, device, step):
        input_lens = input_lens + step
        batch_size = len(input_lens)
        max_item_seq_len = MAX_ITEM_SEQ_LEN + step
        padding_mask = (
            torch.arange(max_item_seq_len)
            .unsqueeze(0)
            .expand(batch_size, max_item_seq_len)
            .to(device)
        )
        padding_mask = padding_mask < (max_item_seq_len - input_lens.unsqueeze(1))
        return padding_mask.unsqueeze(1).unsqueeze(2)

    def forward(self, input_embs, input_lens):
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device

        past_key_values = [None] * self.n_layers
        all_outputs = []

        for step in range(self.reason_step + 1):
            curr_seq_len = seq_len + step

            padding_mask = self._prepare_padding_mask(input_lens, device, step)
            attention_mask = self._prepare_attention_mask(
                batch_size, curr_seq_len, device, padding_mask
            )

            input_embs = self.LayerNorm(input_embs)
            input_embs = self.dropout(input_embs)
            outputs = self.transformer(
                input_embs, attention_mask, kv_caches=past_key_values
            )
            last_hidden_states = outputs[0][-1][:, -1:, :]
            all_outputs.append(last_hidden_states)
            past_key_values = outputs[1]

            if step == self.reason_step:
                break

            input_embs = last_hidden_states

        if self.reason_step == 0:
            return all_outputs[0]

        return torch.cat(
            all_outputs, dim=1
        )  # [batch_size, reason_steps+1, hidden_size]


class ReaRecAutoRegressiveWrapper(AutoRegressiveWrapper):
    def __init__(self, transformer, hidden_size, reason_step):
        super(ReaRecAutoRegressiveWrapper, self).__init__(
            transformer,
            hidden_size,
            reason_step,
        )
        if reason_step > 0:
            self.reason_pos_emb = nn.Embedding(reason_step, hidden_size)

    def forward(self, input_embs, input_lens, noise_factor=0.0, reason_step=None):
        batch_size, seq_len, _ = input_embs.size()
        device = input_embs.device

        repeat_batch_size_factor = (noise_factor > 0.0) + 1
        input_embs = input_embs.repeat(repeat_batch_size_factor, 1, 1)
        input_lens = input_lens.repeat(repeat_batch_size_factor)

        past_key_values = [None] * self.n_layers
        all_outputs = []
        all_noise_outputs = []

        reason_step = reason_step if reason_step is not None else self.reason_step
        for step in range(reason_step + 1):
            curr_seq_len = seq_len + step

            padding_mask = self._prepare_padding_mask(input_lens, device, step)
            attention_mask = self._prepare_attention_mask(
                batch_size * repeat_batch_size_factor,
                curr_seq_len,
                device,
                padding_mask,
            )

            input_embs = self.LayerNorm(input_embs)
            input_embs = self.dropout(input_embs)
            outputs = self.transformer(
                input_embs, attention_mask, kv_caches=past_key_values
            )
            last_hidden_states = outputs[0][-1][:batch_size, -1:, :]
            all_outputs.append(last_hidden_states)
            if noise_factor > 0:
                all_noise_outputs.append(outputs[0][-1][batch_size:, -1:, :])

            if step == reason_step:
                break

            past_key_values = outputs[1]

            # w/ positional embedding
            new_pos_emb = self.reason_pos_emb(
                torch.tensor([step], device=device)
            ).expand(batch_size, 1, -1)
            input_embs = last_hidden_states + new_pos_emb

            # w/o positional embedding
            # input_embs = last_hidden_states

            if noise_factor > 0.0:
                noise = (torch.randn_like(input_embs) * noise_factor).to(device)
                noise_input_embs = input_embs + noise
                input_embs = torch.cat([input_embs, noise_input_embs], dim=0)

        all_outputs = torch.cat(all_outputs, dim=1)
        if noise_factor > 0.0:
            all_noise_outputs = torch.cat(all_noise_outputs, dim=1)
            all_outputs = torch.cat([all_outputs, all_noise_outputs], dim=0)

        return all_outputs  # [batch_size, reason_steps+1, hidden_size]

