import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from models.BaseModel import SequentialModel
from utils import layers
from utils.constants import *


class ReaSASRec(SequentialModel):
    reader = "BaseReader"
    extra_log_args = [
        "emb_size",
        "num_layers",
        "num_heads",
        "dropout",
        "temperature",
        "reason_step",
        "temp_scale",
        "noise_factor",
    ]

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--emb_size", type=int, default=256, help="Size of embeddings"
        )
        parser.add_argument(
            "--num_layers", type=int, default=2, help="Number of layers"
        )
        parser.add_argument("--num_heads", type=int, default=2, help="Number of heads")
        parser.add_argument(
            "--inner_size", type=int, default=300, help="Size of inner hidden layers"
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.5,
            help="Dropout probability for each deep layer",
        )
        parser.add_argument(
            "--hidden_act",
            type=str,
            default="gelu",
            help="Activation function of hidden layers",
        )
        parser.add_argument(
            "--layer_norm_eps",
            type=float,
            default=1e-12,
            help="Layer normalization epsilon",
        )
        parser.add_argument(
            "--initializer_range",
            type=float,
            default=0.02,
            help="Initializer range for parameters",
        )
        parser.add_argument(
            "--temperature", type=float, default=0.07, help="Temperature"
        )
        parser.add_argument(
            "--reason_step",
            type=int,
            default=2,
            help="Number of steps to reason over the sequence",
        )
        parser.add_argument(
            "--pl_weight",
            type=float,
            default=1.0,
            help="Progressive learning weight",
        )
        parser.add_argument(
            "--temp_scale",
            type=float,
            default=5.0,
            help="Epoch steps for progressive learning",
        )
        parser.add_argument(
            "--noise_factor", type=float, default=0.01, help="Noise factor"
        )
        parser.add_argument(
            "--cl_weight", type=float, default=1.0, help="Contrastive loss weight"
        )
        parser.add_argument("--warmup", type=int, default=2, help="Warmup epoch")
        return SequentialModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        SequentialModel.__init__(self, args, corpus)
        self.emb_size = args.emb_size
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.hidden_size = self.emb_size  # same as emb_size
        self.inner_size = args.inner_size
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.initializer_range = args.initializer_range
        self.temperature = args.temperature
        self.reason_step = args.reason_step
        self.pl_weight = args.pl_weight
        self.temp_scale = args.temp_scale
        self.noise_factor = args.noise_factor
        self.cl_weight = args.cl_weight
        self.warmup_epoch = args.warmup
        self._define_params()
        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _define_params(self):
        self.item_emb = nn.Embedding(
            self.item_num, self.emb_size, padding_idx=self.item_num - 1
        )
        self.pos_emb = nn.Embedding(
            MAX_ITEM_SEQ_LEN + 1, self.emb_size, padding_idx=MAX_ITEM_SEQ_LEN
        )
        self.trm_encoder = layers.TransformerEncoder(
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.dropout,
            attn_dropout_prob=self.dropout,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.model = layers.ReaRecAutoRegressiveWrapper(
            self.trm_encoder, self.hidden_size, self.reason_step
        )

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.item_num)

    def forward(self, feed_dict: dict, epoch=0, stage="train") -> dict:
        item_seq_ids, item_seq_len = feed_dict[ITEM_SEQ], feed_dict[ITEM_SEQ_LEN]
        B = item_seq_ids.size(0)

        # Left Padding for item_seq_ids
        padding_mask = item_seq_ids != (self.item_num - 1)
        valid_pos_ids = torch.cumsum(padding_mask.long(), dim=1) - 1
        pos_ids = torch.where(padding_mask, valid_pos_ids, MAX_ITEM_SEQ_LEN)
        pos_embs = self.pos_emb(pos_ids)

        item_embs = self.item_emb(item_seq_ids)
        input_embs = item_embs + pos_embs

        model_output = self.model(
            input_embs,
            item_seq_len,
            noise_factor=(
                self.noise_factor
                if epoch > self.warmup_epoch and stage == "train"
                else 0.0
            ),
        )

        test_item_embs = self.item_emb.weight

        seq_embs = model_output[:B, -1, :]
        logits = (
            torch.matmul(seq_embs, test_item_embs.transpose(0, 1)) / self.temperature
        )
        return {
            "prediction": logits[:B],
            "model_output": model_output,
            "test_item_embs": test_item_embs,
            ITEM_ID: feed_dict[ITEM_ID],
        }

    def loss(self, out_dict: dict) -> torch.Tensor:
        batch_size = len(out_dict[ITEM_ID])
        logits = out_dict["prediction"]
        labels = out_dict[ITEM_ID]
        repeat_times = out_dict["model_output"].shape[0] // len(labels)
        loss = self.loss_fct(logits, labels)

        test_item_embs = out_dict["test_item_embs"]  # (N, D)

        # progressive learning loss
        thinking_embs = out_dict["model_output"][:, :-1, :]  # (B, T, D)
        T = thinking_embs.shape[1]
        all_logits = torch.einsum(
            "btd,nd->btn", thinking_embs[:batch_size], test_item_embs
        )
        temp_scales = self.temperature * (
            self.temp_scale ** torch.arange(T, 0, -1).to(logits.device)
        )
        scaled_logits = all_logits / temp_scales.view(1, T, 1)
        pl_loss = self.loss_fct(
            scaled_logits.view(-1, test_item_embs.size(0)),
            labels.repeat_interleave(T),
        )

        # thinking contrastive loss
        if repeat_times > 1 and self.cl_weight > 0:
            view1_embs = out_dict["model_output"][:batch_size, 1:, :]
            view2_embs = out_dict["model_output"][batch_size:, 1:, :]
            T = view1_embs.shape[1]
            similarity_matrix = (
                torch.einsum("btd,ktd->btk", view2_embs, view1_embs) / self.temperature
            )
            labels = torch.arange(similarity_matrix.shape[0]).to(view1_embs.device)
            cl_loss = self.loss_fct(
                similarity_matrix.permute(0, 2, 1),
                labels.unsqueeze(1).expand(-1, T),
            )

        return (
            loss
            + self.pl_weight * pl_loss
            + self.cl_weight * cl_loss
        )
