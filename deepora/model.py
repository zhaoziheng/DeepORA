"""DeepORA imaging, auxiliary supervision and risk-factor fusion network.

Parameter names are preserved for compatibility with research checkpoints.
CT and MRI are trained as separate models. Input layout is B,C,H,W,D.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from positional_encodings.torch_encodings import PositionalEncoding3D
from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
    PlainConvEncoder,
)
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from .unet_decoder import UNetDecoder
from .risk import RISK_FEATURE_DIM


class DeepORA(nn.Module):
    """Separate CT/MRI network with report, segmentation and risk-factor branches."""

    def __init__(
        self,
        use_ct,
        use_mr,
        do_pretrain,
        add_brainstructure_seg,
        add_vessel_seg,
        add_report,
        target_labels,
        image_size=[384, 384, 32],
        num_transformer_layers=2,
        clinical_feature_dropout_rate=0.0,
        use_risk_factors=False,
        risk_encoder_dropout=0.0,
        text_encoder="FremyCompany/BioLORD-2023-C",
    ):
        super().__init__()
        if use_ct == use_mr:
            raise ValueError("Choose exactly one imaging modality per model.")
        if any(n < 32 or n % 32 for n in image_size):
            raise ValueError("Image dimensions must be positive multiples of 32.")
        (image_height, image_width, frames) = image_size
        self.use_ct = use_ct
        self.use_mr = use_mr
        self.clinical_feature_dropout_rate = clinical_feature_dropout_rate
        self.use_risk_factors = use_risk_factors
        self.risk_encoder_dropout = risk_encoder_dropout
        self.do_pretrain = do_pretrain
        self.add_brainstructure_seg = add_brainstructure_seg
        self.add_vessel_seg = add_vessel_seg
        self.add_report = add_report
        patch_size = [32, 32, 32]
        self.hw_patch_size = patch_size[0]
        self.frame_patch_size = patch_size[-1]
        self.num_spatial_tokens = (
            image_height
            // self.hw_patch_size
            * (image_width // self.hw_patch_size)
            * (frames // self.frame_patch_size)
        )
        query_dim = 512
        self.query_dim = query_dim
        pos_embedding = PositionalEncoding3D(query_dim)(
            torch.zeros(
                1,
                image_height // self.hw_patch_size,
                image_width // self.hw_patch_size,
                frames // self.frame_patch_size,
                query_dim,
            )
        )
        self.pos_embedding = rearrange(pos_embedding, "b h w d c -> (h w d) b c")
        if use_ct:
            self.ct_encoder = PlainConvEncoder(
                input_channels=1,
                n_stages=6,
                features_per_stage=(32, 64, 128, 256, 512, query_dim),
                conv_op=nn.Conv3d,
                kernel_sizes=3,
                strides=(1, 2, 2, 2, 2, 2),
                n_conv_per_stage=(2, 2, 2, 2, 2, 2),
                conv_bias=True,
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={"eps": 1e-05, "affine": True},
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs=None,
                return_skips=True,
                nonlin_first=False,
                pool="conv",
            )
            self.ct_encoder.apply(InitWeights_He(0.01))
            ct_seg_channels = (16 if add_brainstructure_seg else 0) + (
                1 if add_vessel_seg else 0
            )
            if ct_seg_channels > 0:
                self.ct_seg_decoder = UNetDecoder(
                    self.ct_encoder,
                    ct_seg_channels,
                    (2, 2, 2, 2, 2),
                    deep_supervision=False,
                    nonlin_first=False,
                )
                self.ct_seg_decoder.apply(InitWeights_He(0.01))
            self._ct_brain_seg_slice = slice(0, 16) if add_brainstructure_seg else None
            self._ct_vessel_seg_slice = (
                slice(16, 17)
                if add_brainstructure_seg and add_vessel_seg
                else slice(0, 1)
                if add_vessel_seg
                else None
            )
        if use_mr:
            self.unet3d_encoder = PlainConvEncoder(
                input_channels=4,
                n_stages=6,
                features_per_stage=(32, 64, 128, 256, 512, query_dim),
                conv_op=nn.Conv3d,
                kernel_sizes=3,
                strides=(1, 2, 2, 2, 2, 2),
                n_conv_per_stage=(2, 2, 2, 2, 2, 2),
                conv_bias=True,
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={"eps": 1e-05, "affine": True},
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs=None,
                return_skips=True,
                nonlin_first=False,
                pool="conv",
            )
            self.unet3d_encoder.apply(InitWeights_He(0.01))
            mr_seg_channels = (95 if add_brainstructure_seg else 0) + (
                1 if add_vessel_seg else 0
            )
            if mr_seg_channels > 0:
                self.mr_seg_decoder = UNetDecoder(
                    self.unet3d_encoder,
                    mr_seg_channels,
                    (1, 1, 1, 1, 1),
                    deep_supervision=False,
                    nonlin_first=False,
                )
                self.mr_seg_decoder.apply(InitWeights_He(0.01))
            self._mr_brain_seg_slice = slice(0, 95) if add_brainstructure_seg else None
            self._mr_vessel_seg_slice = (
                slice(95, 96)
                if add_brainstructure_seg and add_vessel_seg
                else slice(0, 1)
                if add_vessel_seg
                else None
            )
        if add_report:
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            config = AutoConfig.from_pretrained(text_encoder)
            config.add_pooling_layer = False
            self.report_encoder = AutoModel.from_pretrained(text_encoder, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
            self.report_proj = nn.Sequential(nn.Linear(768, query_dim), nn.GELU())
            for name, param in self.report_encoder.named_parameters():
                param.requires_grad = False
        self.target_labels = []
        if not do_pretrain:
            self.cls_token = nn.Parameter(torch.randn(1, 1, query_dim))
            decoder_layer = TransformerDecoderLayer(d_model=query_dim, nhead=8)
            decoder_norm = nn.LayerNorm(query_dim)
            self.transformer_decoder = TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=num_transformer_layers,
                norm=decoder_norm,
            )
            self.target_labels = target_labels
            cls_hidden_dim = 384
            if self.use_risk_factors:
                self.visual_head = nn.Sequential(
                    nn.Linear(query_dim, cls_hidden_dim), nn.GELU()
                )
                self.risk_embed_dim = 128
                if use_risk_factors:
                    self.factor_embedder = nn.Sequential(
                        nn.Linear(RISK_FEATURE_DIM, self.risk_embed_dim),
                        nn.GELU(),
                        nn.Dropout(self.risk_encoder_dropout),
                        nn.Linear(self.risk_embed_dim, self.risk_embed_dim),
                        nn.GELU(),
                    )
                head_input_dim = cls_hidden_dim
                if use_risk_factors:
                    head_input_dim += self.risk_embed_dim
                self.cls_task_heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(head_input_dim, 256),
                            nn.GELU(),
                            nn.Dropout(
                                self.risk_encoder_dropout if use_risk_factors else 0.0
                            ),
                            nn.Linear(256, 1),
                        )
                        for _ in target_labels
                    ]
                )
            else:
                self.cls_head = nn.Sequential(
                    nn.Linear(query_dim, cls_hidden_dim),
                    nn.GELU(),
                    nn.Linear(cls_hidden_dim, len(target_labels)),
                )

    def _zero_invalid_memory_tokens(self, embedding, valid_mask):
        if valid_mask is None:
            return embedding
        valid_mask = valid_mask.to(device=embedding.device, dtype=torch.bool)
        return embedding * valid_mask.view(1, -1, 1).to(dtype=embedding.dtype)

    def _run_cls_head(self, visual_feature, risk_input=None):
        """Apply the imaging head or concatenate the risk-factor embedding."""
        if self.use_risk_factors:
            hidden_feature = self.visual_head(visual_feature).squeeze(0)
            risk_emb = None
            if self.use_risk_factors:
                if risk_input is None:
                    raise ValueError("Risk-factor model requires risk_input.")
                risk_emb = self.factor_embedder(risk_input)
                if self.training and self.clinical_feature_dropout_rate > 0:
                    B = hidden_feature.shape[0]
                    drop = (
                        torch.rand(B, device=hidden_feature.device)
                        < self.clinical_feature_dropout_rate
                    )
                    if drop.any():
                        choice = torch.randint(0, 2, (B,), device=hidden_feature.device)
                        vis_mask = (
                            (~(drop & (choice == 0)))
                            .view(B, 1)
                            .to(hidden_feature.dtype)
                        )
                        risk_mask = (
                            (~(drop & (choice == 1)))
                            .view(B, 1)
                            .to(hidden_feature.dtype)
                        )
                        hidden_feature = hidden_feature * vis_mask
                        risk_emb = risk_emb * risk_mask
            logits = []
            for label_idx, task_head in enumerate(self.cls_task_heads):
                if risk_emb is not None:
                    task_input = torch.cat([hidden_feature, risk_emb], dim=-1)
                logits.append(task_head(task_input).squeeze(-1))
            return torch.stack(logits, dim=-1).unsqueeze(0)
        return self.cls_head(visual_feature)

    def mean_pooling(self, model_output, attention_mask):
        """Average unpadded text-token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-09
        )

    def forward(
        self,
        ct_input=None,
        mr_input=None,
        ct_mask=None,
        mr_mask=None,
        risk_input=None,
        report=None,
    ):
        """Return classification logits and enabled auxiliary predictions."""
        embeddings_list = []
        device = None
        B = None
        output_dict = None
        report_sim = None
        ct_logits_per_image = ct_logits_per_text = ct_seg_pred = ct_vessel_seg_pred = (
            None
        )
        mr_logits_per_image = mr_logits_per_text = mr_seg_pred = mr_vessel_seg_pred = (
            None
        )
        if self.add_report and report is not None and self.training:
            encoded_input = self.tokenizer(
                report, padding=True, truncation=True, return_tensors="pt"
            )
            encoded_input = {
                k: v.to(self.report_encoder.device) for (k, v) in encoded_input.items()
            }
            report_embedding = self.report_encoder(**encoded_input)
            report_embedding = self.mean_pooling(
                report_embedding, encoded_input["attention_mask"]
            )
            report_embedding = self.report_proj(report_embedding)
            report_embedding = F.normalize(report_embedding, dim=-1)
            report_sim = report_embedding @ report_embedding.t()
        if self.use_ct and ct_input is not None:
            (B, C, H, W, D) = ct_input.shape
            device = ct_input.device
            ct_latent_ls = self.ct_encoder(ct_input)
            ct_embedding = rearrange(ct_latent_ls[-1], "b dim h w d -> (h w d) b dim")
            pos = repeat(self.pos_embedding, "n o c -> n (o b) c", b=B).to(device)
            ct_embedding_w_pos = ct_embedding + pos
            ct_embedding_w_pos = self._zero_invalid_memory_tokens(
                ct_embedding_w_pos, ct_mask
            )
            embeddings_list.append(ct_embedding_w_pos)
            if (self.add_brainstructure_seg or self.add_vessel_seg) and self.training:
                ct_seg_combined = self.ct_seg_decoder(ct_latent_ls)
                if self._ct_brain_seg_slice is not None:
                    ct_seg_pred = ct_seg_combined[:, self._ct_brain_seg_slice]
                if self._ct_vessel_seg_slice is not None:
                    ct_vessel_seg_pred = ct_seg_combined[:, self._ct_vessel_seg_slice]
            if self.add_report and report is not None and self.training:
                ct_embedding_pooled = ct_embedding.mean(dim=0)
                ct_embedding_pooled = F.normalize(ct_embedding_pooled, dim=-1)
                logit_scale = self.logit_scale.exp()
                ct_logits_per_image = (
                    logit_scale * ct_embedding_pooled @ report_embedding.t()
                )
                ct_logits_per_text = ct_logits_per_image.t()
        if self.use_mr and mr_input is not None:
            (B, C, H, W, D) = mr_input.shape
            device = mr_input.device
            mr_latent_ls = self.unet3d_encoder(mr_input)
            mr_embedding = rearrange(mr_latent_ls[-1], "b dim h w d -> (h w d) b dim")
            pos = repeat(self.pos_embedding, "n o c -> n (o b) c", b=B).to(device)
            mr_embedding_w_pos = mr_embedding + pos
            mr_embedding_w_pos = self._zero_invalid_memory_tokens(
                mr_embedding_w_pos, mr_mask
            )
            embeddings_list.append(mr_embedding_w_pos)
            if (self.add_brainstructure_seg or self.add_vessel_seg) and self.training:
                mr_seg_combined = self.mr_seg_decoder(mr_latent_ls)
                if self._mr_brain_seg_slice is not None:
                    mr_seg_pred = mr_seg_combined[:, self._mr_brain_seg_slice]
                if self._mr_vessel_seg_slice is not None:
                    mr_vessel_seg_pred = mr_seg_combined[:, self._mr_vessel_seg_slice]
            if self.add_report and report is not None and self.training:
                mr_embedding_pooled = mr_embedding.mean(dim=0)
                mr_embedding_pooled = F.normalize(mr_embedding_pooled, dim=-1)
                logit_scale = self.logit_scale.exp()
                mr_logits_per_image = (
                    logit_scale * mr_embedding_pooled @ report_embedding.t()
                )
                mr_logits_per_text = mr_logits_per_image.t()
        if not self.do_pretrain:
            image_embedding = torch.cat(embeddings_list, dim=0)
            cls_token = repeat(self.cls_token, "n o c -> n (o b) c", b=B).to(device)
            memory_key_padding_mask = None
            (cls_token, _) = self.transformer_decoder(
                tgt=cls_token,
                memory=image_embedding,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=None,
            )
            cls_results = self._run_cls_head(cls_token, risk_input=risk_input)
            cls_results = rearrange(cls_results, "o b n -> n o b", o=1)
            output_dict = {
                label: pred.reshape(-1)
                for (label, pred) in zip(self.target_labels, cls_results)
            }
        return {
            "predictions": output_dict,
            "ct_seg_pred": ct_seg_pred,
            "ct_vessel_seg_pred": ct_vessel_seg_pred,
            "ct_logits_per_image": ct_logits_per_image,
            "ct_logits_per_text": ct_logits_per_text,
            "mr_seg_pred": mr_seg_pred,
            "mr_vessel_seg_pred": mr_vessel_seg_pred,
            "mr_logits_per_image": mr_logits_per_image,
            "mr_logits_per_text": mr_logits_per_text,
            "report_sim": report_sim,
        }
