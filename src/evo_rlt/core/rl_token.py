from __future__ import annotations

import torch
import torch.nn as nn


class RLTokenModule(nn.Module):
    """RL Token encoder-decoder with configurable number of RL tokens.

    Encoder appends N learned <rl> embeddings to VLA tokens, runs a transformer,
    and reads out the last N positions as z_rl (B, N, D). For downstream RL,
    z_rl is mean-pooled to (B, D).

    Decoder autoregressively reconstructs VLA embeddings from the multi-token
    bottleneck using teacher forcing and causal masking.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        nhead: int = 8,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        ff_dim: int | None = None,
        num_rl_tokens: int = 1,
        inference_only: bool = False,
        normalize_encoder_output: bool = False,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * token_dim

        self.token_dim = token_dim
        self.num_rl_tokens = num_rl_tokens
        self.inference_only = inference_only
        self.normalize_encoder_output = normalize_encoder_output
        self.rl_token_embed = nn.Parameter(torch.randn(1, num_rl_tokens, token_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)
        # Pre-norm layers do not normalize the final residual output. Normalize
        # each RL token before decoding or pooling for the downstream policy.
        # Identity preserves the original behavior and checkpoint keys by default.
        self.encoder_out_norm = nn.LayerNorm(token_dim) if normalize_encoder_output else nn.Identity()

        if not inference_only:
            dec_layer = nn.TransformerDecoderLayer(
                d_model=token_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)
            self.out_proj = nn.Linear(token_dim, token_dim)

    def encode(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """Encode VLA tokens into RL token(s).

        Args:
            vla_tokens: (B, M, D) -- final VLA token embeddings (should be detached)

        Returns:
            z_rl: (B, D) -- mean-pooled RL token for downstream RL state
        """
        z_rl_multi = self.encode_multi(vla_tokens)
        return z_rl_multi.mean(dim=1)  # (B, D)

    def encode_multi(self, vla_tokens: torch.Tensor) -> torch.Tensor:
        """Encode VLA tokens into multiple RL tokens (for decoder).

        Returns:
            z_rl: (B, N, D) where N = num_rl_tokens
        """
        B = vla_tokens.shape[0]
        rl = self.rl_token_embed.expand(B, -1, -1)  # (B, N, D)
        x = torch.cat([vla_tokens, rl], dim=1)  # (B, M+N, D)
        out = self.encoder(x)
        return self.encoder_out_norm(out[:, -self.num_rl_tokens:, :])  # (B, N, D)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ):
        # Several callers use strict=False to omit the decoder. Do not let that
        # silently discard a trained normalization or add one to a legacy model.
        if prefix + "rl_token_embed" in state_dict:
            norm_keys = [prefix + "encoder_out_norm." + name for name in ("weight", "bias")]
            has_norm = any(key in state_dict for key in norm_keys)
            if has_norm != self.normalize_encoder_output:
                error_msgs.append(
                    f"{prefix}normalize_encoder_output={self.normalize_encoder_output} does not match "
                    f"the checkpoint; construct the module with normalize_encoder_output={has_norm}."
                )
            elif has_norm and not all(key in state_dict for key in norm_keys):
                error_msgs.append(f"{prefix}encoder_out_norm checkpoint must contain both weight and bias.")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )

    def decode(self, z_rl: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
        """Autoregressive reconstruction with teacher forcing.

        Args:
            z_rl: (B, D) single token or (B, N, D) multi-token
            teacher_tokens: (B, M, D) -- stop-gradiented VLA embeddings

        Returns:
            pred: (B, M, D)
        """
        M = teacher_tokens.shape[1]
        # Handle both single and multi-token z_rl
        if z_rl.dim() == 2:
            memory = z_rl.unsqueeze(1)  # (B, 1, D)
        else:
            memory = z_rl  # (B, N, D)
        N = memory.shape[1]
        # Shifted input: [rl_tokens..., z_1, ..., z_{M-N}]
        dec_input = torch.cat([memory, teacher_tokens[:, :M - N]], dim=1)  # (B, M, D)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            M, device=z_rl.device
        )
        out = self.decoder(tgt=dec_input, memory=memory, tgt_mask=causal_mask)
        return self.out_proj(out)

    def reconstruction_loss(
        self,
        vla_tokens: torch.Tensor,
        dim_std: torch.Tensor | None = None,
        gamma: float = 0.0,
    ) -> torch.Tensor:
        """L_ro = E[|| (pred_i - z_bar_i) * D^{-gamma} ||^2].

        gamma=0 reduces to vanilla MSE (paper Eq. 2). gamma=1 is full whitening
        in the per-dim std metric. gamma=0.5 partially compensates the few
        high-variance dims that otherwise dominate the gradient.
        """
        z_bar = vla_tokens.detach()
        z_rl_multi = self.encode_multi(z_bar)
        pred = self.decode(z_rl_multi, z_bar)
        diff = pred - z_bar
        if gamma > 0 and dim_std is not None:
            weight = dim_std.clamp_min(1e-6).pow(-gamma).to(device=diff.device, dtype=diff.dtype)
            diff = diff * weight
        return (diff ** 2).mean()
