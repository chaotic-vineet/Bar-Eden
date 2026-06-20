"""
From-scratch transformer models (Act 1.2).

Every parameter is a plain torch.Tensor with requires_grad=True,
created on-device. No torch.nn.Module.

Every class implements the same three-method interface:
    __call__(x)     - the forward pass
    parameters()    - flat list of learnable tensors
    config_dict()   - architecture dict for logging

The live model is Transformer. Commented-out classes (MhaModel,
MhaFfnModel, MiniTransformer) are museum copies from the ablation
study — they predate dropout, fused attention, weight tying, and
residual scaling.

Naming conventions (consistent across all classes):
    embedding_dim   — model width (never dim_emb, EMBEDDING_DIM)
    context_size    — sequence length (never CONTEXT_SIZE)
    ffn_dim         — FFN hidden dimension (never dim_ffn)
    num_heads       — number of attention heads (never h alone)
    n_blocks        — number of transformer blocks (never N)
    dim_qk          — per-head query/key dimension
    dim_v           — per-head value dimension
    init_scale      — extra multiplier on weight init (for residual scaling)
"""

import torch
import math


# ═══════════════════════════════════════════════════════════════
#  BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════

class Embedding:
    """
    Learnable embedding table: (vocab_dim, embedding_dim).
    ids (B, T) → vectors (B, T, embedding_dim).
    """
    def __init__(self, vocab_dim, embedding_dim, device, generator):
        self.embedding_matrix = (
            torch.randn(vocab_dim, embedding_dim,
                        device=device, generator=generator) * 0.1
        ).requires_grad_(True)

    def __call__(self, x):
        return self.embedding_matrix[x]

    def parameters(self):
        return [self.embedding_matrix]

    def config_dict(self):
        vocab_dim, embedding_dim = self.embedding_matrix.shape
        return {"vocab_dim": vocab_dim, "embedding_dim": embedding_dim}


class PositionalEncoding:
    """
    Learnable positional embedding: (context_size, embedding_dim).
    Broadcasts when added to (B, context_size, embedding_dim).
    """
    def __init__(self, context_size, embedding_dim, device, generator):
        self.positional_encoding_matrix = (
            torch.randn(context_size, embedding_dim,
                        device=device, generator=generator) * 0.1
        ).requires_grad_(True)

    def __call__(self):
        return self.positional_encoding_matrix

    def parameters(self):
        return [self.positional_encoding_matrix]

    def config_dict(self):
        context_size, embedding_dim = self.positional_encoding_matrix.shape
        return {"context_size": context_size, "embedding_dim": embedding_dim}


class Linear:
    """
    out = x @ weight + bias.

    weight is (input_dim, output_dim) — no transpose needed.
    Initialized as randn * (input_dim**-0.5) * init_scale.
    init_scale defaults to 1.0; set to 1/√(2·n_blocks) for
    residual output projections (W_out, ffn.linear2).
    """
    def __init__(self, input_dim, output_dim, device, generator,
                 bias=True, init_scale=1.0):
        self.weight = (
            torch.randn(input_dim, output_dim,
                        device=device, generator=generator)
            * (input_dim ** -0.5)
            * init_scale
        ).requires_grad_(True)

        self.bias = (
            torch.zeros(output_dim, device=device).requires_grad_(True)
            if bias else None
        )

    def __call__(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def config_dict(self):
        input_dim, output_dim = self.weight.shape
        return {"input_dim": input_dim, "output_dim": output_dim}


class GeLU:
    """
    Elementwise GeLU (GPT-2 approximation). No learnable parameters.
    Stores output on self.out for activation diagnostics.
    """
    def __call__(self, x):
        self.out = 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )
        return self.out

    def parameters(self):
        return []

    def config_dict(self):
        return {}


class Dropout:
    """
    Inverted dropout. Zeros elements with probability p during training,
    scales survivors by 1/(1-p). No-op during eval.

    flag is a shared mutable list (e.g. ["train"]) — the Transformer
    creates one and passes it to every Dropout so a single mutation
    flips all of them.
    """
    def __init__(self, p, flag):
        self.p = p
        self.flag = flag

    def __call__(self, x):
        if self.p == 1:
            return torch.zeros_like(x)
        if self.flag[0] == "train":
            self.mask = torch.rand_like(x) > self.p
            return (x * self.mask) / (1 - self.p)
        return x

    def parameters(self):
        return []

    def config_dict(self):
        return {}


class FeedForward:
    """
    Position-wise FFN: Linear → GeLU → Linear.
    (..., embedding_dim) → (..., embedding_dim).

    The second linear (the output projection) receives init_scale
    for residual stream variance control.
    """
    def __init__(self, embedding_dim, ffn_dim, device, generator,
                 init_scale=1.0):
        self.linear1 = Linear(embedding_dim, ffn_dim,
                              device=device, generator=generator)
        self.gelu = GeLU()
        self.linear2 = Linear(ffn_dim, embedding_dim,
                              device=device, generator=generator,
                              init_scale=init_scale)

    def __call__(self, x):
        return self.linear2(self.gelu(self.linear1(x)))

    def parameters(self):
        return (self.linear1.parameters()
                + self.gelu.parameters()
                + self.linear2.parameters())

    def config_dict(self):
        _, ffn_dim = self.linear1.weight.shape
        return {"ffn_dim": ffn_dim}


class LayerNorm:
    """
    Layer normalization over the last dimension.
    (B, T, d) → (B, T, d).

    Population variance (divides by d, not d-1).
    gamma=1, beta=0 at init (identity transform).
    """
    def __init__(self, embedding_dim, device, eps=1e-5):
        self.gamma = torch.ones(embedding_dim, device=device).requires_grad_(True)
        self.beta = torch.zeros(embedding_dim, device=device).requires_grad_(True)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / (var + self.eps) ** 0.5
        return self.gamma * x_hat + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

    def config_dict(self):
        return {}


# ═══════════════════════════════════════════════════════════════
#  ATTENTION
# ═══════════════════════════════════════════════════════════════

class MultiheadAttention:
    """
    Fused multi-head causal self-attention.

    Three fused projections (W_Q, W_K, W_V) produce all heads in one
    matmul each. W_out receives init_scale for residual scaling.

    Shape trace (B=batch, T=context_size, C=embedding_dim):
        x:        (B, T, C)
        Q, K:     → (B, num_heads, T, dim_qk)
        V:        → (B, num_heads, T, dim_v)
        scores:   → (B, num_heads, T, T)
        out:      → (B, T, C)
    """
    def __init__(self, context_size, embedding_dim, dim_qk, dim_v,
                 num_heads, device, generator,
                 p=0.0, flag=None, init_scale=1.0):
        self.W_Query = Linear(embedding_dim, num_heads * dim_qk,
                              device=device, generator=generator,
                              bias=False)
        self.W_Key = Linear(embedding_dim, num_heads * dim_qk,
                            device=device, generator=generator,
                            bias=False)
        self.W_Value = Linear(embedding_dim, num_heads * dim_v,
                              device=device, generator=generator,
                              bias=False)
        self.W_out = Linear(num_heads * dim_v, embedding_dim,
                            device=device, generator=generator,
                            bias=False, init_scale=init_scale)

        self.mask = torch.triu(
            torch.ones(1, 1, context_size, context_size, device=device),
            diagonal=1
        ).bool()

        self.attn_dropout = Dropout(p, flag) if flag is not None else None

        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.context_size = context_size

    def __call__(self, x):
        B = x.shape[0]
        T = self.context_size
        h = self.num_heads

        Q = self.W_Query(x).reshape(B, T, h, self.dim_qk).transpose(1, 2)
        K = self.W_Key(x).reshape(B, T, h, self.dim_qk).transpose(1, 2)
        V = self.W_Value(x).reshape(B, T, h, self.dim_v).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.dim_qk ** 0.5
        masked = scores.masked_fill(self.mask, float('-inf'))

        A = torch.nn.functional.softmax(masked, dim=-1)
        if self.attn_dropout is not None:
            A = self.attn_dropout(A)

        out = (A @ V).transpose(1, 2).reshape(B, T, h * self.dim_v)
        return self.W_out(out)

    def parameters(self):
        return (self.W_Query.parameters()
                + self.W_Key.parameters()
                + self.W_Value.parameters()
                + self.W_out.parameters())

    def config_dict(self):
        return {
            "dim_qk": self.dim_qk,
            "dim_v": self.dim_v,
            "num_heads": self.num_heads,
        }


# ═══════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK + MODEL
# ═══════════════════════════════════════════════════════════════

class Block:
    """
    Pre-norm transformer block:
        x = x + dropout(MultiheadAttention(LayerNorm(x)))
        x = x + dropout(FeedForward(LayerNorm(x)))

    (B, T, embedding_dim) → (B, T, embedding_dim).
    """
    def __init__(self, context_size, embedding_dim, dim_qk, dim_v,
                 num_heads, ffn_dim, device, generator,
                 p=0.0, flag=None, init_scale=1.0):
        self.multihead = MultiheadAttention(
            context_size=context_size, embedding_dim=embedding_dim,
            dim_qk=dim_qk, dim_v=dim_v, num_heads=num_heads,
            device=device, generator=generator,
            p=p, flag=flag, init_scale=init_scale,
        )
        self.ln1 = LayerNorm(embedding_dim, device=device)
        self.ffn = FeedForward(
            embedding_dim=embedding_dim, ffn_dim=ffn_dim,
            device=device, generator=generator,
            init_scale=init_scale,
        )
        self.ln2 = LayerNorm(embedding_dim, device=device)
        self.dropout = Dropout(p, flag) if flag is not None else None

    def __call__(self, x):
        attn_out = self.multihead(self.ln1(x))
        x = x + (self.dropout(attn_out) if self.dropout else attn_out)

        ffn_out = self.ffn(self.ln2(x))
        x = x + (self.dropout(ffn_out) if self.dropout else ffn_out)

        return x

    def parameters(self):
        return (self.multihead.parameters()
                + self.ln1.parameters()
                + self.ffn.parameters()
                + self.ln2.parameters())

    def config_dict(self):
        return {
            **self.multihead.config_dict(),
            "ffn_dim": self.ffn.config_dict().get("ffn_dim"),
        }


class Transformer:
    """
    Full transformer: embedding + positional encoding → N pre-norm
    blocks → final LayerNorm → weight-tied LM head.

    Residual output projections (W_out in attention, linear2 in FFN)
    are scaled by 1/√(2·n_blocks) at init to keep residual stream
    variance stable with depth.

    Embedding matrix is shared with the output projection (weight
    tying): logits = normalized @ embedding_matrix.T.

    Dropout (embedding, attention weights, residual on-ramps) is
    controlled by a shared mutable flag — call model.train() or
    model.eval() to flip all dropout layers at once.

    (B, T) token ids → (B, T, vocab_dim) logits.
    """
    def __init__(self, context_size, vocab_dim, embedding_dim, dim_qk,
                 dim_v, num_heads, ffn_dim, n_blocks, device, generator,
                 p=0.0):
        self.mode = ["train"]
        init_scale = (2 * n_blocks) ** -0.5

        self.embedding = Embedding(
            vocab_dim=vocab_dim, embedding_dim=embedding_dim,
            device=device, generator=generator,
        )
        self.positional_encoding = PositionalEncoding(
            context_size=context_size, embedding_dim=embedding_dim,
            device=device, generator=generator,
        )
        self.emb_dropout = Dropout(p=p, flag=self.mode)

        self.blocks = [
            Block(
                context_size=context_size, embedding_dim=embedding_dim,
                dim_qk=dim_qk, dim_v=dim_v, num_heads=num_heads,
                ffn_dim=ffn_dim, device=device, generator=generator,
                p=p, flag=self.mode, init_scale=init_scale,
            )
            for _ in range(n_blocks)
        ]

        self.ln = LayerNorm(embedding_dim=embedding_dim, device=device)

    def __call__(self, x):
        x = self.embedding(x) + self.positional_encoding()
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        logits = x @ self.embedding.embedding_matrix.T
        return logits

    def train(self):
        self.mode[0] = "train"

    def eval(self):
        self.mode[0] = "eval"

    def parameters(self):
        block_params = []
        for block in self.blocks:
            block_params += block.parameters()
        return (self.embedding.parameters()
                + self.positional_encoding.parameters()
                + block_params
                + self.ln.parameters())

    def config_dict(self):
        return {
            **self.embedding.config_dict(),
            **self.positional_encoding.config_dict(),
            **self.blocks[0].config_dict(),
            "n_blocks": len(self.blocks),
        }
