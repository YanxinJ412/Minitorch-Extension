import builtins
import math
import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .kv_cache import KVCache, LayerKVCache, PagedKVCache, PagedLayerKVCache
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple, Union

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False, use_paged_attention: bool=False, use_flash_attention: bool=False):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd    : Dimensionality of embeddings and hidden states
            n_head    : Number of heads
            p_dropout : Dropout ratio for dropout layer
            causal    : If True, then apply a causal mask during self-attention
            bias      : If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection   : Linear layer projecting input to Q matrix
            k_projection   : Linear layer projecting input to K matrix
            v_project      : Linear layer projecting input to V matrix
            out_projection : Linear output projection layer
            dropout        : Dropout layer
        """
        self.backend   = backend
        self.n_embd    = n_embd 
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head
        self.use_fused_kernel = use_fused_kernel
        self.use_paged_attention = use_paged_attention
        self.use_flash_attention = use_flash_attention

        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        self.bias = bias
        self.q_projection = Linear(n_embd, n_embd, self.bias, backend)
        self.k_projection = Linear(n_embd, n_embd, self.bias, backend)
        self.v_projection = Linear(n_embd, n_embd, self.bias, backend)
        self.out_projection = Linear(n_embd, n_embd, self.bias, backend)
        self.dropout = Dropout(p_dropout)

    def create_causal_mask(
        self,
        bs,
        nh,
        query_len,
        key_len=None,
        past_len: int=0,
        query_positions=None,
        key_positions=None,
    ):
        """
        Return a causal mask for attention scores of shape (bs, nh, query_len, key_len).
        """
        if query_positions is None:
            query_positions = past_len + np.arange(query_len, dtype=np.int64)
        else:
            query_positions = np.asarray(query_positions, dtype=np.int64)
        if key_positions is None:
            if key_len is None:
                key_len = query_len
            key_positions = np.arange(key_len, dtype=np.int64)
        else:
            key_positions = np.asarray(key_positions, dtype=np.int64)
        invalid = key_positions[None, :] > query_positions[:, None]
        mask = -np.finfo(datatype).max * invalid.astype(datatype)
        mask = np.broadcast_to(mask, (bs, nh, query_positions.shape[0], key_positions.shape[0])).copy()
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """Project x to Q, transpose of K, V for self attention
        
        Args:
            x: embeddings or hidden states (batch_size x seq_len x n_embd)

        Returns:
            Q : The Query Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            K : The Key Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            V : The Value Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        x_2d = x.view(batch_size * seq_len, n_embd)
        q = self.q_projection(x_2d).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        k = self.k_projection(x_2d).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        v = self.v_projection(x_2d).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        return q, k, v

    def self_attention(
        self,
        q,
        k,
        v,
        layer_cache: Optional[Union[LayerKVCache, PagedLayerKVCache]]=None,
        use_cache: bool=False,
        query_positions=None,
    ):
        """Given q, kT, and v of sizes defined above, return the result of MultiHeadAttention as described in the writeup
        softmax((q @ kT) / sqrt(attn_hidden_dim)) @ V.
        NOTE: We have added support for Batch Matrix Multiplication with 4 dimensions.
        This means given tensors A of shape (a, b, m, n) and B of shape (a, b, n, p), 
        A @ B will be of the shape (a, b, m, p). Take a moment to consider why we need it.

        Args:
            q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            k  : Keys Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, _, k_dim = k.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        # result = None
        past_len = 0
        key_positions = None
        if query_positions is not None:
            query_positions = np.asarray(query_positions, dtype=np.int64)

        if use_cache:
            if layer_cache is None:
                raise ValueError("use_cache=True requires a layer cache")
            past_len = layer_cache.seq_len
            if query_positions is None:
                query_positions = np.arange(past_len, past_len + queries_len, dtype=np.int64)
            layer_cache.append(k, v, query_positions)
            key_positions = layer_cache.positions
            key_len = layer_cache.seq_len
        else:
            key_len = k.shape[2]

        use_fused_decode = (
            self.use_fused_kernel
            and use_cache
            and layer_cache is not None
            and queries_len == 1
            and not self.dropout.training
            and getattr(self.backend, "supports_fused_decode_attn", False)
        )
        fused_buf = (
            layer_cache.fused_decode_buffers()
            if use_fused_decode and layer_cache is not None
            else None
        )

        use_flash_decode = (
            self.use_flash_attention
            and use_cache
            and layer_cache is not None
            and queries_len == 1
            and not self.dropout.training
            and getattr(self.backend, "supports_flash_decode_attn", False)
        )
        flash_buf = None
        if use_flash_decode and layer_cache is not None:
            maybe_buf = layer_cache.fused_decode_buffers()
            if maybe_buf is not None and maybe_buf[0] == "fp32":
                flash_buf = maybe_buf

        use_flash_full = (
            self.use_flash_attention
            and (not use_cache)
            and self.causal
            and (not self.dropout.training)
            and getattr(self.backend, "supports_flash_attn", False)
        )

        use_paged_decode = (
            self.use_paged_attention
            and use_cache
            and layer_cache is not None
            and queries_len == 1
            and not self.dropout.training
            and getattr(self.backend, "supports_paged_decode_attn", False)
            and hasattr(layer_cache, "paged_decode_buffers")
        )
        paged_buf = (
            layer_cache.paged_decode_buffers()
            if use_paged_decode and layer_cache is not None
            else None
        )

        if flash_buf is not None:
            _mode, kn, vn, _ks, _vs, _seq_len = flash_buf
            mask_t = self.create_causal_mask(
                batch_size,
                num_head,
                queries_len,
                key_len,
                past_len=past_len,
                query_positions=query_positions,
                key_positions=key_positions,
            )
            attn_out = self.backend.flash_decode_attn_fw(
                q.contiguous(),
                mask_t.contiguous(),
                kn,
                vn,
                q_dim,
            )
            result = attn_out.permute(0, 2, 1, 3).contiguous()
            result_2d = result.view(batch_size * queries_len, self.n_embd)
            result_2d = self.out_projection(result_2d)
            result = result_2d.view(batch_size, queries_len, self.n_embd)
            return result

        if use_flash_full:
            mask_t = self.create_causal_mask(
                batch_size,
                num_head,
                queries_len,
                key_len,
                past_len=0,
                query_positions=query_positions,
                key_positions=key_positions,
            )
            attn_out = self.backend.flash_attn_fw(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                mask_t.contiguous(),
                q_dim,
            )
            result = attn_out.permute(0, 2, 1, 3).contiguous()
            result_2d = result.view(batch_size * queries_len, self.n_embd)
            result_2d = self.out_projection(result_2d)
            result = result_2d.view(batch_size, queries_len, self.n_embd)
            return result

        if paged_buf is not None:
            mode, kn, vn, page_offsets, k_scales, v_scales, total_seq, page_size = paged_buf
            mask_t = self.create_causal_mask(
                batch_size,
                num_head,
                queries_len,
                total_seq,
                past_len=past_len,
                query_positions=query_positions,
                key_positions=key_positions,
            )
            attn_out = self.backend.paged_decode_attn_fw(
                q.contiguous(),
                mask_t.contiguous(),
                mode,
                kn,
                vn,
                page_offsets,
                k_scales,
                v_scales,
                q_dim,
                page_size=page_size,
            )
            result = attn_out.permute(0, 2, 1, 3).contiguous()
            result_2d = result.view(batch_size * queries_len, self.n_embd)
            result_2d = self.out_projection(result_2d)
            result = result_2d.view(batch_size, queries_len, self.n_embd)
            return result

        if fused_buf is not None:
            # fused_buf may include seq_len for packed formats (e.g. int4)
            if len(fused_buf) == 5:
                mode, kn, vn, ks, vs = fused_buf
                seq_len_override = None
            else:
                mode, kn, vn, ks, vs, seq_len_override = fused_buf
            mask_t = self.create_causal_mask(
                batch_size,
                num_head,
                queries_len,
                key_len,
                past_len=past_len,
                query_positions=query_positions,
                key_positions=key_positions,
            )
            attn_out = self.backend.fused_decode_attn_fw(
                q.contiguous(),
                mask_t.contiguous(),
                mode,
                kn,
                vn,
                ks,
                vs,
                q_dim,
                seq_len_override=seq_len_override,
            )
            result = attn_out.permute(0, 2, 1, 3).contiguous()
            result_2d = result.view(batch_size * queries_len, self.n_embd)
            result_2d = self.out_projection(result_2d)
            result = result_2d.view(batch_size, queries_len, self.n_embd)
            return result

        if use_cache:
            k = layer_cache.key
            v = layer_cache.value

        kT = k.permute(0, 1, 3, 2)
        key_len = k.shape[2]
        
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # raise NotImplementedError
            attn = (q @ kT) / math.sqrt(q_dim)
            if self.causal:
                attn = attn + self.create_causal_mask(
                    batch_size,
                    num_head,
                    queries_len,
                    key_len,
                    past_len=past_len,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
            attn = softmax(attn, dim=3)
            attn = self.dropout(attn)
            result = attn @ v
            result = result.permute(0, 2, 1, 3).contiguous()
            result_2d = result.view(batch_size * queries_len, self.n_embd)
            result_2d = self.out_projection(result_2d)
            result = result_2d.view(batch_size, queries_len, self.n_embd)
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            attn = (q @ kT) / math.sqrt(q_dim)
            if self.causal:
                attn = attn + self.create_causal_mask(
                    batch_size,
                    num_head,
                    queries_len,
                    key_len,
                    past_len=past_len,
                    query_positions=query_positions,
                    key_positions=key_positions,
                )
            # NOTE: the provided attn_softmax CUDA kernel expects a different mask layout
            # and can silently compute incorrect results with the 4D causal mask used here.
            # Keep correctness by using the reference softmax while still allowing our
            # fused decode attention path above to accelerate single-token KV-cache decoding.
            attn = softmax(attn, dim=3)
            attn = self.dropout(attn)
            result = attn @ v
            result = result.permute(0, 2, 1, 3).contiguous()
            result_2d = result.view(batch_size * queries_len, self.n_embd)
            result_2d = self.out_projection(result_2d)
            result = result_2d.view(batch_size, queries_len, self.n_embd)
            # END ASSIGN3_3

        return result

    def forward(self, x, layer_cache: Optional[Union[LayerKVCache, PagedLayerKVCache]]=None, use_cache: bool=False, query_positions=None):
        """Computes MultiHeadAttention with causal masking if needed. 

        Args:
            x : Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        q, k, v = self.project_to_query_key_value(x)
        return self.self_attention(
            q,
            k,
            v,
            layer_cache=layer_cache,
            use_cache=use_cache,
            query_positions=query_positions,
        )


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """The Feed Forward Module.
        
        Args:
            n_embd     : in_size of first linear layer and out_size of last linear layer
            middle_dim : out_size of first linear layer and in_size of last linear layer
            p_dropout  : Dropout probability
            bias       : If bias should be applied in linear layers
        
        Attributes:
            linear_in  : first linear layer
            linear_out : second linear layer
            dropout    : dropout layer
        """
        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        """A FFN Module in a Pre-LN Transformer with GELU Activation and dropout.

        Args:
            x : Tensor of shape (batch_size x seq_len x n_embd)

        Returns:
            output : Tensor of shape (batch_size x seq_len x n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)

        return x

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-8, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False, use_paged_attention: bool=False, use_flash_attention: bool=False):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args: 
            n_embd : Dimensionality of embeddings and hidden states
            n_head : Number of heads for MultiHeadAttention
            p_dropout : Dropout ratio for dropout layer
            ln_eps : A value added for numerical stability in LayerNorm
            bias : If bias should be added in linear layers
        
        Attributes:
            ln_1 : First LayerNorm1d layer before MultiHeadAttention
            ln_2 : Second LayerNorm1d layer after MultiHeadAttention
            attention : MultiHeadAttention layer
            ff : FeedForward layer
        """
        
        # COPY FROM ASSIGN2_4
        # self.attention
        # self.ff
        # raise NotImplementedError
        self.attention = MultiHeadAttention(n_embd, n_head, True, p_dropout, bias, backend, use_fused_kernel=use_fused_kernel, use_paged_attention=use_paged_attention, use_flash_attention=use_flash_attention)
        self.ff = FeedForward(n_embd, p_dropout=p_dropout, bias=bias, backend=backend)

        self.use_fused_kernel = use_fused_kernel
        self.use_paged_attention = use_paged_attention
        self.use_flash_attention = use_flash_attention
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # self.ln_1
            # self.ln_2
            self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend)
            self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend)
            # raise NotImplementedError
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend)
            self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend)
            # END ASSIGN3_3

    def forward(self, x, layer_cache: Optional[Union[LayerKVCache, PagedLayerKVCache]]=None, use_cache: bool=False, query_positions=None):
        """
        The forward function of a Transformer Layer for a PRENORM Transformer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """
        batch_size, seq_len, x_dim = x.shape
        
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # raise NotImplementedError
            x_2d = x.view(batch_size * seq_len, x_dim)
            x_1norm = self.ln_1(x_2d).view(batch_size, seq_len, x_dim)
            x_1attn = self.attention(
                x_1norm,
                layer_cache=layer_cache,
                use_cache=use_cache,
                query_positions=query_positions,
            )
            x_1sum = x_2d + x_1attn.view(batch_size * seq_len, x_dim)
            x_2norm = self.ln_2(x_1sum).view(batch_size, seq_len, x_dim)
            x_2ff = self.ff(x_2norm).view(batch_size * seq_len, x_dim)
            x_2sum = x_1sum + x_2ff
            x_out = x_2sum.view(batch_size, seq_len, x_dim)
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            x_2d = x.view(batch_size * seq_len, x_dim)
            x_1norm_2d = x_2d.layernorm(self.ln_1.weights.value, self.ln_1.bias.value)
            x_1norm = x_1norm_2d.view(batch_size, seq_len, x_dim)
            x_1attn = self.attention(
                x_1norm,
                layer_cache=layer_cache,
                use_cache=use_cache,
                query_positions=query_positions,
            )
            x_1sum = x_2d + x_1attn.view(batch_size * seq_len, x_dim)
            x_2norm_2d = x_1sum.layernorm(self.ln_2.weights.value, self.ln_2.bias.value)
            x_2norm = x_2norm_2d.view(batch_size, seq_len, x_dim)
            x_2ff = self.ff(x_2norm).view(batch_size * seq_len, x_dim)
            x_2sum = x_1sum + x_2ff
            x_out = x_2sum.view(batch_size, seq_len, x_dim)
            # END ASSIGN3_3

        return x_out


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        n_layer: int=4,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False,
        use_paged_attention: bool=False,
        use_flash_attention: bool=False,
        kv_cache_page_size: int=64,
    ):
        super().__init__()
        """A Full Decoder-only Pre-LN Transformer with 4 Transformer Layers.

        Args:
            n_vocab : Vocabulary size defines the number of different tokens that can be represented by the input.
            n_embd  :  Dimensionality of the embeddings and hidden states.
            n_head  : Number of attention heads for each attention layer in the Transformer.
            n_positions : The maximum sequence length that this model might ever be used with.
            p_dropout : The dropout ratio for any dropout layer.
            ln_eps : The epsilon to use in the layer normalization layers.
            bias : If linear layers should include a bias.
        
        Attributes:
            token_embeddings : Embedding layer for tokens.
            position_embeddings : Embedding layer for token positions.
            t_layer_1 : 1st Transformer Layer.
            t_layer_2 : 2nd Transformer Layer.
            t_layer_3 : 3rd Transformer Layer.
            t_layer_4 : 4th Transformer Layer.
            dropout : Dropout layer before first transformer layer.
            ln : LayerNorm layer after last transformer layer.
            lm_head : Linear layer for projection from (*, n_embd) to (*, n_vocab)
        """
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_head              = n_head
        self.n_positions         = n_positions
        self.n_vocab             = n_vocab
        self.n_layer             = n_layer
        if n_layer < 1 or n_layer > 4:
            raise ValueError("DecoderLM supports between 1 and 4 transformer layers")
        
        # COPY FROM ASSIGN2_4
        # self.token_embeddings    = 
        # self.position_embeddings = 
        # self.t_layer_1           = 
        # self.t_layer_2           = 
        # self.t_layer_3           = 
        # self.t_layer_4           = 
        # self.dropout             = 
        # self.lm_head             = 
        # raise NotImplementedError
        self.token_embeddings = Embedding(n_vocab, n_embd, backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend)
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel=use_fused_kernel, use_paged_attention=use_paged_attention, use_flash_attention=use_flash_attention)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel=use_fused_kernel, use_paged_attention=use_paged_attention, use_flash_attention=use_flash_attention)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel=use_fused_kernel, use_paged_attention=use_paged_attention, use_flash_attention=use_flash_attention)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend, use_fused_kernel=use_fused_kernel, use_paged_attention=use_paged_attention, use_flash_attention=use_flash_attention)
        self.layers = [
            self.t_layer_1,
            self.t_layer_2,
            self.t_layer_3,
            self.t_layer_4,
        ][:n_layer]
        self.dropout = Dropout(p_dropout)
        self.lm_head = Linear(n_embd, n_vocab, bias, backend)

        self.use_fused_kernel = use_fused_kernel
        self.use_paged_attention = use_paged_attention
        self.use_flash_attention = use_flash_attention
        self.kv_cache_page_size = builtins.max(int(kv_cache_page_size), 1)
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # self.ln                  = 
            # raise NotImplementedError
            self.ln = LayerNorm1d(n_embd, ln_eps, backend)
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            self.ln = LayerNorm1d(n_embd, ln_eps, backend)
            # END ASSIGN3_3
        
    def init_kv_cache(
        self,
        quantization=None,
        max_cache_bytes: Optional[int]=None,
    ):
        if self.use_paged_attention:
            return PagedKVCache(
                n_layers=self.n_layer,
                backend=self.backend,
                quantization=quantization,
                max_cache_bytes=max_cache_bytes,
                page_size=self.kv_cache_page_size,
            )
        return KVCache(
            n_layers=self.n_layer,
            backend=self.backend,
            quantization=quantization,
            max_cache_bytes=max_cache_bytes,
        )

    def forward(
        self,
        idx,
        kv_cache: Optional[Union[KVCache, PagedKVCache]]=None,
        use_cache: bool=False,
        kv_cache_quantization: Optional[str]=None,
        kv_cache_max_bytes: Optional[int]=None,
    ):
        """A Forward pass of a Decoder-only Transformer Language model.
        Args: 
            idx: input of shape (batch_size, seq_len)
        
        Returns: 
            logits: logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape
        if use_cache and kv_cache is None:
            kv_cache = self.init_kv_cache(
                quantization=kv_cache_quantization,
                max_cache_bytes=kv_cache_max_bytes,
            )

        past_len = 0 if kv_cache is None else kv_cache.tokens_seen
        token_positions = np.arange(past_len, past_len + seq_len, dtype=np.int64)
        pos = tensor(token_positions.tolist(), backend=self.backend).view(1, seq_len)

        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # raise NotImplementedError
            taken_embd = self.token_embeddings(idx)
            pos_embd = self.position_embeddings(pos).view(1, seq_len, self.n_embd)
            x_embd = taken_embd + pos_embd
            x_embd = self.dropout(x_embd)
            for layer_idx, layer in enumerate(self.layers):
                layer_cache = None if kv_cache is None else kv_cache[layer_idx]
                x_embd = layer(
                    x_embd,
                    layer_cache=layer_cache,
                    use_cache=use_cache,
                    query_positions=token_positions,
                )
            x_embd = x_embd.view(batch_size * seq_len, self.n_embd)
            x_embd = self.ln(x_embd)
            x_embd = self.lm_head(x_embd)
            x_embd = x_embd.view(batch_size, seq_len, self.n_vocab)
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            taken_embd = self.token_embeddings(idx)
            pos_embd = self.position_embeddings(pos).view(1, seq_len, self.n_embd)
            x_embd = taken_embd + pos_embd
            x_embd = self.dropout(x_embd)
            for layer_idx, layer in enumerate(self.layers):
                layer_cache = None if kv_cache is None else kv_cache[layer_idx]
                x_embd = layer(
                    x_embd,
                    layer_cache=layer_cache,
                    use_cache=use_cache,
                    query_positions=token_positions,
                )
            x_embd = x_embd.view(batch_size * seq_len, self.n_embd)
            x_embd = x_embd.layernorm(self.ln.weights.value, self.ln.bias.value)
            x_embd = self.lm_head(x_embd)
            x_embd = x_embd.view(batch_size, seq_len, self.n_vocab)
            # END ASSIGN3_3

        if use_cache:
            kv_cache.record_tokens(seq_len)
            kv_cache.enforce_budget()
            return x_embd, kv_cache
        return x_embd
