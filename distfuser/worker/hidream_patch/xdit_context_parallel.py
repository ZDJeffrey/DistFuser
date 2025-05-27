# Adapted from https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/models/transformers/transformer_hidream_image.py
# and https://github.com/HiDream-ai/HiDream-I1/blob/main/hi_diffusers/models/attention_processor.py
import torch

from einops import repeat
from typing import Any, Dict, Optional, Tuple, List

from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group
from xfuser.core.long_ctx_attention import xFuserLongContextAttention, AttnType

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from distfuser.utils import init_logger

logger = init_logger(__name__)

try:
    from hi_diffusers.models.attention import HiDreamAttention
    from hi_diffusers.models.attention_processor import apply_rope
except ImportError:
    logger.info("hi_diffusers is not installed")


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, use_sage_attn: bool = False, sage_type: str = "fp8"):
    if use_sage_attn:
        if sage_type == "auto":
            fn = xFuserLongContextAttention(attn_type=AttnType.SAGE_AUTO)
        elif sage_type == "fp8":
            fn = xFuserLongContextAttention(attn_type=AttnType.SAGE_FP8)
        elif sage_type == "fp16":
            fn = xFuserLongContextAttention(attn_type=AttnType.SAGE_FP16)
        elif sage_type == "fp16_triton":
            fn = xFuserLongContextAttention(attn_type=AttnType.SAGE_FP16_TRITON)
        else:
            raise ValueError(f"Invalid sage_type: {sage_type}")
    else:
        fn = xFuserLongContextAttention()
    hidden_states = fn(None, query=query, key=key, value=value)

    hidden_states = hidden_states.flatten(-2)
    hidden_states = hidden_states.to(query.dtype)

    return hidden_states


class HiDreamAttnProcessor_usp:
    """Attention processor used USP attention"""

    def __init__(
        self,
        use_sage_attn: bool = False,
        sage_type: str = "fp8",
    ):
        self.use_sage_attn = use_sage_attn
        self.sage_type = sage_type

    def __call__(
        self,
        attn,
        image_tokens: torch.FloatTensor,
        image_tokens_masks: Optional[torch.FloatTensor] = None,
        text_tokens: Optional[torch.FloatTensor] = None,
        rope: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        dtype = image_tokens.dtype
        batch_size = image_tokens.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(image_tokens)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(image_tokens)).to(dtype=dtype)
        value_i = attn.to_v(image_tokens)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if image_tokens_masks is not None:
            key_i = key_i * image_tokens_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(text_tokens)).to(dtype=dtype)
            key_t = attn.k_rms_norm_t(attn.to_k_t(text_tokens)).to(dtype=dtype)
            value_t = attn.to_v_t(text_tokens)

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i

        if query.shape[-1] == rope.shape[-3] * 2:
            query, key = apply_rope(query, key, rope)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, rope)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        hidden_states = attention(query, key, value, self.use_sage_attn, self.sage_type)

        if not attn.single:
            hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states


def usp_dit_forward(
    self,
    hidden_states: torch.Tensor,
    timesteps: torch.LongTensor = None,
    encoder_hidden_states: torch.Tensor = None,
    pooled_embeds: torch.Tensor = None,
    img_sizes: Optional[List[Tuple[int, int]]] = None,
    img_ids: Optional[torch.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    def segment_hidden_states(hidden_states):
        rank = get_sequence_parallel_rank()
        world_size = get_sequence_parallel_world_size()

        hidden_states = torch.chunk(hidden_states, world_size, dim=1)[rank]

        return hidden_states

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.info("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

    # spatial forward
    batch_size = hidden_states.shape[0]
    hidden_states_type = hidden_states.dtype

    # 0. time
    timesteps = self.expand_timesteps(timesteps, batch_size, hidden_states.device)
    timesteps = self.t_embedder(timesteps, hidden_states_type)
    p_embedder = self.p_embedder(pooled_embeds)
    adaln_input = timesteps + p_embedder

    hidden_states, image_tokens_masks, img_sizes = self.patchify(hidden_states, self.max_seq, img_sizes)
    if image_tokens_masks is None:
        pH, pW = img_sizes[0]
        img_ids = torch.zeros(pH, pW, 3, device=hidden_states.device)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH, device=hidden_states.device)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW, device=hidden_states.device)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    hidden_states = self.x_embedder(hidden_states)

    T5_encoder_hidden_states = encoder_hidden_states[0]
    encoder_hidden_states = encoder_hidden_states[-1]
    encoder_hidden_states = [encoder_hidden_states[k] for k in self.llama_layers]

    if self.caption_projection is not None:
        new_encoder_hidden_states = []
        for i, enc_hidden_state in enumerate(encoder_hidden_states):
            enc_hidden_state = self.caption_projection[i](enc_hidden_state)
            enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
            new_encoder_hidden_states.append(enc_hidden_state)
        encoder_hidden_states = new_encoder_hidden_states
        T5_encoder_hidden_states = self.caption_projection[-1](T5_encoder_hidden_states)
        T5_encoder_hidden_states = T5_encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        encoder_hidden_states.append(T5_encoder_hidden_states)

    encoder_hidden_states = [segment_hidden_states(encoder_hidden_state) for encoder_hidden_state in encoder_hidden_states]

    txt_ids = torch.zeros(
        batch_size,
        encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
        3,
        device=img_ids.device,
        dtype=img_ids.dtype,
    )

    img_ids = segment_hidden_states(img_ids)
    hidden_states = segment_hidden_states(hidden_states)

    ids = torch.cat((img_ids, txt_ids), dim=1)
    rope = self.pe_embedder(ids)

    # 2. Blocks
    block_id = 0
    initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
    initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]

    for bid, block in enumerate(self.double_stream_blocks):
        cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
        cur_encoder_hidden_states = torch.cat([initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states, initial_encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                image_tokens_masks,
                cur_encoder_hidden_states,
                adaln_input,
                rope,
                **ckpt_kwargs,
            )
        else:
            hidden_states, initial_encoder_hidden_states = block(
                image_tokens=hidden_states,
                image_tokens_masks=image_tokens_masks,
                text_tokens=cur_encoder_hidden_states,
                adaln_input=adaln_input,
                rope=rope,
            )
        initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
        block_id += 1

    image_tokens_seq_len = hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
    hidden_states_seq_len = hidden_states.shape[1]
    if image_tokens_masks is not None:
        encoder_attention_mask_ones = torch.ones(
            (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
            device=image_tokens_masks.device,
            dtype=image_tokens_masks.dtype,
        )
        image_tokens_masks = torch.cat([image_tokens_masks, encoder_attention_mask_ones], dim=1)

    for bid, block in enumerate(self.single_stream_blocks):
        cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
        hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                image_tokens_masks,
                None,
                adaln_input,
                rope,
                **ckpt_kwargs,
            )
        else:
            hidden_states = block(
                image_tokens=hidden_states,
                image_tokens_masks=image_tokens_masks,
                text_tokens=None,
                adaln_input=adaln_input,
                rope=rope,
            )
        hidden_states = hidden_states[:, :hidden_states_seq_len]
        block_id += 1

    hidden_states = hidden_states[:, :image_tokens_seq_len, ...].contiguous()

    hidden_states = get_sp_group().all_gather(hidden_states, dim=1)
    image_tokens_seq_len = hidden_states.shape[1]

    output = self.final_layer(hidden_states, adaln_input)
    output = self.unpatchify(output, img_sizes, self.training)
    if image_tokens_masks is not None:
        image_tokens_masks = image_tokens_masks[:, :image_tokens_seq_len]

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output, image_tokens_masks)
    return Transformer2DModelOutput(sample=output, mask=image_tokens_masks)
