## Custom attention implementation with FlashAttention
import torch
import torch.nn.functional as F

class CustomAttention(torch.nn.Module):
  def __init__(self, *, dim = 768, heads = 12, dim_head = 64):
    super().__init__()
    self.heads = heads
    inner_dim = dim_head * heads
    self.key = torch.nn.Linear(dim, inner_dim)
    self.query = torch.nn.Linear(dim, inner_dim)
    self.value = torch.nn.Linear(dim, inner_dim)


  def forward(
        self,
        hidden_states,
        attention_mask=None,
        # the following parameters are expected by the HuggingFace
        # implementation of Attention but not used here:
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
    # import pdb; pdb.set_trace()
    h = self.heads
    B, T, C = hidden_states.shape
    k = self.key(hidden_states)
    q = self.query(hidden_states)
    v = self.value(hidden_states)
    q = q.view(B, T, h, -1).permute(0, 2, 1, 3)
    k = k.view(B, T, h, -1).permute(0, 2, 1, 3)
    v = v.view(B, T, h, -1).permute(0, 2, 1, 3)
    attention_mask = attention_mask.view(attention_mask.shape[0], 1, attention_mask.shape[-1], 1)
    # attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    #         # scaled_dot_product_attention expects attention_mask shape to be
    #         # (batch, heads, source_length, target_length)
    # attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    # print(q.shape, k.shape, v.shape)
    out = F.scaled_dot_product_attention(q, k, v)
    out = out.transpose(1, 2).contiguous().view(B, T, -1)
    return (out,)