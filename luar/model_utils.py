from .layers import MemoryEfficientAttention
from .scaled_attention import CustomAttention
from copy import deepcopy


def change_attention_to_FlashAttention(model):
  for name, module in model.named_modules():
    if name and name != 'linear':
      set_attribute_recursively(model, name, replace_attention_with_FlashAttention(module))

def replace_attention_with_FlashAttention(module, dim = 768, heads = 12, dim_head = 64):
  if isinstance(module, MemoryEfficientAttention):
    new_module = CustomAttention(dim = dim, heads = heads, dim_head = dim_head)
    target_state_dict = deepcopy(module.state_dict())
    new_module.load_state_dict(target_state_dict)
    return new_module
  else:
    return module

def set_attribute_recursively(obj, attribute_name, value):
  attributes_split = attribute_name.split('.', 1)
  if len(attributes_split) == 1:
    setattr(obj, attributes_split[0], value)
  else:
    set_attribute_recursively(getattr(obj, attributes_split[0]), attributes_split[1], value)
