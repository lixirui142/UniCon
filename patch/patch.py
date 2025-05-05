import copy
from typing import Type, Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from peft.tuners.lora.layer import Linear, BaseTunerLayer
import pdb

logger = logging.get_logger(__name__) 

def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


def lora_forward_hack(self):
    """
    Hack forward function of LoRA layers.
    Let each adapter selectively applies to inputs specified by a mask.
    """
    def forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                lora_mask = self.lora_mask[active_adapter]
                lora_mask = lora_mask.repeat_interleave(x.shape[0] // len(lora_mask), dim=0)
                masked_x = x[lora_mask]

                if not self.use_dora[active_adapter]:
                    result_lora = lora_B(lora_A(dropout(masked_x))) * scaling
                else:
                    masked_x = dropout(masked_x)
                    result_lora = self._apply_dora(masked_x, lora_A, lora_B, scaling, active_adapter)

                result[lora_mask] += result_lora

            result = result.to(torch_result_dtype)
        return result
    return forward


def make_diffusers_unicon_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a class of UniCon blocks.
    It adds joint cross attention to the forward function and enables related functions for initialzation and training.
    """


    class UniConBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def set_joint_layer_requires_grad(self, adapter_names, requires_grad):
            
            for module in self.attn1n.modules():
                if not isinstance(module, BaseTunerLayer):
                    continue
                if isinstance(adapter_names, str):
                    adapter_names = [adapter_names]

                # Deactivate grads on the inactive adapter and activate grads on the active adapter
                for layer_name in module.adapter_layer_names:
                    module_dict = getattr(module, layer_name)
                    for key, layer in module_dict.items():
                        if key in adapter_names:
                            # Note: It is possible that not a single layer is called with requires_grad_(True) here. This may
                            # happen if a completely different adapter layer is being activated.
                            layer.requires_grad_(requires_grad)

            self.conv1n.requires_grad_(requires_grad)

        
        @property
        def post_joint(self):
            if self.post == "scale":
                return self.scale1n
            elif self.post == "conv" or self.post == "conv_fuse":
                return self.conv1n

        def add_post_joint(self, name, post = "conv", add_bias = False):
            if not hasattr(self, "post1n"):
                self.post1n = nn.ModuleDict({})
                self.post_type = dict()

            if name in self.post1n:
                return

            if post == "conv":
                conv_dim = self.attn1n.out_dim
            elif post == "conv_fuse":
                conv_dim = self.attn1n.out_dim * 2
            else:
                assert False, f"Unkown post processing type {post}"
            conv1n = nn.Linear(conv_dim, conv_dim, bias = add_bias)
            post_joint = zero_module(conv1n)
        
            self.post1n[name] = post_joint
            self.post_type[name] = post

        def initialize_joint_layers(self, post = "conv", add_bias = False):
            self.attn1n = copy.deepcopy(self.attn1)

            if post == "conv":
                conv_dim = self.attn1n.out_dim
            elif post == "conv_fuse":
                conv_dim = self.attn1n.out_dim * 2
            else:
                assert False, f"Unkown post processing type {post}"
            conv1n = nn.Linear(conv_dim, conv_dim, bias = add_bias)

            self.conv1n = zero_module(conv1n)
            self.post = post

            self.joint_scale = 1.0
            self.enable_joint_attention = True

        def set_joint_attention(self, enable = True):
            self.enable_joint_attention = enable

        def set_joint_scale(self, joint_scale = 1.0):
            self.joint_scale = joint_scale
        
        def post_proj(self, x_out, y_out, post_op, post_type):
            if post_type == "conv":
                xy_out = torch.cat([x_out, y_out], dim = 0)
                xy_post_out = post_op(xy_out)
                x_post_out, y_post_out = xy_post_out.chunk(2, dim = 0) 
            elif post_type == "conv_fuse":
                xy_out = torch.cat([x_out, y_out], dim = -1)
                xy_post_out = post_op(xy_out)
                x_post_out, y_post_out = xy_post_out.chunk(2, dim = -1)
            
            return x_post_out, y_post_out
            
        def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        ) -> torch.FloatTensor:
            if cross_attention_kwargs is not None:
                if cross_attention_kwargs.get("scale", None) is not None:
                    logger.warning("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")

            # Notice that normalization is always applied before the real computation in the following blocks.
            # 0. Self-Attention
            batch_size = hidden_states.shape[0]

            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1)
            else:
                raise ValueError("Incorrect norm used")

            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            # 1. Prepare GLIGEN inputs
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

            # Joint cross attention
            if self.enable_joint_attention:
                # Original self-attention
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )


                joint_norm_hidden_states = norm_hidden_states
                batch_size = joint_norm_hidden_states.shape[0]
                
                # Get paired joint cross attention inputs
                x_ids, y_ids, x_weights, y_weights = self._unicon_config["attn_config"]
                input_ids = torch.cat([x_ids, y_ids])
                joint_ids = torch.cat([y_ids, x_ids])
                input_norm_hidden_states = joint_norm_hidden_states[input_ids]
                joint_norm_hidden_states = joint_norm_hidden_states[joint_ids]

                # Attention
                attn_output1n = self.attn1n(
                    input_norm_hidden_states,
                    encoder_hidden_states=joint_norm_hidden_states,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )

                # Post projection
                output1n = torch.zeros_like(attn_output)

                attn_output1n_x, attn_output1n_y = attn_output1n.chunk(2, dim = 0)
                is_train = self._unicon_config["train"]
                if is_train:
                    attn_output1n_x, attn_output1n_y = self.post_proj(attn_output1n_x, attn_output1n_y, self.conv1n, self.post)
                else:
                    cond_masks = self._unicon_config["cond_masks"]
                    for cur_cond, cond_mask in cond_masks.items():
                        # cond_mask = [model_name == cur_cond for model_name in model_names]
                        x_post_out, y_post_out = self.post_proj(attn_output1n_x[cond_mask], attn_output1n_y[cond_mask], self.post1n[cur_cond], self.post_type[cur_cond])
                        attn_output1n_x[cond_mask], attn_output1n_y[cond_mask] = x_post_out, y_post_out
                
                # Aggregate output
                attn_output1n_x = attn_output1n_x * x_weights
                attn_output1n_y = attn_output1n_y * y_weights
                B, N, C = attn_output1n_x.shape
                x_indexes = x_ids.view(-1,1,1).expand(-1, N, C)
                y_indexes = y_ids.view(-1,1,1).expand(-1, N, C)
                torch.scatter_reduce(output1n, dim = 0, index=x_indexes, src=attn_output1n_x, reduce = "sum")
                torch.scatter_reduce(output1n, dim = 0, index=y_indexes, src=attn_output1n_y, reduce = "sum")
                attn_output = attn_output + output1n * self.joint_scale
            else:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )


            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output

            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])


            # 3. Cross-Attention
            if self.attn2 is not None:
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 4. Feed-forward
            # i2vgen doesn't have this norm 🤷‍♂️
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)

            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)

            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)

            return hidden_states

    return UniConBlock

def apply_patch(
        model: torch.nn.Module,
        train = False,
        name_skip = None):
    """
    Patches a diffusion model from diffusers with UniCon.

    Args:
     - model: A top level Stable Diffusion module to patch in place.
     - train: Whether to train the model.
     - name_skip: name for module you do not want to patch UniCon, e.g., name_skip="down_blocks" will skip the UNet encoder.

    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(
        model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    
    assert is_diffusers, "Only support diffusers model currently."

    diffusion_model = model.unet if hasattr(model, "unet") else model

    diffusion_model._unicon_config = {
        "train": train,
    }
    make_unicon_block_fn = make_diffusers_unicon_block

    for name, module in diffusion_model.named_modules():
        if name_skip is not None and name_skip in name:
            continue
        if isinstance_str(module, "BasicTransformerBlock"):
            module.__class__ = make_unicon_block_fn(module.__class__)
            module._unicon_config = diffusion_model._unicon_config

    return model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """


    model = model.unet if hasattr(model, "unet") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.__class__ = module._parent

    return model


def set_patch_lora_mask(model: torch.nn.Module, lora_name, lora_mask, kv_lora_mask = None):
    """ Set mask for LoRA layers"""

    model = model.unet if hasattr(model, "unet") else model
    qo_lora_mask = torch.tensor(lora_mask, dtype=torch.bool)
    kv_lora_mask = qo_lora_mask if kv_lora_mask is None else torch.tensor(kv_lora_mask, dtype=torch.bool)

    for name, module in model.named_modules():
        if isinstance(module, Linear) and lora_name in module.active_adapters:
            if not hasattr(module, "lora_mask"):
                module.lora_mask = dict()
            if "attn1n.to_k" in name or "attn1n.to_v" in name:
                module.lora_mask[lora_name] = kv_lora_mask
            else:
                module.lora_mask[lora_name] = qo_lora_mask
    return model

def set_joint_layer_requires_grad(model: torch.nn.Module, adapter_names, requires_grad):
    """ Set requires_grad for all unicon parameters """

    model = model.unet if hasattr(model, "unet") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.set_joint_layer_requires_grad(adapter_names, requires_grad)
    return model

def hack_lora_forward(model: torch.nn.Module):
    """ Replace the forward function of LoRA layers """

    model = model.unet if hasattr(model, "unet") else model
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            module.forward = lora_forward_hack(module)
    return model

def set_joint_attention(model: torch.nn.Module, enable = True, name_filter = None):
    """ Set whether to enable the joint cross attention """

    model = model.unet if hasattr(model, "unet") else model
    for name, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            if name_filter is None or name_filter in name:
                module.set_joint_attention(enable = enable)
    return model

def set_joint_scale(model: torch.nn.Module, scale = 1.0):
    """ Set the scale of joint cross attention """

    model = model.unet if hasattr(model, "unet") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.set_joint_scale(scale = scale)
    return model

def initialize_joint_layers(model: torch.nn.Module, post = "conv"):
    """ Initialize all joint cross attentions """

    model = model.unet if hasattr(model, "unet") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.initialize_joint_layers(post = post)
    return model

def add_post_joint(model: torch.nn.Module, name, post = "conv", add_bias = False, **kwargs):
    """ Add a post projection """

    model = model.unet if hasattr(model, "unet") else model
    for _, module in model.named_modules():
        if module.__class__.__name__ == "UniConBlock":
            module.add_post_joint(name, post, add_bias)
    return model

def set_unicon_config(model: torch.nn.Module, k, v):
    """ Update joint cross attention configurations in patched modules """

    model = model.unet if hasattr(model, "unet") else model
    model._unicon_config[k] = v
    return model

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    
    return False