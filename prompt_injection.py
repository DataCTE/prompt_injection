#Modified/simplified version of the node from: https://github.com/pamparamm/sd-perturbed-attention
#If you want the one with more options see the above repo.

#My modified one here is more basic but has less chances of breaking with ComfyUI updates.

import comfy.model_patcher
import comfy.samplers
import torch

def build_patch(patchedBlocks):
    def prompt_injection_patch(n, context_attn1: torch.Tensor, value_attn1, extra_options):
        (block, block_index) = extra_options.get('block', (None,None)) 
        
        if (block and f'{block}:{block_index}' in patchedBlocks and patchedBlocks[f'{block}:{block_index}']):
            cond = torch.stack(
                (
                    context_attn1[0].unsqueeze(0),
                    patchedBlocks[f'{block}:{block_index}'][0][0].to(context_attn1.device)
                )
            ).to(dtype=context_attn1.dtype)
            return n, cond, cond
        
        return n, context_attn1, value_attn1
    return prompt_injection_patch

class PromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "in4":  ("CONDITIONING",),
                "in5":  ("CONDITIONING",),
                "in7":  ("CONDITIONING",),
                "in8":  ("CONDITIONING",),
                "mid0": ("CONDITIONING",),
                "out0": ("CONDITIONING",),
                "out1": ("CONDITIONING",),
                "out2": ("CONDITIONING",),
                "out3": ("CONDITIONING",),
                "out4": ("CONDITIONING",),
                "out5": ("CONDITIONING",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, in4=None, in5=None, in7=None, in8=None, mid0=None, out0=None, out1=None, out2=None, out3=None, out4=None, out5=None):
        m = model.clone()
        if any((in4, in5, in7, in8, mid0, out0, out1, out2, out3, out4, out5)):
            patchedBlocks = {
                'input:4': in4,
                'input:5': in5,
                'input:7': in7,
                'input:8': in8,
                'middle:0': mid0,
                'output:0': out0,
                'output:1': out1,
                'output:2': out2,
                'output:3': out3,
                'output:4': out4,
                'output:5': out5,
            }
            m.set_model_attn2_patch(build_patch(patchedBlocks))

        return (m,)

class SimplePromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "block": (("output", "middle", "input"),),
                "index": ("INT", {"default": 0, "min": 0, "max": 7, "step": 1}),
                "conditioning": ("CONDITIONING",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, block, index, conditioning):
        m = model.clone()
        m.set_model_attn2_patch(build_patch({f"{block}:{index}": conditioning}))

        return (m,)

class AdvancedPromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "locations": ("STRING", {"multiline": True, "default": "output:0\noutput:1"}),
                "conditioning": ("CONDITIONING",)
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, locations: str, conditioning):
        m = model.clone()
        patched_blocks = {}
        for line in locations.splitlines():
            patched_blocks[line.strip().strip('\n')] = conditioning
        m.set_model_attn2_patch(build_patch(patched_blocks))

        return (m,)

NODE_CLASS_MAPPINGS = {
    "PromptInjection": PromptInjection,
    "SimplePromptInjection": SimplePromptInjection,
    "AdvancedPromptInjection": AdvancedPromptInjection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptInjection": "Inject Prompt in Attention",
    "SimplePromptInjection": "Inject Prompt in Attention (simple)",
    "AdvancedPromptInjection": "Inject Prompt in Attention (advanced)"
}