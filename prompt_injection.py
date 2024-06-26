#Modified/simplified version of the node from: https://github.com/pamparamm/sd-perturbed-attention
#If you want the one with more options see the above repo.

#My modified one here is more basic but has less chances of breaking with ComfyUI updates.

import comfy.model_patcher
import comfy.samplers
import torch
import torch.nn.functional as F

def build_patch(patchedBlocks, weight=1.0, sigma_start=0.0, sigma_end=1.0):
    def prompt_injection_patch(n, context_attn1: torch.Tensor, value_attn1, extra_options):
        (block, block_index) = extra_options.get('block', (None,None))
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9
        
        batch_prompt = n.shape[0] // len(extra_options["cond_or_uncond"])

        if sigma <= sigma_start and sigma >= sigma_end:
            if (block and f'{block}:{block_index}' in patchedBlocks and patchedBlocks[f'{block}:{block_index}']):
                if context_attn1.dim() == 3:
                    c = context_attn1[0].unsqueeze(0)
                else:
                    c = context_attn1[0][0].unsqueeze(0)
                b = patchedBlocks[f'{block}:{block_index}'][0][0].repeat(c.shape[0], 1, 1).to(context_attn1.device)
                out = torch.stack((c, b)).to(dtype=context_attn1.dtype) * weight
                out = out.repeat(1, batch_prompt, 1, 1) * weight

                return n, out, out 

        return n, context_attn1, value_attn1
    return prompt_injection_patch

def build_svd_patch(patchedBlocks, weight=1.0, sigma_start=0.0, sigma_end=1.0):
    def prompt_injection_patch(n, context_attn1: torch.Tensor, value_attn1, extra_options):
        (block, block_index) = extra_options.get('block', (None, None))
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        if sigma_start <= sigma <= sigma_end:
            if block and f'{block}:{block_index}' in patchedBlocks and patchedBlocks[f'{block}:{block_index}']:
                if context_attn1.dim() == 3:
                    c = context_attn1[0].unsqueeze(0)
                else:
                    c = context_attn1[0][0].unsqueeze(0)
                b = patchedBlocks[f'{block}:{block_index}'][0][0].repeat(c.shape[0], 1, 1).to(context_attn1.device)
                
                # Interpolate to match the sizes
                if c.size() != b.size():
                    b = F.interpolate(b.unsqueeze(0), size=c.size()[1:], mode='nearest').squeeze(0)
                
                out = torch.cat((c, b), dim=-1).to(dtype=context_attn1.dtype) * weight
                return n, out  # Ensure exactly two values are returned for SVD
        return n, context_attn1, value_attn1  # Ensure exactly three values are returned

    return prompt_injection_patch

class SVDPromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": ("MODEL",)},
            "optional": {
                "all": ("CONDITIONING",),
                "time_embed": ("CONDITIONING",),
                "label_emb": ("CONDITIONING",),
                "input_blocks_0": ("CONDITIONING",),
                "input_blocks_1": ("CONDITIONING",),
                "input_blocks_2": ("CONDITIONING",),
                "input_blocks_3": ("CONDITIONING",),
                "input_blocks_4": ("CONDITIONING",),
                "input_blocks_5": ("CONDITIONING",),
                "input_blocks_6": ("CONDITIONING",),
                "input_blocks_7": ("CONDITIONING",),
                "input_blocks_8": ("CONDITIONING",),
                "middle_block_0": ("CONDITIONING",),
                "middle_block_1": ("CONDITIONING",),
                "middle_block_2": ("CONDITIONING",),
                "output_blocks_0": ("CONDITIONING",),
                "output_blocks_1": ("CONDITIONING",),
                "output_blocks_2": ("CONDITIONING",),
                "output_blocks_3": ("CONDITIONING",),
                "output_blocks_4": ("CONDITIONING",),
                "output_blocks_5": ("CONDITIONING",),
                "output_blocks_6": ("CONDITIONING",),
                "output_blocks_7": ("CONDITIONING",),
                "output_blocks_8": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, all=None, time_embed=None, label_emb=None, input_blocks_0=None, input_blocks_1=None, input_blocks_2=None, input_blocks_3=None, input_blocks_4=None, input_blocks_5=None, input_blocks_6=None, input_blocks_7=None, input_blocks_8=None, middle_block_0=None, middle_block_1=None, middle_block_2=None, output_blocks_0=None, output_blocks_1=None, output_blocks_2=None, output_blocks_3=None, output_blocks_4=None, output_blocks_5=None, output_blocks_6=None, output_blocks_7=None, output_blocks_8=None, weight=1.0, start_at=0.0, end_at=1.0):
        if not any((all, time_embed, label_emb, input_blocks_0, input_blocks_1, input_blocks_2, input_blocks_3, input_blocks_4, input_blocks_5, input_blocks_6, input_blocks_7, input_blocks_8, middle_block_0, middle_block_1, middle_block_2, output_blocks_0, output_blocks_1, output_blocks_2, output_blocks_3, output_blocks_4, output_blocks_5, output_blocks_6, output_blocks_7, output_blocks_8)):
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        patchedBlocks = {}
        blocks = {
            'time_embed': [0],
            'label_emb': [0],
            'input_blocks': list(range(9)),
            'middle_block': list(range(3)),
            'output_blocks': list(range(9))
        }

        for block in blocks:
            for index in blocks[block]:
                block_name = f"{block}_{index}"
                value = locals().get(block_name, None)
                if value is None:
                    value = all
                if value is not None:
                    patchedBlocks[f"{block}:{index}"] = value

        m.set_model_attn2_patch(build_svd_patch(patchedBlocks, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class PromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "all":  ("CONDITIONING",),
                "input_4":  ("CONDITIONING",),
                "input_5":  ("CONDITIONING",),
                "input_7":  ("CONDITIONING",),
                "input_8":  ("CONDITIONING",),
                "middle_0": ("CONDITIONING",),
                "output_0": ("CONDITIONING",),
                "output_1": ("CONDITIONING",),
                "output_2": ("CONDITIONING",),
                "output_3": ("CONDITIONING",),
                "output_4": ("CONDITIONING",),
                "output_5": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, all=None, input_4=None, input_5=None, input_7=None, input_8=None, middle_0=None, output_0=None, output_1=None, output_2=None, output_3=None, output_4=None, output_5=None, weight=1.0, start_at=0.0, end_at=1.0):
        if not any((all, input_4, input_5, input_7, input_8, middle_0, output_0, output_1, output_2, output_3, output_4, output_5)):
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        patchedBlocks = {}
        blocks = {'input': [4, 5, 7, 8], 'middle': [0], 'output': [0, 1, 2, 3, 4, 5]}

        for block in blocks:
            for index in blocks[block]:
                value = locals()[f"{block}_{index}"] if locals()[f"{block}_{index}"] is not None else all
                if value is not None:
                    patchedBlocks[f"{block}:{index}"] = value

        m.set_model_attn2_patch(build_patch(patchedBlocks, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class SimplePromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "block": (["input:4", "input:5", "input:7", "input:8", "middle:0", "output:0", "output:1", "output:2", "output:3", "output:4", "output:5"],),
                "conditioning": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, block, conditioning=None, weight=1.0, start_at=0.0, end_at=1.0):
        if conditioning is None:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        m.set_model_attn2_patch(build_patch({f"{block}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class SimplePromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "block": (["input:4", "input:5", "input:7", "input:8", "middle:0", "output:0", "output:1", "output:2", "output:3", "output:4", "output:5"],),
                "conditioning": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, block, conditioning=None, weight=1.0, start_at=0.0, end_at=1.0):
        if conditioning is None:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        m.set_model_attn2_patch(build_patch({f"{block}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class AdvancedPromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "locations": ("STRING", {"multiline": True, "default": "output:0,1.0\noutput:1,1.0"}),
                "conditioning": ("CONDITIONING",),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "advanced/model"

    def patch(self, model: comfy.model_patcher.ModelPatcher, locations: str, conditioning=None, start_at=0.0, end_at=1.0):
        if not conditioning:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        for line in locations.splitlines():
            line = line.strip().strip('\n')
            weight = 1.0
            if ',' in line:
                line, weight = line.split(',')
                line = line.strip()
                weight = float(weight)
            if line:
                m.set_model_attn2_patch(build_patch({f"{line}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)


NODE_CLASS_MAPPINGS = {
    "PromptInjection": PromptInjection,
    "SimplePromptInjection": SimplePromptInjection,
    "AdvancedPromptInjection": AdvancedPromptInjection,
    "SVDPromptInjection": SVDPromptInjection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptInjection": "Attn2 Prompt Injection",
    "SimplePromptInjection": "Attn2 Prompt Injection (simple)",
    "AdvancedPromptInjection": "Attn2 Prompt Injection (advanced)",
    "SVDPromptInjection": "Attn2 SVD Prompt Injection"
}
