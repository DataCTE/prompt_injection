# Prompt Injection Node for ComfyUI

This custom node for ComfyUI allows you to inject specific prompts at specific blocks of the Stable Diffusion UNet, providing fine-grained control over the generated image. It is based on the concept that the content/subject understanding of the model is primarily contained within the MID0 and MID1 blocks, as demonstrated in the B-Lora (Content Style implicit separation) paper.
Features

Inject different prompts into specific UNet blocks
Three different node variations for flexible workflow integration
Customize the learning rate of specific blocks to focus on content, lighting, style, or other aspects
Potential for developing a "Mix of Experts" approach by swapping blocks on-the-fly based on prompt content

# Usage

Add the prompt_injection.py node to your ComfyUI custom nodes directory
In your ComfyUI workflow, connect the desired node variation based on your input preferences
Specify the prompts for each UNet block you want to customize
Connect the output to the rest of your workflow and generate the image

# Node Variations

Prompt Injection (Single Prompt): Injects a single prompt into the specified UNet blocks
Prompt Injection (Multiple Prompts): Allows injecting different prompts into each specified UNet block
Prompt Injection (Prompt Dictionary): Accepts a dictionary of block names and their corresponding prompts

# Example
Injecting the prompt "white cat" into the OUTPUT0 and OUTPUT1 blocks, while using the prompt "blue dog" for all other blocks, results in an image with the composition of the "blue dog" prompt but with a cat as the subject/content.
Acknowledgements

Modified and simplified version of the node from: https://github.com/pamparamm/sd-perturbed-attention
Inspired by discussions and findings shared by @Mobioboros and @DataVoid

# Future Work

Investigate the location of different concepts (e.g., lighting) within the UNet blocks
Develop a "guts diagram" of the SDXL UNet to understand where each aspect is stored
Explore the use of different learning rates for specific blocks during fine-tuning or LoRA training
Implement a "Mix of Experts" approach by swapping blocks on-the-fly based on prompt content

Feel free to contribute, provide feedback, and share your findings!
