import torch
import numpy as np
import matplotlib.pyplot as plt

def sd3m(prompt):
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("mps")
    img = pipe(prompt, height=512, width=512).images[0]
    del pipe
    return img

def sdp(prompt, model, inference, guidance):
    from CustomSDPipeline import CustomSDPipeline
    pipe = CustomSDPipeline.from_pretrained(model, variant="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("mps")
    sd_result = pipe(prompt, num_inference_steps=inference, guidance_scale=guidance)
    intermediate = pipe.get_intermediate_images()
    del pipe
    return sd_result.images[0], intermediate

def generate_image(model_option, prompt, inference, guidance):
    if model_option == "stabilityai/stable-diffusion-3-medium":
        return sd3m(prompt)
    else:
        return sdp(prompt, model_option, inference, guidance)
    
def build_grid(intermediate):
    cols = 5
    rows = (len(intermediate) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    for i, image_set in enumerate(intermediate):
        image = image_set[0]

        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.max() > 1.0:
            image = image / 255.0

        ax = axes[i // cols, i % cols]
        ax.imshow(image)
        ax.axis("off")

    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j // cols, j % cols])
        
    plt.tight_layout()
    return fig
  