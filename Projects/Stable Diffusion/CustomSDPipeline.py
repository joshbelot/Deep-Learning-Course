import torch
from diffusers import StableDiffusionPipeline
import numpy as np

class CustomSDPipeline(StableDiffusionPipeline):
    def __init__(self, vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor, image_encoder=None, requires_safety_checker=True):
        super().__init__(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=safety_checker, feature_extractor=feature_extractor, image_encoder=image_encoder, requires_safety_checker=requires_safety_checker)
        self.intermediate_images = []

    def save_intermediate_images(self, latents, step):
        with torch.no_grad():
            decoded_images = self.vae.decode(latents / 0.18215)
        
        images = decoded_images if isinstance(decoded_images, torch.Tensor) else decoded_images["sample"]
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        self.intermediate_images.append(images)

    def __call__(self, prompt, num_inference_steps, interval=1, **kwargs):
        self.intermediate_images = []

        def callback_fn(step, timestep, latents):
            if step % interval == 0:
                self.save_intermediate_images(latents, step)

        kwargs["callback"] = callback_fn
        kwargs["callback_steps"] = interval

        return super().__call__(prompt, num_inference_steps=num_inference_steps, **kwargs)

    def get_intermediate_images(self):
        return self.intermediate_images

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):

        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )

        return cls(
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            safety_checker=pipeline.safety_checker,
            feature_extractor=pipeline.feature_extractor,
            image_encoder=pipeline.image_encoder,
            requires_safety_checker=pipeline.requires_safety_checker
        )
