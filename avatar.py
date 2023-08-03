from diffusers import StableDiffusionPipeline
import torch
model_id = "riccardogiorato/avatar-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "a magical witch with blue hair with avatartwow style"
image = pipe(prompt).images[0]
image.save("./magical_witch.png")
