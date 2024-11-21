# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, AutoPipelineForImage2Image
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
from PIL import Image

img1 = Image.open('sam_canonical_0.png') 
img2 = Image.open('sam_canonical_1.png')  
new_img = Image.new('RGB', (img1.width + img2.width, img1.height))
new_img.paste(img1, (0, 0))
new_img.paste(img2, (img1.width, 0))
image = np.array(new_img)

# get canny image
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

canny_image.save("edited_canny.png")

edit_prompt = "edit prompt"

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)
image = pipe(
    edit_prompt, num_inference_steps=30, generator=generator, image=canny_image
).images[0]
image.save("edited_images.png")

image = Image.open("edited_images.png")
width, height = image.size
crop_width, crop_height = img1.width, img1.height
for i in range(width // crop_width):
    left = i * crop_width
    right = left + crop_width
    cropped_image = image.crop((left, 0, right, height))
    cropped_image.save(f"edited_canonical_{i}.png")

sdedit_pipeline = AutoPipelineForImage2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
sdedit_pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
sdedit_pipeline.enable_xformers_memory_efficient_attention()

for image_path in ['edited_canonical_0.png', 'edited_canonical_1.png']:
    init_image = load_image(image_path)
    image = sdedit_pipeline(edit_prompt, image=init_image, strength=0.5).images[0]
    image.save(f"refine_{image_path}")





