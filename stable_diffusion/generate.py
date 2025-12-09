import os
import logging
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F
from tqdm.auto import tqdm

from dataset import prepare_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()
pipe.safety_checker = None

num_inference_steps = 100
guidance_scale = 10.0
batch_size = 2
dataloader = prepare_dataloader(batch_size, name="", ratio=1.0)
path = os.path.abspath("")
os.makedirs(path, exist_ok=True)
os.makedirs(os.path.join(path, "images"), exist_ok=True)
image_list = np.load(
    "image_list.npy", allow_pickle=True
)

# 使用 logging 记录日志到指定目录
log_file = os.path.join(path, "run.log")
logging.basicConfig(
    filename=log_file,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

img_size = 512
resize_transform = transforms.Resize((img_size, img_size))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def pil_to_uint8_tensor(image: Image.Image) -> torch.Tensor:
    tensor = to_tensor(image).unsqueeze(0)
    return tensor.mul(255.0).clamp(0, 255).to(torch.uint8)


def pil_to_lpips_tensor(image: Image.Image) -> torch.Tensor:
    tensor = to_tensor(image).unsqueeze(0).to(device)
    return tensor.mul(2.0).sub(1.0)


def calculate_ssim(img1, img2):
    x = to_tensor(img1).unsqueeze(0)
    y = to_tensor(img2).unsqueeze(0)
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y * mu_y
    sigma_x = torch.clamp(sigma_x, min=0)  # 确保非负

    sigma_y = torch.clamp(sigma_y, min=0)
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean().item()


inception_model = inception_v3(
    weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
).to(device)
inception_model.eval()


all_generated_probs_for_is = []
all_real_probs_for_is = []


def compute_inception_probs(image):
    img_tensor = to_tensor(resize_transform(image))
    img_tensor = F.interpolate(
        img_tensor.unsqueeze(0),
        size=(299, 299),
        mode="bilinear",
        align_corners=False,
    ).to(device)
    img_tensor = normalize(img_tensor)

    with torch.no_grad():
        logits = inception_model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs[0]


def update_is_prob_lists(gen_images, real_images):
    global all_generated_probs_for_is, all_real_probs_for_is
    if not gen_images or not real_images:
        return
    for gen_img, real_img in zip(gen_images, real_images):
        gen_prob = compute_inception_probs(gen_img)
        real_prob = compute_inception_probs(real_img)
        all_generated_probs_for_is.append(gen_prob)
        all_real_probs_for_is.append(real_prob)


def calculate_is_from_probs(probs_list, splits=1):
    if not probs_list:
        return float("nan")
    probs = np.array(probs_list)
    scores = []
    n = len(probs)

    for i in range(splits):
        part = probs[(i * n) // splits : ((i + 1) * n) // splits]
        if part.size == 0:
            continue
        part = np.clip(part, 1e-10, 1.0)
        py = np.mean(part, axis=0)
        py = np.clip(py, 1e-10, 1.0)
        scores.append(
            np.exp(np.mean(np.sum(part * (np.log(part) - np.log(py[None, :])), axis=1)))
        )
    return float(np.mean(scores)) if scores else float("nan")



all_images = []
for i, (eeg, img_label, text) in enumerate(tqdm(dataloader)):
    eeg = eeg.to(device, dtype=torch.float16)
    with torch.no_grad():
        images = pipe(prompt_embeds=eeg, num_inference_steps=100, guidance_scale=10).images

    image_tensors = [transforms.ToTensor()(img) for img in images]  # 转换为 Tensor
    image_numpy = [img.numpy() for img in image_tensors]  # 转换为 NumPy 数组
    all_images.extend(image_numpy)
    print(f"{i+1} / {len(dataloader)}")

np.save(os.path.join(path, "generated_images.npy"), np.array(all_images), allow_pickle=True, fix_imports=True)