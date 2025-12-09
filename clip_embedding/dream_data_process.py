imagenet_labels_40 = {
    "n02389026": "sorrel (horse)",
    "n03888257": "parachute",
    "n03584829": "iron, smoothing iron",
    "n02607072": "anemone fish, clownfish",
    "n03297495": "espresso maker",
    "n03063599": "coffee mug",
    "n03792782": "mountain bike, all-terrain bike",
    "n04086273": "revolver, six-shooter",
    "n02510455": "giant panda",
    "n11939491": "daisy",
    "n02951358": "canoe",
    "n02281787": "lycaenid, lycaenid butterfly",
    "n02106662": "German shepherd",
    "n04120489": "running shoe",
    "n03590841": "jack-o'-lantern",
    "n02992529": "cellular telephone, cellphone",
    "n03445777": "golf ball",
    "n03180011": "desktop computer",
    "n02906734": "broom",
    "n07873807": "pizza",
    "n03773504": "missile",
    "n02492035": "capuchin, ringtail (Cebus capucinus)",
    "n03982430": "pool table, billiard table",
    "n03709823": "mailbag, postbag",
    "n03100240": "convertible",
    "n03376595": "folding chair",
    "n03877472": "pajama, pyjama",
    "n03775071": "mitten",
    "n03272010": "electric guitar",
    "n04069434": "camera",
    "n03452741": "grand piano",
    "n03792972": "mountain tent",
    "n07753592": "banana",
    "n13054560": "bolete (mushroom)",
    "n03197337": "digital watch",
    "n02504458": "African elephant",
    "n02690373": "airliner",
    "n03272562": "electric locomotive",
    "n04044716": "radio telescope",
    "n02124075": "Egyptian cat",
}

import os
import numpy as np
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
from compel import Compel
from diffusers import StableDiffusionPipeline

flag = 2  # 0,1,2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = BlipProcessor.from_pretrained("")
model = BlipForConditionalGeneration.from_pretrained("").to(device)

sd_pipe = StableDiffusionPipeline.from_single_file("",torch_dtype=torch.float16)

tokenizer = sd_pipe.tokenizer
text_encoder = sd_pipe.text_encoder.to(device)
comp = Compel(
    tokenizer=tokenizer,
    text_encoder=text_encoder.to(device),
    device=device,
)


image_list = np.load("", allow_pickle=True)


data = np.load("",allow_pickle=True)
if flag == 0:
    save_path = "../data/class_caption/data_total.npy"
elif flag == 1:
    save_path = "../data/uncondition_caption/data_total.npy"
else:
    save_path = "../data/condition_caption/data_total.npy"
new_data = []

for one in tqdm(data):
    img_label = one["img_label"]
    img_name = image_list[img_label]
    class_name = img_name.split("_")[0]
    class_template = f"This image shows {imagenet_labels_40[class_name]}. The {imagenet_labels_40[class_name]}"

    if flag == 0:
        caption = f"a picture of {imagenet_labels_40[class_name]}"
    else:
        img_path = os.path.join(
            "",
            img_name.split("_")[0],
            img_name + ".JPEG",
        )

        img = Image.open(img_path).convert("RGB")

        if flag == 1:
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                text_ids = model.generate(
                    **inputs,
                    max_length=30,
                    min_length=10,
                    repetition_penalty=1.2,
                    early_stopping=True,
                    num_beams=5,
                )
        else: 
            inputs = processor(images=img, text=class_template, return_tensors="pt").to(
                device
            )
            with torch.no_grad():
                text_ids = model.generate(
                    **inputs,
                    max_length=50,  
                    min_length=10,
                    repetition_penalty=1.2,  
                    early_stopping=True,  
                    num_beams=5,  
                    do_sample=True,  
                    top_p=0.9,  
                )

        
        caption = processor.batch_decode(text_ids, skip_special_tokens=True)[0]
    
    one["text"] = caption

    with torch.no_grad():
        text_embeddings = comp(caption)

    one["text_embedding"] = text_embeddings
    del one["image"]
    new_data.append(one)

np.save(save_path, new_data)
print(len(new_data))
