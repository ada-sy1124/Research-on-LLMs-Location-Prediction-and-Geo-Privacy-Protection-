# import os
# import json
# from openai import OpenAI
# from PIL import Image

# client = OpenAI(api_key="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")

# # 六大类名称 + 提示词（与之前 prompt 库一致）
# CATEGORIES = {
#     "buildings": [
#         "residential house", "porch", "garage", "balcony", "roof shape", "facade style"
#     ],
#     "traffic_signs_markings": [
#         "road sign", "intersection sign", "no parking sign", "bike lane marking"
#     ],
#     "street_infrastructure": [
#         "streetlight", "fire hydrant", "trash bins", "utility pole", "bus stop"
#     ],
#     "vegetation_landscape": [
#         "tree type", "palm tree", "maple tree", "hedge", "grass strip", "hill"
#     ],
#     "text_language": [
#         "store sign", "address number", "parking permit sign", "school sign", "city logo"
#     ],
#     "vehicles": [
#         "car", "bus", "license plate", "service vehicle"
#     ]
# }

# def analyze_image(img_path):
#     """
#     让 GPT 判断图像中每类是否存在
#     """
#     with open(img_path, "rb") as f:
#         img_bytes = f.read()

#     prompt = (
#         "You are a geo-visual cue classifier. For the given image, determine whether the following "
#         "six semantic categories are present at least once in the scene. Return result in JSON ONLY:\n\n"
#         "CATEGORIES:\n"
#         "1. buildings (residential or institutional structures)\n"
#         "2. traffic_signs_markings\n"
#         "3. street_infrastructure\n"
#         "4. vegetation_landscape\n"
#         "5. text_language (signs, store names, address numbers)\n"
#         "6. vehicles\n\n"
#         "Respond strictly in JSON:\n"
#         "{\n"
#         "  \"buildings\": true/false,\n"
#         "  \"traffic_signs_markings\": true/false,\n"
#         "  \"street_infrastructure\": true/false,\n"
#         "  \"vegetation_landscape\": true/false,\n"
#         "  \"text_language\": true/false,\n"
#         "  \"vehicles\": true/false,\n"
#         "  \"reason\": \"one short sentence explanation\"\n"
#         "}"
#     )

#     response = client.chat.completions.create(
#         model="gpt-5-vision-preview",  # GPT-O / Omni Vision
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": prompt},
#                     {"type": "image", "image": img_bytes}
#                 ]
#             }
#         ]
#     )

#     text = response.choices[0].message.content.strip()
#     try:
#         data = json.loads(text)
#         return data
#     except:
#         return None

# def filter_dataset(input_folder, output_folder):
#     os.makedirs(output_folder, exist_ok=True)
#     kept = []

#     for filename in os.listdir(input_folder):
#         if not filename.lower().endswith(("jpg", "jpeg", "png", "webp")):
#             continue
        
#         img_path = os.path.join(input_folder, filename)
#         result = analyze_image(img_path)

#         if result is None:
#             continue
        
#         # 判断是否六类都 True
#         if all(result.get(cat, False) for cat in CATEGORIES.keys()):
#             kept.append((filename, result))
#             os.system(f"cp '{img_path}' '{output_folder}/'")

#     return kept

# # 运行筛选:
# kept_samples = filter_dataset("images", "filtered_images")

# print(f"✅ 保留数量：{len(kept_samples)}")
# for name, info in kept_samples:
#     print(name, info)


from datasets import load_from_disk
from transformers import pipeline
import matplotlib.pyplot as plt

# 1) 加载 Arrow 数据集
dataset = load_from_disk("/Users/sy1124/Downloads/Documents/geoai/random_streetview/chunk_96")

# 2) 取一张图片
sample = dataset[14]   # 你可以换 index
image = sample["image"]

# 3) 加载模型
pipe = pipeline("zero-shot-object-detection", model="IDEA-Research/grounding-dino-base", device="mps")

# 4) 检测类别（你可以随便换）
labels = ["garage door"]

# 5) 推理
results = pipe(image, candidate_labels=labels)

# 6) 可视化
plt.imshow(image)
ax = plt.gca()

for r in results:
    box = r["box"]
    ax.add_patch(plt.Rectangle(
        (box["xmin"], box["ymin"]),
        box["xmax"] - box["xmin"],
        box["ymax"] - box["ymin"],
        fill=False, color="red", linewidth=2
    ))
    ax.text(box["xmin"], box["ymin"], f'{r["label"]}: {r["score"]:.2f}',
            color="yellow", fontsize=8, backgroundcolor="black")

plt.axis("off")
plt.show()



