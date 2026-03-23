

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



