import json
import os

from datasets import load_from_disk
from tqdm import tqdm

from geoai_pipeline.config import get_path


def run():
    dataset_path = get_path(
        "TRAINSET_DATASET_PATH",
        "./data/train_set+no",
    )
    image_save_dir = get_path(
        "TRAINSET_IMAGE_SAVE_DIR",
        "./data/Image",
    )
    jsonl_output_path = get_path(
        "TRAINSET_JSONL_OUTPUT_PATH",
        "./data/trainset.jsonl",
    )

    dataset = load_from_disk(dataset_path)
    os.makedirs(image_save_dir, exist_ok=True)

    with open(jsonl_output_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(dataset)):
            img = sample["image"]
            img_filename = f"street_view_{i}.jpg"
            img_absolute_path = os.path.join(image_save_dir, img_filename)
            img.save(img_absolute_path)

            target_answer = str(sample.get("label"))
            swift_dict = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "<image>Analyze the image and output the target category from 'Architecture', "
                            "'Infrastructure', 'Road Markings', 'Signage & Text', 'Vegetation', "
                            "'Vehicles', 'Nothing' that, when ignored or masked, maximizes the "
                            "prevention of precise geolocation prediction by an LLM while minimizing "
                            "information destruction.\n\n"
                            "[Example]\n"
                            "Output: Vehicles\n\n"
                            "Strictly follow the example format and output ONLY ONE of these 7 options. "
                            "Do not include any other explanations or extra characters."
                        ),
                    },
                    {"role": "assistant", "content": target_answer},
                ],
                "images": [img_absolute_path],
            }

            f.write(json.dumps(swift_dict, ensure_ascii=False) + "\n")

    print("\n 转换大功告成！")


if __name__ == "__main__":
    run()
