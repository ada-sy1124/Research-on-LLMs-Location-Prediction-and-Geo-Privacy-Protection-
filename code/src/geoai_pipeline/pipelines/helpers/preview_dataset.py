from datasets import load_from_disk

from geoai_pipeline.config import get_path


def run():
    dataset_path = get_path(
        "PREVIEW_DATASET_PATH",
        "./data/YES/chunk_4",
    )

    dataset = load_from_disk(dataset_path)

    for i in range(len(dataset)):
        dataset[i]["image"].show()


if __name__ == "__main__":
    run()
