from datasets import load_from_disk, load_dataset
import matplotlib.pyplot as plt


dataset = load_from_disk("/Applications/Documents/geoai/random_streetview_gemini/YES/chunk_0")

# for i in range(len(dataset)):
#     dataset[i]['image'].show()
#     print(i)
#     print(dataset[i]["d"],dataset[i]["q"],dataset[i]["reason_class"])

dataset[2]["image"].show()
# print(dataset[0]["d"],dataset[0]["q"],dataset[0]["reason_class"])
# print(dataset[1]["d"],dataset[1]["q"],dataset[1]["reason_class"])
# print(dataset[8]["latitude"],dataset[8]["longitude"])