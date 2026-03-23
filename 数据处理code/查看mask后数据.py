from datasets import load_from_disk


dataset = load_from_disk("/Applications/Documents/geoai/random_streetview_gemini/YES_mask/chunk_0")

sample = dataset[3]

i = sample["q_ratio"].index(min(sample["q_ratio"]))

print(sample["ablated_class"][i])



    # new_item = {
    #         "image": image_obj,
    #         "latitude_true": lat_true,
    #         "longitude_true": lon_true,
    #         "d_original": d_orig,
    #         "q_original": q_orig,
    #         "ablated_class": ablated_classes,
    #         "q_prime": q_primes,
    #         "q_ratio": q_ratios,
    #         "d_prime": d_primes,
    #         "d_diff": d_diffs
    #     }