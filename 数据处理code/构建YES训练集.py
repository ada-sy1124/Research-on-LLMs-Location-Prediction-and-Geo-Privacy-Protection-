from datasets import load_from_disk, Dataset
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm
import os
import time
import numpy as np
import re

# =========================
# 1) 配置项
# =========================
load_dotenv()
GEMINI_MODEL = "gemini-3-flash-preview" 

# 输入：你已经合并好的 YES 数据集路径
INPUT_DATASET_PATH = "/Applications/Documents/geoai/random_streetview_gemini/YES" ##############################################
# 输出：消融实验结果保存路径
OUTPUT_DIR = "/Applications/Documents/geoai/random_streetview_gemini/YES_mask" ##############################################

BUFFER_SIZE = 30
SLEEP_SECONDS = 15 # 防止 API 报错 429 (Too Many Requests)

# 👇【新增】在这里控制输入范围
START_INDEX = 0       # 从第几条开始。##############################################
END_INDEX = 2061        # 到第几条结束 (跑全量可以设为 999999)  #############################
# 👆

# =========================
# 2) 辅助与计算函数
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    """计算两点之间的haversin距离"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))

def calculate_q_prime(reason_text, excluded_class):
    """解析原始 reason,统计除了 excluded_class 之外的物体数量 (q')"""
    if not reason_text: return 0
    q_prime = 0
    for seg in reason_text.split(";"):
        if ":" in seg:
            cls_name, objs_text = seg.split(":", 1)
            if cls_name.strip() != excluded_class:
                objs = [o.strip() for o in objs_text.split(",") if o.strip()]
                q_prime += len(objs)
    return q_prime

def is_float(element):
    """辅助判断字符串是否可以转为浮点数"""
    try:
        float(element)
        return True
    except ValueError:
        return False

# =========================
# 3) Prompt 与 API 调用函数
# =========================
def get_ablation_prompt(excluded_class):
    return f"""You are an advanced geolocation model.

CRITICAL ABLATION INSTRUCTION (IMPORTANT):
For this specific task, you MUST completely IGNORE any objects belonging to the class "{excluded_class}". 
Pretend objects of this class do NOT exist in the image. 
Do NOT base any of your geographic reasoning on them, and strictly EXCLUDE the "{excluded_class}" class from your final REASONING output.

TASK OVERVIEW (IMPORTANT):
This task consists of TWO SEQUENTIAL STAGES that must be completed IN ORDER.

STAGE 1 — GEOLOCATION REASONING (INTERNAL):
First, analyze the image and estimate its geographic coordinates using only visible evidence (EXCLUDING "{excluded_class}").
Do this reasoning internally. Do NOT output your internal reasoning. 
Output format: 'COORDINATES: <latitude>, <longitude>'

STAGE 2 — EVIDENCE EXTRACTION (OUTPUT):
After determining the coordinates in Stage 1, examine the image again.
Identify ONLY the concrete, physical, visible objects that directly support or justify your predicted location.
Then output the final result using this format: 'REASONING: <structured object list>'

Your final output format (EXACTLY TWO LINES) should be:
Line 1: COORDINATES: <latitude>, <longitude>
Line 2: REASONING: <structured object list>

STRUCTURE OF LINE 2 (MUST FOLLOW EXACTLY):
- Line 2 must start with "REASONING: "
- Format: ClassName: obj1, obj2, obj3; NextClass: obj4, obj5; ...
- Classes must be separated by a semicolon and a space: "; "
- Objects within a class must be separated only by a comma and a space: ", "
- If a class has no relevant objects in the image, do NOT include that class in the output.

REASONING CONTENT RULES (STAGE 2 ONLY):
1. List ONLY the individual, countable, physical objects that SUPPORT the predicted location from Stage 1.
2. Every object must be a single concrete visible instance with an objective visual descriptor.
3. Do NOT use vague quantities such as "many", "some", or "several".
4. Use ONLY the exact numbering format "#X" starting from 1 with consecutive integers.
5. Numbering rule (CRUCIAL):
   - If an object type appears only once, it MUST be written as "<object name> #1".
   - If an object type appears N times, it MUST be written as:
     "<object name> #1, <object name> #2, ... <object name> #N".
6. Do NOT merge identical objects into a single entry.
7. Do NOT repeat the same object in more than one class.
8. Include ONLY physical, visible objects. Do NOT include abstract concepts or inferred assumptions.
9. Use ONLY the following predefined class names (excluding "{excluded_class}"):
   "Road Markings", "Signage & Text", "Vehicles", "Architecture", "Vegetation", "Infrastructure"

EXAMPLE OUTPUT:
COORDINATES: 51.5074, -0.1278
REASONING: Signage & Text: street name sign #1, warning sign #1, parking sign #1; Road Markings: lane line #1, zebra crossing #1; Vehicles: bus #1; Architecture: house #1, house #2; Vegetation: tree #1, tree #2, tree #3; Infrastructure: bollard #1, bollard #1"""

def gemini_predict_ablation(client, model, image, excluded_class):
    """调用 API 进行消融预测并提取新坐标"""
    prompt = get_ablation_prompt(excluded_class)
    lat_val, lon_val = 0.0, 0.0

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[image, prompt],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        text = resp.text.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        
        for line in lines:
            if "COORDINATES:" in line.upper():
                nums = re.findall(r'-?\d+\.?\d*', line)
                valid_nums = [float(n) for n in nums if is_float(n)]
                if len(valid_nums) >= 2:
                    lat_val, lon_val = valid_nums[0], valid_nums[1]
                    break # 成功拿到坐标，跳出循环
    except Exception as e:
        print(f"API Error for class '{excluded_class}': {e}")
    
    return lat_val, lon_val

def save_chunk(buffer_list, out_dir, chunk_id):
    """保存数据块到本地"""
    os.makedirs(out_dir, exist_ok=True)
    Dataset.from_list(buffer_list).save_to_disk(f"{out_dir}/chunk_{chunk_id}")
    print(f"💾 Chunk {chunk_id} saved | Count: {len(buffer_list)}")

# =========================
# 4) 主程序
# =========================
def main():
    client = genai.Client(api_key="") 
    
    print("正在加载数据集...")
    if not os.path.exists(INPUT_DATASET_PATH):
        print(f"找不到数据集路径: {INPUT_DATASET_PATH}")
        return
        
    dataset = load_from_disk(INPUT_DATASET_PATH)
    
    # 👇【新增】利用 Dataset 自带的 select 进行截取，防止 END_INDEX 超出范围报错
    actual_end = min(END_INDEX, len(dataset))
    dataset = dataset.select(range(START_INDEX, actual_end))
    print(f"✅ 已截取范围: {START_INDEX} 到 {actual_end}")
    # 👆
    
    buffer = []
    chunk_id = 0 ##############################################

    # 外层循环：遍历每张图片样本
    for item in tqdm(dataset, desc="Ablation Study"):
        image_obj = item["image"]
        lat_true, lon_true = item["latitude"], item["longitude"]
        d_orig = item["d"]
        q_orig = item["q"]
        reason_classes = item["reason_class"]
        reason_text = item["reason"]

        # 准备并行列表，用来存储该样本的“消融特征表”
        ablated_classes = []
        q_primes = []
        q_ratios = []
        d_primes = []
        d_diffs = []

        # 内层循环：遍历这张图里原先识别出的每一个类
        if reason_classes and q_orig > 0:
            for target_class in reason_classes:
                
                # 1. 计算 q' 和 q'/q
                q_prime = calculate_q_prime(reason_text, target_class)
                q_ratio = q_prime / q_orig
                
                # 2. 调用 API 获取忽略 target_class 后的新坐标
                lat_pred, lon_pred = gemini_predict_ablation(client, GEMINI_MODEL, image_obj, target_class)
                
                # 3. 计算 d' 和 d'-d
                if lat_pred == 0.0 and lon_pred == 0.0:
                    d_prime = 99999.0
                    d_diff = 99999.0 # 解析失败或 API 报错
                else:
                    d_prime = haversine_km(lat_pred, lon_pred, lat_true, lon_true)
                    d_diff = d_prime - d_orig

                # 4. 记录当前类别的消融结果
                ablated_classes.append(target_class)
                q_primes.append(q_prime)
                q_ratios.append(q_ratio)
                d_primes.append(d_prime)
                d_diffs.append(d_diff)

                # 休眠防止触发 API 速率限制
                time.sleep(SLEEP_SECONDS)

        # 组装这条样本的最终数据结构
        new_item = {
            "image": image_obj,
            "latitude_true": lat_true,
            "longitude_true": lon_true,
            "d_original": d_orig,
            "q_original": q_orig,
            "ablated_class": ablated_classes,
            "q_prime": q_primes,
            "q_ratio": q_ratios,
            "d_prime": d_primes,
            "d_diff": d_diffs
        }
        
        buffer.append(new_item)

        # 缓存满则写入硬盘
        if len(buffer) >= BUFFER_SIZE:
            save_chunk(buffer, OUTPUT_DIR, chunk_id)
            buffer.clear()
            chunk_id += 1

    # 保存最后一批不足 BUFFER_SIZE 的尾巴数据
    if buffer:
        save_chunk(buffer, OUTPUT_DIR, chunk_id)

    print("🎉 全部消融实验处理完成！")

if __name__ == "__main__":
    main()