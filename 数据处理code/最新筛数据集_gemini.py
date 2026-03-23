


from datasets import load_dataset, Dataset
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tqdm import tqdm
import os
import time
import numpy as np
import re
import gc # <---【新增】引入垃圾回收模块，用于释放内存

# =========================
# 1) 可修改变量
# =========================
load_dotenv()

GEMINI_MODEL = "gemini-3-flash-preview" 

DATASET_NAME = "stochastic/random_streetview_images_pano_v0.0.2"


START_INDEX = 0       # 从第几张开始
END_INDEX = 11054      # 到第几张结束
BATCH_SIZE = 30       # ★关键：每次只从服务器加载 batchsize 张图片处理，处理完释放内存


YES_DIR = "/Applications/Documents/geoai/random_streetview_gemini/YES"
NO_DIR  = "/Applications/Documents/geoai/random_streetview_gemini/NO"

BUFFER_SIZE = 30
DIST_THRESHOLD_KM = 5
SLEEP_SECONDS = 20
Temperature = 0.0

GEO_PROMPT = """You are an advanced geolocation model.

TASK OVERVIEW (IMPORTANT):
This task consists of TWO SEQUENTIAL STAGES that must be completed IN ORDER.

STAGE 1 — GEOLOCATION REASONING (INTERNAL):
First, analyze the image and estimate its geographic coordinates using only visible evidence.
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
2. Every object must be a single concrete visible instance with an objective visual descriptor
   (e.g., "blue street name sign #1", "yellow rear license plate #1", "red phone box #1").
3. Do NOT use vague quantities such as "many", "some", or "several".
4. Use ONLY the exact numbering format "#X" starting from 1 with consecutive integers.
5. Numbering rule (CRUCIAL):
   - If an object type appears only once, it MUST be written as "<object name> #1".
   - If an object type appears N times, it MUST be written as:
     "<object name> #1, <object name> #2, ... <object name> #N".
6. Do NOT merge identical objects into a single entry.
7. Do NOT repeat the same object in more than one class.
8. Include ONLY physical, visible objects. Do NOT include abstract concepts or inferred assumptions
   (e.g., "British style", "tropical climate", "European atmosphere").
9. Use ONLY the following predefined class names:
   "Road Markings", "Signage & Text", "Vehicles", "Architecture", "Vegetation", "Infrastructure"

EXAMPLE OUTPUT:
COORDINATES: 51.5074, -0.1278
REASONING: Signage & Text: street name sign #1, warning sign #1, parking sign #1; Road Markings: lane line #1, zebra crossing #1; Vehicles: bus #1; Architecture: house #1, house #2; Vegetation: tree #1, tree #2, tree #3; Infrastructure: bollard #1, bollard #1"""

# =========================
# 2) 函数
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


# def gemini_predict_latlon(client, model, image_obj, prompt):
#     try:
#         resp = client.models.generate_content(
#             model=model,
#             contents=[image_obj, prompt],
#             config=types.GenerateContentConfig(temperature=Temperature), 
#         )
#         text = resp.text.strip()
#     except Exception as e:
#         print(f"API调用错误: {e}")
#         return 0.0, 0.0, "", [], 0

#     lines = [l.strip() for l in text.splitlines() if l.strip()]
    
#     lat_val, lon_val = 0.0, 0.0
#     reason_text = ""
#     classes = []
#     q = 0

#     for line in lines:
#         if "COORDINATES:" in line.upper(): 
#             try:
#                 nums = re.findall(r'-?\d+\.?\d*', line)
#                 valid_nums = []
#                 for n in nums:
#                     try:
#                         valid_nums.append(float(n))
#                     except:
#                         continue
#                 if len(valid_nums) >= 2:
#                     lat_val = valid_nums[0]
#                     lon_val = valid_nums[1]
#             except Exception as e:
#                 print(f"坐标解析异常: {line} | Error: {e}")

#         elif line.startswith("REASONING:"):
#             try:
#                 parts = line.split("REASONING:", 1)
#                 if len(parts) > 1:
#                     reason_text = parts[1].strip()
#                     for seg in reason_text.split(";"):
#                         seg = seg.strip()
#                         if ":" in seg:
#                             cls, objs_text = seg.split(":", 1)
#                             cls = cls.strip()
#                             if cls:
#                                 classes.append(cls)
#                             objs = [o.strip() for o in objs_text.split(",") if o.strip()]
#                             q += len(objs)
#             except Exception as e:
#                 print(f"Reasoning解析异常: {line} | Error: {e}")

#     return lat_val, lon_val, reason_text, classes, q


def gemini_predict_latlon(client, model, image_obj, prompt):
    '''
    模型做预测 (带 503 自动重试机制)
    '''
    # ================= 配置重试参数 =================
    max_retries = 5      # 最大重试次数 (遇到503最多试5次)
    base_wait_time = 5   # 基础等待时间 (秒)
    # ==============================================

    text = ""
    
    # 循环尝试
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[image_obj, prompt],
                config=types.GenerateContentConfig(temperature=Temperature), 
            )
            text = resp.text.strip()
            
            # 如果成功获取到文本，直接跳出重试循环
            if text:
                break 
                
        except Exception as e:
            error_msg = str(e)
            # 检查是否是 503 (服务器忙) 或 429 (配额超限)
            if "503" in error_msg or "429" in error_msg:
                wait_time = base_wait_time * (attempt + 1) # 递增等待：5秒, 10秒, 15秒...
                print(f"⚠️ 服务器繁忙 (503/429)，正在进行第 {attempt+1}/{max_retries} 次重试，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                # 如果是其他严重错误（如参数错误），不重试，直接退出
                print(f"❌ API 致命错误: {e}")
                return 0.0, 0.0, "", [], 0

    # 如果重试多次后 text 依然为空
    if not text:
        print("❌ 多次重试失败，跳过此图")
        return 0.0, 0.0, "", [], 0

    # ---------------- 以下是原本的解析逻辑 (保持不变) ----------------
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    lat_val, lon_val = 0.0, 0.0
    reason_text = ""
    classes = []
    q = 0

    for line in lines:
        if "COORDINATES:" in line.upper(): 
            try:
                nums = re.findall(r'-?\d+\.?\d*', line)
                valid_nums = []
                for n in nums:
                    try:
                        valid_nums.append(float(n))
                    except:
                        continue
                if len(valid_nums) >= 2:
                    lat_val = valid_nums[0]
                    lon_val = valid_nums[1]
            except Exception as e:
                print(f"坐标解析异常: {line} | Error: {e}")

        elif line.startswith("REASONING:"):
            try:
                parts = line.split("REASONING:", 1)
                if len(parts) > 1:
                    reason_text = parts[1].strip()
                    for seg in reason_text.split(";"):
                        seg = seg.strip()
                        if ":" in seg:
                            cls, objs_text = seg.split(":", 1)
                            cls = cls.strip()
                            if cls:
                                classes.append(cls)
                            objs = [o.strip() for o in objs_text.split(",") if o.strip()]
                            q += len(objs)
            except Exception as e:
                print(f"Reasoning解析异常: {line} | Error: {e}")

    return lat_val, lon_val, reason_text, classes, q


def save_chunk(buffer_list, out_dir, chunk_id):
    ds_chunk = Dataset.from_list(buffer_list)
    save_path = f"{out_dir}/chunk_{chunk_id}"
    ds_chunk.save_to_disk(save_path)
    print(f" saved to {save_path} | Count: {len(buffer_list)}")


# =========================
# 3) 主程序
# =========================
def main():
    client = genai.Client(api_key="")
    
    os.makedirs(YES_DIR, exist_ok=True)
    os.makedirs(NO_DIR, exist_ok=True)

    # 状态变量放在循环外，保证 chunk 编号连续
    yes_buffer, no_buffer = [], []
    yes_chunk_id, no_chunk_id = 0, 0

    # ---【修改核心】外层循环：分批次处理 ---
    # 例如 range(0, 1110, 50) -> 0, 50, 100...
    for batch_start in range(START_INDEX, END_INDEX, BATCH_SIZE):
        
        # 计算当前批次的结束位置
        batch_end = min(batch_start + BATCH_SIZE, END_INDEX)
        current_split = f"train[{batch_start}:{batch_end}]"
        
        print(f"\n🚀 开始处理批次: {batch_start} 到 {batch_end} (Split: {current_split})")
        
        # try:
        #     # 每次只加载一小部分数据集
        dataset = load_dataset(DATASET_NAME, split=current_split)
        # except Exception as e:
        #     print(f"❌ 数据集加载失败，跳过此批次: {e}")
        #     continue

        # 这里的 tqdm 是针对这 50 张图的进度条
        for item in tqdm(dataset, desc=f"Batch {batch_start}-{batch_end}"):
            lat_true = float(item["latitude"])
            lon_true = float(item["longitude"])

            lat_pred, lon_pred, reason_text, reason_classes, q = gemini_predict_latlon(
                client, GEMINI_MODEL, item["image"], GEO_PROMPT
            )
       
            if lat_pred == 0.0 and lon_pred == 0.0 and q == 0:
                dist = 99999.0 # 解析失败
            else:
                dist = haversine_km(lat_pred, lon_pred, lat_true, lon_true)
            
            label = "YES" if dist <= DIST_THRESHOLD_KM else "NO"

            item_out = {
                "image": item["image"],
                "latitude_pred": lat_pred,
                "longitude_pred": lon_pred,
                "latitude": lat_true,
                "longitude": lon_true,
                "d": dist,
                "reason": reason_text,
                "reason_class": reason_classes,
                "q": q,
            }

            if label == "YES":
                yes_buffer.append(item_out)
                if len(yes_buffer) >= BUFFER_SIZE:
                    save_chunk(yes_buffer, YES_DIR, yes_chunk_id)
                    yes_buffer.clear()
                    yes_chunk_id += 1
            else:
                no_buffer.append(item_out)
                if len(no_buffer) >= BUFFER_SIZE:
                    save_chunk(no_buffer, NO_DIR, no_chunk_id)
                    no_buffer.clear()
                    no_chunk_id += 1

            time.sleep(SLEEP_SECONDS)

        # ---【内存清理】当前批次结束后，手动清理 ---
        del dataset
        gc.collect()
        print(f"✅ 批次 {batch_start}-{batch_end} 完成，内存已清理。")

    # 处理剩余的 buffer
    if yes_buffer:
        save_chunk(yes_buffer, YES_DIR, yes_chunk_id)
    if no_buffer:
        save_chunk(no_buffer, NO_DIR, no_chunk_id)

    print("🎉 所有批次处理完成!")


if __name__ == "__main__":
    main()





# image latitude_pred longitude_pred latitude longitude d reason reason_class q