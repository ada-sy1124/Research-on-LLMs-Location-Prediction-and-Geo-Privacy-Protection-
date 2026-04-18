import re
import time

from google.genai import types


def gemini_predict_latlon(client, model: str, image_obj, prompt: str, temperature: float = 0.0, max_retries: int = 5, base_wait_time: int = 5):
    text = ""

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[image_obj, prompt],
                config=types.GenerateContentConfig(temperature=temperature),
            )
            text = (resp.text or "").strip()
            if text:
                break
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            if "503" in error_msg or "429" in error_msg:
                wait_time = base_wait_time * (attempt + 1)
                print(f"⚠️ API 繁忙 (503/429)，第 {attempt + 1}/{max_retries} 次重试，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"❌ API 致命错误: {e}")
                return 0.0, 0.0

    if not text:
        return 0.0, 0.0

    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        if "COORDINATES:" not in line.upper():
            continue
        nums = re.findall(r"-?\d+\.?\d*", line)
        valid_nums = []
        for n in nums:
            try:
                valid_nums.append(float(n))
            except ValueError:
                continue
        if len(valid_nums) >= 2:
            return valid_nums[0], valid_nums[1]

    return 0.0, 0.0
