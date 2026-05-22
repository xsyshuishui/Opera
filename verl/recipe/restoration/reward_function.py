import re
import json
import os
from typing import Optional, List, Dict, Any
import requests
from openai import OpenAI

# 这里是API接口的基本URL
tool_api_base = os.environ.get("TOOL_BASE", "http://127.0.0.1:23200/")
llm_judge_base_url = os.environ.get("LLM_JUDGE_BASE", "http://localhost:8002/v1")
available_tools = [
    'restormer.gaussian_denoise_15',
    'restormer.gaussian_denoise_25',
    'restormer.gaussian_denoise_50',
    'restormer.real_denoise',
    'restormer.derain',
    'restormer.defocus_deblur',
    'restormer.motion_deblur',
    'xrestormer.denoise_50',
    'xrestormer.derain',
    'xrestormer.dehaze',
    'xrestormer.deblur',
    'xrestormer.super_resolution',
    'swinir.super_resolution',
    'swinir.gaussian_denoise_15',
    'swinir.gaussian_denoise_25',
    'swinir.gaussian_denoise_50',
    'swinir.dejpeg',
    "brighten.gamma_correction",
    "brighten.constant_shift"
]
available_degradations = [
    'noise',
    'rain',
    'haze',
    'defocus_blur',
    'motion_blur',
    'low_resolution',
    # 'jpeg'
    'dark'
]
# 从API接口调用图像复原
def restore_image(image_name: str, model_order: List[int]) -> Dict[str, Any]:

    # 构建请求数据
    data = {
        "image_id": image_name,
        "models": model_order
    }
    # 调用REST API进行图像复原
    response = requests.post(f"{tool_api_base}/restore", json=data, timeout=180)
    
    if response.status_code != 200:
        raise Exception(f"Failed to restore image. API returned {response.status_code}: {response.text}")
    
    # 从响应中获取复原图像和质量评估
    result = response.json()
    score = result.get('score')
    # 提取PSNR和SSIM评分
    try:
        psnr = score["psnr"]['score']
        ssim = score["ssim"]['score']
        lpips = score['lpips']['score']
        maniqa = score['maniqa']['score']
        clipiqa = score['clipiqa']['score']
        musiq = score['musiq']['score']
    except Exception as e:
        print(f"Failed to parse score from response: {e}")
        raise Exception("Failed to get score from return")
    return psnr, ssim, lpips, maniqa, clipiqa, musiq

# 提取<answer></answer>标签内的内容
def extract_answer(text: str) -> Optional[str]:
    """
    从给定的文本中提取<answer></answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部的内容，如果未找到则返回None。
    """
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_model_order_from_answer(answer_text: str) -> List[str]:
    """
    从给定的 answer_text 中提取模型顺序。
    
    参数:
        answer_text (str): JSON 数组的字符串，包含任务名称
        
    返回:
        List[int]: 对应的模型 ID 顺序列表
    """
    try:
        import ast
        # 解析 JSON 数组
        tools = ast.literal_eval(answer_text)
        for tool in tools:
            if tool not in available_tools:
                print(f"Unavailable tool: {tool}")
                return []
        assert type(tools) == list
        return tools
    except Exception as e:
        print(f"[Warning] Error decoding answer_text: {e}")
        print(f"Answer = {answer_text}")
        return []  # 如果解析失败，返回空列表

def extract_degradation_from_answer(degradation_text: str) -> List[str]:
    try:
        import ast
        # 解析 JSON 数组
        degradations = ast.literal_eval(degradation_text)
        for degradation in degradations:
            if degradation not in available_degradations:
                print(f"Unavailable degradation: {degradation}")
                return []
        return degradations
    except Exception as e:
        print(f"Error decoding answer_text or invalid task: {e}")
        return []  # 如果解析失败，返回空列表

def extract_gt_degradation_from_image_id(image_id: str) -> List[str]:
    # image_id = "1234+rain+lr.png"
    tmp = image_id.split('.')[0]
    tmp = tmp.split('+')[1:]
    mp = {
        'lr': 'low_resolution',
        'noise': 'noise',
        'haze': 'haze',
        'rain': 'rain',
        'motionblur': 'motion_blur',
        'defocusblur': 'defocus_blur',
        "jpeg": "jpeg",
        'dark': "dark"
    }
    ret = []
    for item in tmp:
        if item in mp:
            ret.append(mp[item])
    return ret


client = OpenAI(api_key="EMPTY", base_url=llm_judge_base_url)

def llm_judge_call(model_output: str) -> str:
    sys_prompt = """\
You are a rigorous planning problem evaluator.

I will provide you with two parts:
A Reasoning Process describing how a planning problem is analyzed and solved
A Final Plan representing the final planning decision or outcome

Your task is to evaluate them according to the following criteria:
1. Evaluate the Reasoning Process
- The reasoning process must NOT be empty
- It must contain meaningful, coherent, and logical reasoning steps
- It should include analysis of constraints, assumptions, or decision logic
- If the reasoning process is missing, empty, superficial, or logically flawed, mark it as unreasonable

2. Check Consistency Between Reasoning Process and Final Plan
- The final plan must be logically derivable from the reasoning process
- There should be no contradictions between the reasoning process and the final plan
- If the reasoning supports one conclusion but the final plan states another, mark them as inconsistent

3. Provide a Clear Judgment and Explanation
Only output a single "Yes" or "No". Do not provide other explanations or text.
"""
    content = [{"type": "text", "text": model_output}]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": content},
    ]
    response = client.chat.completions.create(
        model="pangu_embedded_7b",
        messages=messages,
        temperature=0,
        top_p=0.001,
        max_tokens=10096,
    )
    output_text = response.choices[0].message.content
    print("LLM judge output:", output_text)
    output_text = output_text.split("[unused17]")[1].strip().split(" ")[0]
    print(f"Parsed LLM judge: {output_text}")
    return output_text

def compute_score(data_source, solution_str: str, ground_truth: str, extra_info: Dict[str, Any], step=0) -> float:
    predict_str = solution_str
    print(f'[compute_score] [*BEGIN*]{json.dumps(predict_str)}[*END*]')
    
    # 变量初始化
    is_format_error = False
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    
    if count_think_1 != 1 or count_think_2 != 1:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    
    if count_answer_1 != 1 or count_answer_2 != 1:
        is_format_error = True

    count_degradation_1 = predict_no_think.count("<degradation>")
    count_degradation_2 = predict_no_think.count("</degradation>")

    if count_degradation_1 != 1 or count_degradation_2 != 1:
        is_format_error = True

    degradation_text = predict_str.split("<degradation>")[-1].split("</degradation>")[0].strip()

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()

    model_order = extract_model_order_from_answer(answer_text)

    image_name = extra_info.get('image_name')

    reward_consistency = 0
    for retry in range(3):
        try:
            reward_consistency = 1 if (llm_judge_call(predict_str) == "Yes") else 0
            break
        except Exception as e:
            print(f"Error when get consistency reward: {e}, retry: {retry}")

    reward2 = 0
    num_degradations = 0
    if "Test" not in image_name:
        try:
            degradation_set = extract_degradation_from_answer(degradation_text)

            gt_degradation_set = extract_gt_degradation_from_image_id(image_name)
            
            s1, s2 = set(degradation_set), set(gt_degradation_set)
            
            num_degradations = len(s2)
            if not s1 and not s2:
                reward2 = 1.0  # 都为空，视为完全相等

            intersection = len(s1 & s2)
            precision = intersection / len(s1) if s1 else 0
            recall = intersection / len(s2) if s2 else 0

            reward2 = (precision + recall) / 2
            if len(set(degradation_set)) != len(degradation_set):
                reward2 = 0
        except Exception as e:
            print(f"Error when get degradation reward: {e}")
        
    acc_reward = 0.0
    reward = 0.0
    format_reward = 0.0
    psnr, ssim, lpips, maniqa, clipiqa, musiq = 0, 0, 0, 0, 0, 0

    if model_order == [] or len(model_order) > 6:
        pass
    else:
        # 从extra_info中提取image_name
        image_name = extra_info.get('image_name')
        
        # 根据模型顺序调用REST API，获取PSNR和SSIM
        for retry in range(5):
            try:
                psnr, ssim, lpips, maniqa, clipiqa, musiq = restore_image(image_name, model_order=model_order)
                break
            except Exception as e:
                print(f"Error during image restoration: {str(e)}")
        if psnr < 1:
            print("Retry limit reached.")
        else:
            print(f"Judgement: PSNR={psnr}, SSIM={ssim}, LPIPS={lpips}, MANIQA={maniqa}, CLIP-IQA={clipiqa}, MUSIQ={musiq}")
            
            acc_reward = psnr / 9 + ssim / 0.3 + ( -lpips / 0.3 + clipiqa / 0.1 + musiq / 7.5) * 2
            # 格式错误的奖励
            format_reward = 0.0 if is_format_error else 1.0
            print(f"reward = {acc_reward}, {format_reward}")
            reward = format_reward * (reward2 + (reward2>0.8) * acc_reward) * reward_consistency
    
    # 计算最终的奖励
    ok = format_reward * (reward2 > 0.8)
    ret = {
        "score": reward,
        "num_tools": len(model_order),
        "num_degradations": num_degradations,
        "psnr": psnr * ok,
        "ssim": ssim * ok,
        "lpips": -100 if not ok else -lpips,
        "maniqa": maniqa * ok,
        "clipiqa": clipiqa * ok,
        "musiq": musiq * ok,
        "degradation_reward": reward2 * format_reward,
        "consistency_reward": reward_consistency,
        "format_reward": format_reward
    }
    for tools in available_tools:
        ret[tools] = model_order.count(tools)
    return ret

from concurrent.futures import ThreadPoolExecutor
def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for data_source, predict_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos
        ):
            future = executor.submit(compute_score, data_source, predict_str, ground_truth, extra_info)
            futures.append(future)
        results = [future.result() for future in futures]
    return results

if __name__ == "__main__":
    ret = compute_score(
        "source",
        "<think></think><degradation>['motion_blur', 'dark']</degradation>"
        "<answer>['brighten.gamma_correction', 'restormer.motion_deblur']</answer>",
        "",
        {'image_name': '000001+dark+motionblur.png'},
    )
    print(ret)