# --- START OF FILE test.py ---

from flask import Flask, render_template, request, jsonify, Response
import time
import serial
import threading
import cv2
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests # 用于 API 调用
import base64   # 用于 Base64 编码
import hashlib  # 用于 SHA256 签名
import uuid     # 用于生成 salt
import subprocess # 用于调用 ffmpeg

# 导入您自己的模块
from arm_module import Arm
from tpu_module import Control_TPU # <-- 使用新的TPU模块

# --- 有道智云短语音识别配置 ---
YOUDAO_ASR_URL = "https://openapi.youdao.com/asrapi"
YOUDAO_APP_KEY = "739d5abf63f230b6"  # 替换为您的应用ID
YOUDAO_APP_SECRET = "oPgrJvXIN1LCjvACCsj1VeP7Fqf8QGnG" # 替换为您的应用密钥
YOUDAO_ASR_RATE = 16000
WAVE_OUTPUT_FILENAME = "temp_recorded_audio.raw" # 原始上传文件，扩展名可能不准，ffmpeg会处理

# --- DeepSeek NLU 配置 ---
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY = "sk-6e296264a8f24154982870f9a8efcb21" # 替换为您的 DeepSeek API Key
DEEPSEEK_MODEL = "deepseek-chat" # 或 "deepseek-reasoner"

# --- TPU Model Configuration (New) ---
TPU_MODEL_NAME = "rcModel_3"      # 要使用的模型名称(无后缀)
TPU_CONF_THRES = 0.5            # 置信度阈值
TPU_IOU_THRES = 0.65             # IOU阈值
TPU_IMAGE_WIDTH = 640           # 模型推理宽度
TPU_IMAGE_HEIGHT = 640          # 模型推理高度
TPU_REMOTE_MODE = False
TPU_REMOTE_URL = '192.168.137.1'

# --- 其他全局配置 ---
STM32_SERIAL_PORT = "/dev/ttyUSB0"
STM32_SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1
DETECTION_ENABLED = True
PICK_HEIGHT = 62 # 拾取时机械臂末端的目标高度
DROP_HEIGHT_OFFSET = 30 # 放置时，物体底部距离放置平面的高度，机械臂末端会比这个再高一些
ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z = 0, -180, 200 # 机械臂安全HOME点
ARM_RESET_TIME = 1500
ARM_TRANSLATE_X_STEP, ARM_TRANSLATE_Y_STEP, ARM_TRANSLATE_Z_STEP = 15, 15, 15 # 手动移动步长
ARM_MOVE_TIME = 300 # 手动移动时间
PUMP_ROTATE_SMALL_STEP = 15

CHINESE_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CHINESE_FONT_SIZE = 16
pil_font = None
TABLE_Z_PLANE = 50 # 假设桌面的物体底部接触高度是50mm
DROP_HEIGHT_OFFSET = 30 # 放置时，物体底部悬空的高度，机械臂末端吸盘会比这个稍高
# --- 机械臂状态定义 ---
ARM_STATE_IDLE = "IDLE"
ARM_STATE_RUNNING_TRAJECTORY = "RUNNING_TRAJECTORY"
ARM_STATE_PAUSED_TRAJECTORY = "PAUSED_TRAJECTORY"
ARM_STATE_HOMING = "HOMING"
ARM_STATE_MANUAL_MOVE = "MANUAL_MOVE"
ARM_STATE_PICKING_SUBMITTED = "PICKING_SUBMITTED"
ARM_STATE_PICKING_MOVE_TO_SUCK = "PICKING_MOVE_TO_SUCK"
ARM_STATE_PICKING_SUCKING = "PICKING_SUCKING"
ARM_STATE_PICKING_LIFTING = "PICKING_LIFTING"
# ARM_STATE_PICKING_MOVE_TO_DROP (由 MOVING_TO_PLACE 替代)
# ARM_STATE_PICKING_RELEASING (由 PLACING_OBJECT 替代)
ARM_STATE_PICKING_HOMING_AFTER = "PICKING_HOMING_AFTER"
ARM_STATE_LISTENING = "LISTENING" # 主要由客户端管理
ARM_STATE_RECOGNIZING_VOICE = "RECOGNIZING_VOICE" # ASR处理中
ARM_STATE_UNDERSTANDING_NLU = "UNDERSTANDING_NLU" # NLU处理中
ARM_STATE_MOVING_TO_PLACE = "MOVING_TO_PLACE"     # 移动到放置点
ARM_STATE_PLACING_OBJECT = "PLACING_OBJECT"       # 放置物体

# --- 全局变量 ---
app_lock = threading.Lock()
current_arm_state = ARM_STATE_IDLE
current_trajectory_index = 0
PREDEFINED_TRAJECTORY = [
    (50, -150, 180, 1200), (50, -200, 170, 1200), (0,  -200, 160, 1000),
    (-50,-180, 180, 1200),
    (50, -150, 180, 1200), (50, -200, 170, 1200), (0,  -200, 160, 1000),
    (ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, 1500)
]
TOTAL_TRAJECTORY_POINTS = len(PREDEFINED_TRAJECTORY)
stm32_current_light_status = "ALL_OFF"
was_running_when_homing_triggered = False
current_pump_status = "OFF"
current_pump_angle = 0
detected_objects_list = [] # 由 TPU 任务填充
tpu_instance = None
stop_threads_flag = False
stm32_serial_conn = None
my_arm_instance = None
stm32_reader_thread_obj = None
arm_bg_thread_obj = None
object_detection_thread_obj = None # TPU检测线程对象
camera = None
camera_lock = threading.Lock()
CAMERA_INDEX = 0
last_voice_command_text = "待识别..." # ASR的原始输出
last_nlu_result_text = "待理解..."    # NLU处理后的反馈或主要内容
last_nlu_status = "就绪"             # NLU处理状态

# 预定义的放置位置名称到坐标的映射 (简化示例)
TABLE_Z_PLANE = 50 # 假设桌面的物体底部接触高度是50mm
PREDEFINED_PLACE_LOCATIONS = {
    "桌子左上角": {"x": -80, "y": -150, "z": TABLE_Z_PLANE + DROP_HEIGHT_OFFSET},
    "桌子中间": {"x": 0, "y": -200, "z": TABLE_Z_PLANE + DROP_HEIGHT_OFFSET},
    "桌子右上角": {"x": 80, "y": -150, "z": TABLE_Z_PLANE + DROP_HEIGHT_OFFSET},
    "桌子左边": {"x": -70, "y": -180, "z": TABLE_Z_PLANE + DROP_HEIGHT_OFFSET},
    "桌子右边": {"x": 70, "y": -180, "z": TABLE_Z_PLANE + DROP_HEIGHT_OFFSET},
    "默认放置区": {"x": ARM_HOME_POS_X + 20, "y": ARM_HOME_POS_Y + 20, "z": ARM_HOME_POS_Z - 50}
}

app = Flask(__name__)

# --- 有道智云语音识别辅助函数 ---
def truncate_q_for_sign(q_base64):
    q_len = len(q_base64)
    if q_len <= 20: return q_base64
    return q_base64[:10] + str(q_len) + q_base64[q_len-10:]

def calculate_youdao_sign(app_key, q_for_sign, salt, curtime, app_secret):
    sign_str = app_key + q_for_sign + salt + curtime + app_secret
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(sign_str.encode('utf-8'))
    return hash_algorithm.hexdigest()

def convert_audio_to_youdao_wav(input_filepath, output_filepath):
    command = [
        'ffmpeg', '-y', '-i', input_filepath,
        '-ar', str(YOUDAO_ASR_RATE), '-ac', '1',
        '-sample_fmt', 's16', '-acodec', 'pcm_s16le',
        output_filepath
    ]
    try:
        print(f"执行FFmpeg转换: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=15)
        if process.returncode != 0:
            print(f"FFmpeg 转换失败 (返回码 {process.returncode}):\n  STDOUT: {stdout.decode(errors='ignore')}\n  STDERR: {stderr.decode(errors='ignore')}")
            return False
        if not os.path.exists(output_filepath) or os.path.getsize(output_filepath) == 0:
            print(f"FFmpeg 转换后输出文件不存在或为空: {output_filepath}\n  STDOUT: {stdout.decode(errors='ignore')}\n  STDERR: {stderr.decode(errors='ignore')}")
            return False
        print(f"音频成功转换为: {output_filepath}")
        return True
    except FileNotFoundError: print("FFmpeg错误：找不到ffmpeg命令。"); return False
    except subprocess.TimeoutExpired:
        print("FFmpeg错误：转换超时。")
        if 'process' in locals() and process.poll() is None: process.kill(); process.wait()
        return False
    except Exception as e: print(f"FFmpeg转换时发生未知错误: {e}"); return False

def recognize_speech_youdao(audio_filepath_original):
    global last_voice_command_text, last_nlu_status, app_lock # current_arm_state 由调用者管理或在NLU后统一处理

    if YOUDAO_APP_KEY == "YOUR_YOUDAO_APP_KEY" or YOUDAO_APP_SECRET == "YOUR_YOUDAO_APP_SECRET":
        # ... (配置错误处理)
        with app_lock: last_nlu_status = "ASR失败: 配置错误"
        return None

    if not os.path.exists(audio_filepath_original):
        # ... (文件丢失错误处理)
        with app_lock: last_nlu_status = "ASR失败: 原始音频丢失"
        return None

    converted_audio_path = os.path.splitext(audio_filepath_original)[0] + "_converted.wav"
    if not convert_audio_to_youdao_wav(audio_filepath_original, converted_audio_path):
        with app_lock: last_nlu_status = "ASR失败: 音频转换错误"
        return None
    
    audio_to_process_path = converted_audio_path
    try:
        if not os.path.exists(audio_to_process_path) or os.path.getsize(audio_to_process_path) == 0:
            with app_lock: last_nlu_status = "ASR失败: 转换后音频丢失"
            return None

        with open(audio_to_process_path, 'rb') as audio_file:
            audio_content_binary = audio_file.read()
        q_base64 = base64.b64encode(audio_content_binary).decode('utf-8')
        
        curtime, salt = str(int(time.time())), str(uuid.uuid4())
        q_for_sign_calc = truncate_q_for_sign(q_base64)
        sign = calculate_youdao_sign(YOUDAO_APP_KEY, q_for_sign_calc, salt, curtime, YOUDAO_APP_SECRET)

        payload = {
            'q': q_base64, 'langType': 'zh-CHS', 'appKey': YOUDAO_APP_KEY, 'salt': salt,
            'curtime': curtime, 'sign': sign, 'signType': 'v3', 'format': "wav",
            'rate': str(YOUDAO_ASR_RATE), 'channel': '1', 'type': '1'
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(YOUDAO_ASR_URL, data=payload, headers=headers, timeout=20)
        response.raise_for_status()
        result_data = response.json()
        print(f"YOUDAO_ASR_RESPONSE: {result_data}")

        error_code = result_data.get("errorCode")
        if error_code == "0" and result_data.get("result"):
            recognized_text = result_data["result"][0] if result_data["result"] else ""
            with app_lock: last_voice_command_text = recognized_text
            print(f"YOUDAO_ASR_SUCCESS: Recognized text: '{recognized_text}'")
            return recognized_text
        else:
            with app_lock: last_nlu_status = f"ASR失败({error_code})"
            return None
    except requests.exceptions.RequestException as e:
        print(f"YOUDAO_ASR_ERROR: Request failed: {e}")
        with app_lock: last_nlu_status = "ASR网络错误"
        return None
    except json.JSONDecodeError:
        print(f"YOUDAO_ASR_ERROR: Failed to decode JSON response")
        with app_lock: last_nlu_status = "ASR响应格式错误"
        return None
    except Exception as e:
        print(f"YOUDAO_ASR_ERROR: Unexpected error: {e}")
        with app_lock: last_nlu_status = "ASR未知错误"
        return None
    finally:
        if os.path.exists(converted_audio_path):
            try: os.remove(converted_audio_path)
            except Exception as e_del: print(f"删除转换文件失败: {e_del}")

# --- DeepSeek NLU 函数 ---
def call_deepseek_nlu(text_input):
    global last_nlu_result_text, last_nlu_status, app_lock # 确保这些全局变量可访问

    if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY": # 检查配置
        error_msg = "DeepSeek API Key 未配置。"
        print(f"DEEPSEEK_NLU_ERROR: {error_msg}")
        with app_lock:
            last_nlu_status = f"NLU失败: {error_msg}"
            last_nlu_result_text = "配置错误"
            # current_arm_state 的重置应由调用此函数的上层逻辑（如 process_audio_and_execute_command_task）处理
            # 以免在NLU作为更大流程一部分时过早改变状态
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    
    system_prompt = (
        "你是一个智能助手，用于解析用户对机械臂在指定工作台面上的操作指令。工作台面的坐标范围定义如下：左上角为 (-150, -100)，右下角为 (150, -260)。"
        "你需要从用户输入中提取关键信息，并以严格的JSON格式返回。JSON对象应包含以下字段：\n"
        "1. 'action': 字符串，表示用户的意图，可能的值为 'pick_and_place' (拾取并放置), 'pick' (仅拾取), 'inform' (一般对话或无法解析为动作)。\n"
        "2. 'pick_object': 字符串或null，用户想要拾取的物体名称。\n"
        "3. 'place_coordinates': 对象或null。如果需要放置，此对象应包含 'x' 和 'y' 两个数字类型的键，表示在工作台面上的目标放置坐标。例如 {\"x\": 0, \"y\": -200}。"
        "   - 如果用户描述“桌子中间”，可以理解为靠近 (0, -180) 的区域，可返回如 {\"x\": 0, \"y\": -180}。\n"
        "   - 如果用户描述“左上角”，可以理解为靠近 (-150, -100) 的区域，可返回如 {\"x\": -150, \"y\": -100}。\n"
        "   - 如果用户描述“左下角”，可以理解为靠近 (-150, -260) 的区域，可返回如 {\"x\": -150, \"y\": -260}。\n"
        "   - 如果用户描述“右下角”，可以理解为靠近 (150, -260) 的区域，可返回如 {\"x\": 150, \"y\": -260}。\n"
        "   - 如果用户说“放到X轴50，Y轴负100的位置”，则应返回 {\"x\": 50, \"y\": -100}。\n"
        "   - 确保返回的x,y坐标在 [-180, 180] 和 [-150, 150] 的大致范围内，并尽量使其合理。\n"
        "   - 如果用户没有指定放置位置，或者动作是 'pick' 或 'inform'，则此字段为 null。\n"
        "4. 'feedback_text': 字符串，一句友好且确认操作的回复给用户。\n"
        "例如，如果用户说“帮我把苹果放到桌子中间”，你应该返回类似（具体坐标可能略有不同）：\n"
        "{\"action\": \"pick_and_place\", \"pick_object\": \"苹果\", \"place_coordinates\": {\"x\": 0, \"y\": -190}, \"feedback_text\": \"好的，正在帮您把苹果放到桌子中间。\"}\n"
        "如果用户说“拿起那个杯子”，你应该返回：\n"
        "{\"action\": \"pick\", \"pick_object\": \"杯子\", \"place_coordinates\": null, \"feedback_text\": \"好的，准备拿起那个杯子。\"}\n"
        "请确保返回的 JSON 格式严格正确，不要包含任何额外的解释或文本在 JSON 对象之外。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_input}
    ]
    data = {
        "model": DEEPSEEK_MODEL, 
        "messages": messages, 
        "stream": False, 
        "temperature": 0.1, # 较低的温度使输出更具确定性
        "max_tokens": 250   # 限制输出长度，以防模型输出过多无关内容
    }

    try:
        # 状态的设置（如 ARM_STATE_UNDERSTANDING_NLU）应该在调用此函数之前，
        # 在 process_audio_and_execute_command_task 中进行，以便更清晰地管理主流程状态。
        # 此函数主要负责调用和解析。
        with app_lock: # 更新状态文本
            last_nlu_status = "正在调用NLU服务..." # 临时状态

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=25)
        response.raise_for_status() # 如果HTTP状态码是4xx或5xx，则引发HTTPError
        response_json = response.json()
        print(f"DEEPSEEK_NLU_RESPONSE (RAW): {response_json}") # 打印原始响应

        if response_json and response_json.get("choices") and \
           len(response_json["choices"]) > 0 and response_json["choices"][0].get("message"):
            content_str = response_json["choices"][0]["message"].get("content")
            if content_str:
                print(f"DEEPSEEK_NLU_CONTENT_STR (Before strip): '{content_str}'")
                
                pure_json_str = content_str.strip()
                if pure_json_str.startswith("```json"):
                    pure_json_str = pure_json_str[len("```json"):].strip()
                elif pure_json_str.startswith("```"): # 有些模型可能只返回 ``` 而没有 json 标识
                    pure_json_str = pure_json_str[len("```"):].strip()
                
                if pure_json_str.endswith("```"):
                    pure_json_str = pure_json_str[:-len("```")].strip()
                
                pure_json_str = pure_json_str.strip() # 确保没有前后空字符
                print(f"DEEPSEEK_NLU_PURE_JSON_STR (After strip): '{pure_json_str}'")

                try:
                    parsed_nlu_result = json.loads(pure_json_str)
                    
                    # 验证核心字段是否存在
                    required_keys = ["action", "pick_object", "place_coordinates", "feedback_text"]
                    if not all(k in parsed_nlu_result for k in required_keys):
                        raise ValueError(f"NLU JSON缺少必要字段，需要: {required_keys}")

                    # 进一步验证 place_coordinates 的结构 (如果 action 需要它)
                    action_val = parsed_nlu_result.get("action")
                    place_coords_val = parsed_nlu_result.get("place_coordinates")

                    if action_val in ["pick_and_place", "place"]: # "place" 动作也可能需要坐标
                        if place_coords_val is None: # 如果是这些动作，但坐标是null，也算一种有效情况（例如让用户后续指定）
                            pass # 或者可以认为这是一种不完整指令，但按prompt设计，null是允许的
                        elif not (isinstance(place_coords_val, dict) and \
                                  "x" in place_coords_val and "y" in place_coords_val and \
                                  isinstance(place_coords_val["x"], (int, float)) and \
                                  isinstance(place_coords_val["y"], (int, float))):
                            raise ValueError("NLU JSON中 'place_coordinates' 格式不正确，应为包含x, y数字的对象或null。")
                    elif place_coords_val is not None: # 对于 pick 或 inform，place_coordinates 应该是 null
                         # 可以选择在这里警告或修正，但prompt已指明为null
                         print(f"DEEPSEEK_NLU_WARN: Action is '{action_val}' but 'place_coordinates' is not null: {place_coords_val}")
                         # parsed_nlu_result["place_coordinates"] = None # 可选：强制修正

                    with app_lock:
                        last_nlu_result_text = parsed_nlu_result.get("feedback_text", "理解成功，但无反馈文本。")
                        last_nlu_status = "指令理解成功"
                    return parsed_nlu_result
                
                except (json.JSONDecodeError, ValueError) as e_parse:
                    print(f"DEEPSEEK_NLU_ERROR: 解析模型输出为JSON失败: {e_parse}. Cleaned content: '{pure_json_str}'")
                    with app_lock:
                        last_nlu_result_text = f"理解结果格式错误: {e_parse}"
                        last_nlu_status = "NLU结果格式错误"
                    # 返回一个 inform 类型的默认结果
                    return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": f"抱歉，我解析指令时遇到了格式问题。"}
            else: # content_str 为空或None
                print(f"DEEPSEEK_NLU_ERROR: 'content' 字段为空或不存在。")
                with app_lock:
                    last_nlu_result_text = "NLU服务未返回有效内容。"
                    last_nlu_status = "NLU无有效内容"
                return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": "抱歉，NLU服务未返回有效内容。"}

        # 如果 choices 或 message 结构不符合预期
        error_msg = "NLU响应格式不符合预期 (choices或message字段缺失)。"
        print(f"DEEPSEEK_NLU_ERROR: {error_msg} Response: {response_json}")
        with app_lock:
            last_nlu_result_text = error_msg
            last_nlu_status = "NLU响应格式错误"
        return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": f"抱歉，我暂时无法处理您的请求（响应结构错误）。"}

    except requests.exceptions.RequestException as e:
        print(f"DEEPSEEK_NLU_ERROR: 请求失败: {e}")
        with app_lock:
            last_nlu_result_text = "网络请求失败。"
            last_nlu_status = "NLU网络错误"
        return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": "抱歉，我现在无法连接到理解服务。"}
    except Exception as e:
        print(f"DEEPSEEK_NLU_ERROR: 未知错误: {e}")
        import traceback
        traceback.print_exc() 
        with app_lock:
            last_nlu_result_text = "发生未知内部错误。"
            last_nlu_status = "NLU未知错误"
        return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": "抱歉，处理您的指令时发生了内部错误。"}

# --- 任务线程 (Pick and Place) ---
def threaded_pick_and_place_task(pick_object_name_from_tpu, pixel_x, pixel_y_original, place_location_coords_xyz=None): # 参数名修改
    global current_arm_state, current_pump_status, my_arm_instance, app_lock, PREDEFINED_PLACE_LOCATIONS
    
    # pick_object_name_from_tpu 是经过TPU确认的物体名称
    print(f"  PICK_PLACE_START: Task for '{pick_object_name_from_tpu}' to {place_location_coords_xyz if place_location_coords_xyz else 'default drop'}.")
    
    pixel_y_adjusted = pixel_y_original - 20 
    try:
        # --- 拾取阶段 (逻辑不变) ---
        with app_lock: current_arm_state = ARM_STATE_PICKING_MOVE_TO_SUCK
        print(f"  PICK_PLACE: Moving to suck '{pick_object_name_from_tpu}' at ({pixel_x}, {pixel_y_adjusted}), height {PICK_HEIGHT}")
        _, final_coord_suck = my_arm_instance.move_pixel(pixel_x, pixel_y_adjusted, PICK_HEIGHT)
        time.sleep(0.5)

        with app_lock: current_arm_state = ARM_STATE_PICKING_SUCKING
        my_arm_instance.suck_up()
        with app_lock: current_pump_status = "ON"
        time.sleep(0.8) 

        with app_lock: current_arm_state = ARM_STATE_PICKING_LIFTING
        lift_height = final_coord_suck.z + 80
        my_arm_instance.move_point(final_coord_suck.x, final_coord_suck.y, lift_height, 800)
        time.sleep(0.5)

        # --- 放置阶段 ---
        # place_location_coords_xyz 已经是包含 x, y, z 的字典或 None
        if place_location_coords_xyz and all(k in place_location_coords_xyz for k in ["x", "y", "z"]):
            target_drop_x = place_location_coords_xyz["x"]
            target_drop_y = place_location_coords_xyz["y"]
            target_drop_z = place_location_coords_xyz["z"] # 这个 z 是计算好的悬停高度
            print(f"  PICK_PLACE: Target place coordinates (from NLU/default): ({target_drop_x}, {target_drop_y}, {target_drop_z})")
        else: # 如果是 pick_only 或者 NLU 没给有效坐标，则使用默认放置区
            default_loc = PREDEFINED_PLACE_LOCATIONS["默认放置区"]
            target_drop_x, target_drop_y, target_drop_z = default_loc["x"], default_loc["y"], default_loc["z"]
            print(f"  PICK_PLACE: No specific place coords from NLU or pick_only. Using default drop-off area.")


        with app_lock: current_arm_state = ARM_STATE_MOVING_TO_PLACE
        print(f"  PICK_PLACE: Moving to drop-off point ({target_drop_x}, {target_drop_y}, {target_drop_z})")
        my_arm_instance.move_point(target_drop_x, target_drop_y, max(lift_height, target_drop_z + 50) , 1200) 
        time.sleep(0.2)
        my_arm_instance.move_point(target_drop_x, target_drop_y, target_drop_z, 800)
        time.sleep(0.5)
        
        with app_lock: current_arm_state = ARM_STATE_PLACING_OBJECT
        print(f"  PICK_PLACE: Releasing object at ({target_drop_x}, {target_drop_y}, {target_drop_z}).")
        my_arm_instance.suck_release()
        with app_lock: current_pump_status = "OFF"
        time.sleep(0.8)

        print(f"  PICK_PLACE: Lifting after release and homing.")
        my_arm_instance.move_point(target_drop_x, target_drop_y, target_drop_z + 80, 500)
        time.sleep(0.2)

        with app_lock: current_arm_state = ARM_STATE_PICKING_HOMING_AFTER
        my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, 1000)
        print(f"  PICK_PLACE_SUCCESS: '{pick_object_name_from_tpu}' pick and place completed, arm homed.")
    except Exception as e_pick_place:
        print(f"  PICK_PLACE_ERROR: {e_pick_place}")
        try: 
            my_arm_instance.suck_release()
            with app_lock: current_pump_status = "OFF"
            my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, 1500)
        except Exception as e_recovery: print(f"  PICK_PLACE_RECOVERY_ERROR: {e_recovery}")
    finally:
        with app_lock: current_arm_state = ARM_STATE_IDLE
        print(f"  PICK_PLACE_FINALLY: Task finished. Arm state IDLE.")

# --- 主处理线程 (ASR -> NLU -> Action) ---
# test.py (process_audio_and_execute_command_task 函数修改)

def process_audio_and_execute_command_task(audio_filepath_original):
    global current_arm_state, app_lock, last_voice_command_text, last_nlu_result_text, last_nlu_status, \
           detected_objects_list, TABLE_Z_PLANE, DROP_HEIGHT_OFFSET, PREDEFINED_PLACE_LOCATIONS
    
    try:
        # 1. ASR (逻辑不变)
        with app_lock:
            current_arm_state = ARM_STATE_RECOGNIZING_VOICE
            last_nlu_status = "正在识别语音..."
        recognized_text = recognize_speech_youdao(audio_filepath_original)
        
        if not recognized_text:
            with app_lock:
                if current_arm_state == ARM_STATE_RECOGNIZING_VOICE: current_arm_state = ARM_STATE_IDLE
            return

        # 2. NLU (逻辑不变，但期望的输出格式变了)
        with app_lock:
            current_arm_state = ARM_STATE_UNDERSTANDING_NLU
            last_nlu_status = "正在理解指令..."
        nlu_output = call_deepseek_nlu(recognized_text)

        if not nlu_output:
            with app_lock:
                if current_arm_state == ARM_STATE_UNDERSTANDING_NLU: current_arm_state = ARM_STATE_IDLE
            return

        # 3. 根据NLU结果执行动作
        action = nlu_output.get("action")
        pick_object_name_from_nlu = nlu_output.get("pick_object")
        # place_location_desc_from_nlu 不再直接使用，改为 place_coordinates_from_nlu
        place_coordinates_from_nlu = nlu_output.get("place_coordinates") # 这应该是一个 {"x": val, "y": val} 或 null
        
        # feedback_text_from_nlu 已在 call_deepseek_nlu 中更新到 last_nlu_result_text

        if action == "pick_and_place" or action == "pick":
            if not pick_object_name_from_nlu:
                with app_lock:
                    last_nlu_status = "NLU结果: 未指定拾取物体"
                    current_arm_state = ARM_STATE_IDLE
                return

            object_instance_to_pick = None
            with app_lock:
                if current_arm_state != ARM_STATE_IDLE and current_arm_state != ARM_STATE_UNDERSTANDING_NLU :
                    print(f"  NLU_ACTION_WARN: Arm not IDLE ({current_arm_state}) before pick action.")
                    last_nlu_status = f"机械臂忙,无法执行"
                    return

                best_conf = -1.0
                temp_list_detected = list(detected_objects_list) # 使用副本进行迭代
                for obj_detected in temp_list_detected:
                    if pick_object_name_from_nlu.lower() in obj_detected['name'].lower() or \
                       obj_detected['name'].lower() in pick_object_name_from_nlu.lower():
                        if obj_detected['confidence'] > best_conf:
                            best_conf = obj_detected['confidence']
                            object_instance_to_pick = obj_detected
                
                if object_instance_to_pick:
                    current_arm_state = ARM_STATE_PICKING_SUBMITTED
                    last_nlu_status = f"准备拾取'{object_instance_to_pick['name']}'"
                    print(f"  NLU_ACTION: Found '{object_instance_to_pick['name']}' via TPU. Submitting task.")
                else:
                    last_nlu_status = f"未在画面中找到'{pick_object_name_from_nlu}'"
                    current_arm_state = ARM_STATE_IDLE
                    return
            
            # --- 确定放置位置 ---
            final_place_coords_for_arm = None # 包含 x, y, z
            if action == "pick_and_place":
                if place_coordinates_from_nlu and isinstance(place_coordinates_from_nlu, dict) and \
                   "x" in place_coordinates_from_nlu and "y" in place_coordinates_from_nlu:
                    
                    # 从NLU获取x, y，并计算z
                    place_x = place_coordinates_from_nlu["x"]
                    place_y = place_coordinates_from_nlu["y"]
                    
                    # 边界检查 (可选，但推荐)
                    # 使用您定义的桌子边界
                    # min_x, min_y = -180, -150
                    # max_x, max_y = 180, 150
                    
                    # place_x = max(min_x, min(place_x, max_x))
                    # place_y = max(min_y, min(place_y, max_y))

                    place_z = TABLE_Z_PLANE + DROP_HEIGHT_OFFSET # 使用全局定义的放置高度相关参数
                    
                    final_place_coords_for_arm = {"x": place_x, "y": place_y, "z": place_z}
                    print(f"  NLU_ACTION: NLU provided place (x,y): ({place_coordinates_from_nlu['x']}, {place_coordinates_from_nlu['y']}). Arm target: {final_place_coords_for_arm}")
                else: # NLU 未提供有效坐标，使用默认放置区
                    print(f"  NLU_ACTION_WARN: NLU did not provide valid place_coordinates. Using default.")
                    default_loc = PREDEFINED_PLACE_LOCATIONS["默认放置区"]
                    final_place_coords_for_arm = {"x": default_loc["x"], "y": default_loc["y"], "z": default_loc["z"]}
                    with app_lock: last_nlu_result_text += " (将放置在默认区域)"
            
            if object_instance_to_pick:
                task_thread = threading.Thread(
                    target=threaded_pick_and_place_task, 
                    args=(
                        object_instance_to_pick['name'], 
                        object_instance_to_pick['box_center_x'], 
                        object_instance_to_pick['box_center_y'],
                        final_place_coords_for_arm # 传递包含x,y,z的字典，如果是pick_only则为None
                    ), daemon=True)
                task_thread.start()

        elif action == "inform" or action is None:
            with app_lock: current_arm_state = ARM_STATE_IDLE
        else:
            with app_lock:
                last_nlu_status = f"未知NLU动作: {action}"
                current_arm_state = ARM_STATE_IDLE
    
    finally:
        if os.path.exists(audio_filepath_original):
            try: os.remove(audio_filepath_original)
            except Exception as e_del: print(f"删除原始上传文件失败: {e_del}")
        
        with app_lock:
            if current_arm_state not in [
                ARM_STATE_PICKING_SUBMITTED, ARM_STATE_PICKING_MOVE_TO_SUCK, 
                ARM_STATE_PICKING_LIFTING, ARM_STATE_MOVING_TO_PLACE, 
                ARM_STATE_PLACING_OBJECT, ARM_STATE_PICKING_HOMING_AFTER,
                ARM_STATE_RUNNING_TRAJECTORY, ARM_STATE_HOMING, ARM_STATE_MANUAL_MOVE
            ]: # 如果不是在一个明确的活动状态中
                 if current_arm_state != ARM_STATE_IDLE:
                    print(f"  PROCESS_AUDIO_FINALLY: Resetting state from {current_arm_state} to IDLE.")
                    current_arm_state = ARM_STATE_IDLE


# --- STM32 Reader Thread (原样保留) ---
def stm32_reader_thread_func(ser):
    global stop_threads_flag, stm32_current_light_status, current_arm_state, current_trajectory_index, was_running_when_homing_triggered
    print("[STM32 RX Thread] Started for WebUI.")
    buffer = b""
    while not stop_threads_flag:
        try:
            if ser and ser.is_open and ser.in_waiting > 0:
                data_bytes = ser.read(ser.in_waiting)
                if data_bytes:
                    buffer += data_bytes
                    while b'\n' in buffer:
                        line_bytes, buffer = buffer.split(b'\n', 1)
                        line_bytes = line_bytes.rstrip(b'\r')
                        try:
                            decoded_line = line_bytes.decode('utf-8', errors='replace').strip()
                            if decoded_line and decoded_line.startswith("STM32_LIGHT_STATUS:"):
                                new_status = decoded_line.split(":", 1)[1].strip()
                                with app_lock:
                                    if new_status != stm32_current_light_status:
                                        stm32_current_light_status = new_status
                                        print(f"[STM32 Event] Light status changed to {new_status}")
                                        if new_status == "RED_ON":
                                            if current_arm_state != ARM_STATE_HOMING :
                                                if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY:
                                                    was_running_when_homing_triggered = True
                                                else:
                                                    was_running_when_homing_triggered = False
                                                current_arm_state = ARM_STATE_HOMING
                                        elif new_status == "GREEN_ON":
                                            if current_arm_state in [ARM_STATE_IDLE, ARM_STATE_PAUSED_TRAJECTORY, ARM_STATE_UNDERSTANDING_NLU, ARM_STATE_RECOGNIZING_VOICE] or \
                                               (current_arm_state == ARM_STATE_HOMING and not was_running_when_homing_triggered):
                                                current_arm_state = ARM_STATE_RUNNING_TRAJECTORY
                                                current_trajectory_index = 0
                                                was_running_when_homing_triggered = False
                                        elif new_status == "YELLOW_ON":
                                            if current_arm_state == ARM_STATE_PAUSED_TRAJECTORY:
                                                if current_trajectory_index < TOTAL_TRAJECTORY_POINTS:
                                                    current_arm_state = ARM_STATE_RUNNING_TRAJECTORY
                                                    was_running_when_homing_triggered = False
                                                else: # Trajectory finished, paused at end
                                                    current_arm_state = ARM_STATE_IDLE 
                                                    current_trajectory_index = 0
                                        elif new_status == "ALL_OFF":
                                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY:
                                                current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
                        except Exception as decode_err:
                            print(f"[STM32 RX Thread WebUI] Decode Error: {decode_err}, Raw: {line_bytes!r}")
            else:
                time.sleep(0.02)
        except serial.SerialException:
            if not stop_threads_flag: print(f"[STM32 RX Thread WebUI] SerialException.")
            break
        except Exception as e:
            if not stop_threads_flag: print(f"[STM32 RX Thread WebUI] Error: {e}.")
            break
    print("[STM32 RX Thread WebUI] Stopped.")

# --- Object Detection Task (原样保留, tpu_instance.predict 在这里) ---
def object_detection_task():
    global detected_objects_list, tpu_instance, camera, app_lock
    print("[Object Detection Task WebUI] Started.")
    frame_counter = 0
    DETECTION_INTERVAL = 3
    
    while not stop_threads_flag:
        if not DETECTION_ENABLED or tpu_instance is None:
            time.sleep(0.5)
            continue
        
        original_frame = None
        with camera_lock:
            if camera and camera.isOpened():
                ret, frame = camera.read()
                if ret:
                    original_frame = frame
            else:
                time.sleep(1)
                continue
        
        if original_frame is None:
            time.sleep(0.1)
            continue

        with app_lock:
            is_arm_idle_for_detection = current_arm_state == ARM_STATE_IDLE
        if not is_arm_idle_for_detection:
            time.sleep(0.2)
            continue

        frame_counter += 1
        if frame_counter % DETECTION_INTERVAL != 0:
            time.sleep(0.03)
            continue

        try:
            boxes = tpu_instance.predict(
                original_frame,
                mode=TPU_REMOTE_MODE,
                model_name=TPU_MODEL_NAME,
                iou=TPU_IOU_THRES,
                conf=TPU_CONF_THRES,
                w=TPU_IMAGE_WIDTH,
                h=TPU_IMAGE_HEIGHT,
            )
            
            current_detected = []
            if boxes:
                actual_height, actual_width, _ = original_frame.shape
                target_h, target_w = TPU_IMAGE_HEIGHT, TPU_IMAGE_WIDTH
                
                pad_y = (target_h - actual_height) // 2
                pad_x = (target_w - actual_width) // 2
                
                for i, box_data in enumerate(boxes):
                    cx_padded, cy_padded, w_padded, h_padded, score, label_name = box_data
                    cx_corrected = cx_padded - pad_x
                    cy_corrected = cy_padded - pad_y
                    
                    if not (0 <= cx_corrected < actual_width and 0 <= cy_corrected < actual_height):
                        continue

                    current_detected.append({
                        "id": f"obj_{time.time_ns()}_{i}",
                        "name": label_name,
                        "label_id": -1,
                        "confidence": float(score),
                        "box_center_x": int(cx_corrected),
                        "box_center_y": int(cy_corrected),
                        "box_width": int(w_padded),
                        "box_height": int(h_padded),
                        "timestamp": time.time()
                    })
            
            with app_lock:
                detected_objects_list = current_detected
        except Exception as e:
            print(f"[Object Detection Task WebUI] Error during prediction: {e}")
            import traceback
            traceback.print_exc()
            with app_lock:
                detected_objects_list = []
    print("[Object Detection Task WebUI] Stopped.")

# --- Arm Background Task (原样保留) ---
def arm_background_task():
    global current_arm_state, current_trajectory_index, was_running_when_homing_triggered, app_lock, my_arm_instance, stop_threads_flag
    print("[Arm BG Task WebUI] Started.")
    while not stop_threads_flag:
        state_to_process = None
        with app_lock:
            state_to_process = current_arm_state # Get current state under lock
        
        if my_arm_instance:
            if state_to_process == ARM_STATE_HOMING:
                print("  BG_TASK: Arm state is HOMING. Executing reset...");
                try:
                    my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, ARM_RESET_TIME)
                    print("  BG_TASK: Arm homing complete.")
                except Exception as e_home:
                    print(f"  BG_TASK_ERROR: Homing failed: {e_home}")
                with app_lock: # Update state under lock
                    if was_running_when_homing_triggered and stm32_current_light_status == "RED_ON": # Check light status also under lock
                        current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
                        print(f"  BG_TASK: Homing (RED) done. State set to PAUSED.")
                    else:
                        current_arm_state = ARM_STATE_IDLE
                        current_trajectory_index = 0 
                        print("  BG_TASK: Homing done. State set to IDLE.")
                    was_running_when_homing_triggered = False # Reset flag
            
            elif state_to_process == ARM_STATE_RUNNING_TRAJECTORY:
                can_run_step, point_data, temp_traj_index = False, None, -1
                with app_lock: # Access trajectory data under lock
                    if current_trajectory_index < TOTAL_TRAJECTORY_POINTS:
                        can_run_step = True
                        point_data = PREDEFINED_TRAJECTORY[current_trajectory_index]
                        temp_traj_index = current_trajectory_index
                    else: # Trajectory completed
                        print("  BG_TASK: Trajectory fully completed.");
                        current_arm_state = ARM_STATE_IDLE
                        current_trajectory_index = 0
                
                if can_run_step and point_data:
                    target_x, target_y, target_z, move_duration = point_data
                    print(f"  BG_TASK_TRAJ: Moving to point {temp_traj_index+1}/{TOTAL_TRAJECTORY_POINTS}: ({target_x}, {target_y}, {target_z})")
                    try:
                        my_arm_instance.move_point(target_x, target_y, target_z, move_duration)
                        with app_lock: # Update index under lock IF state hasn't changed by external event
                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY and current_trajectory_index == temp_traj_index:
                                current_trajectory_index += 1
                                if current_trajectory_index >= TOTAL_TRAJECTORY_POINTS:
                                    print("  BG_TASK_TRAJ: Trajectory fully completed (last point).")
                                    current_arm_state = ARM_STATE_IDLE
                                    current_trajectory_index = 0
                    except Exception as e_traj:
                        print(f"  BG_TASK_ERROR: Moving to trajectory point {temp_traj_index + 1} failed: {e_traj}")
                        with app_lock: # On error, pause
                            current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
            # Other states like PICKING_*, PLACING_* are handled by their own threads
        time.sleep(0.1) # Loop delay
    print("[Arm BG Task WebUI] Stopped.")

# --- Generate Frames (原样保留) ---
def generate_frames():
    global camera, detected_objects_list, pil_font, app_lock, camera_lock
    if pil_font is None or pil_font == "FONT_LOAD_FAILED":
        try:
            pil_font = ImageFont.truetype(CHINESE_FONT_PATH, CHINESE_FONT_SIZE)
            print(f"成功加载中文字体: {CHINESE_FONT_PATH}")
        except IOError:
            print(f"!!! 错误: 无法加载中文字体 '{CHINESE_FONT_PATH}'.")
            pil_font = "FONT_LOAD_FAILED"
        except Exception as e:
            print(f"!!! 错误: 加载字体时发生未知错误: {e}")
            pil_font = "FONT_LOAD_FAILED"
    while not stop_threads_flag:
        frame_to_yield = None
        try:
            current_boxes_for_overlay = []
            with app_lock:
                current_boxes_for_overlay = [(obj['box_center_x'], obj['box_center_y'], obj['box_width'], obj['box_height'], obj['name'], obj['confidence']) for obj in detected_objects_list]
            with camera_lock:
                if camera is None or not camera.isOpened():
                    try:
                        camera = cv2.VideoCapture(CAMERA_INDEX)
                        if not camera.isOpened():
                            print("WARN: Camera not opened, sleeping.")
                            time.sleep(0.5)
                            continue
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        print("INFO: Camera re-initialized.")
                    except Exception as e_cam_init:
                        print(f"CRITICAL: Camera re-init error: {e_cam_init}")
                        camera = None
                        time.sleep(1)
                        continue
                success, frame_bgr_original = camera.read()
                if not success or frame_bgr_original is None:
                    time.sleep(0.05)
                    continue
            processed_frame = frame_bgr_original.copy()
            if pil_font and pil_font != "FONT_LOAD_FAILED":
                try:
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_image)
                    for (x_center, y_center, w, h, name, score) in current_boxes_for_overlay:
                        x1, y1, x2, y2 = int(x_center - w / 2), int(y_center - h / 2), int(x_center + w / 2), int(y_center + h / 2)
                        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
                        text_y = y1 - CHINESE_FONT_SIZE - 2 if y1 - CHINESE_FONT_SIZE - 2 >=0 else y1 + 2
                        draw.text((x1, text_y), f"{name}: {score:.2f}", font=pil_font, fill=(0, 255, 0))
                    processed_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e_pil_draw:
                    print(f"ERROR during Pillow drawing: {e_pil_draw}")
            else:
                for (x_center, y_center, w, h, name, score) in current_boxes_for_overlay:
                    x1, y1, x2, y2 = int(x_center - w / 2), int(y_center - h / 2), int(x_center + w / 2), int(y_center + h / 2)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"{name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    frame_to_yield = buffer.tobytes()
        except Exception as e_outer:
            print(f"CRITICAL ERROR in generate_frames loop: {e_outer}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
        if frame_to_yield:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')
        time.sleep(1/30)

def send_command_to_stm32(ser, command_str):
    if ser and ser.is_open:
        try:
            ser.write(f"{command_str}\n".encode('utf-8'))
            print(f"[STM32 TX] Sent: {command_str}")
            return True
        except serial.SerialException as e:
            print(f"[STM32 TX Error] Failed to send '{command_str}': {e}")
            return False
    return False


# --- send_command_to_stm32 (原样保留) ---
def send_command_to_stm32(ser, command_str):
    if ser and ser.is_open:
        try:
            ser.write(f"{command_str}\n".encode('utf-8'))
            print(f"[STM32 TX] Sent: {command_str}")
            return True
        except serial.SerialException as e:
            print(f"[STM32 TX Error] Failed to send '{command_str}': {e}")
            return False
    return False

# --- Flask Routes ---
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_voice_audio', methods=['POST'])
def upload_voice_audio_route():
    global current_arm_state, app_lock, last_voice_command_text, last_nlu_status
    
    if 'audio_data' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio data in request.'}), 400
    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

    # Use a fixed temporary filename for the original upload
    # The extension here might not match the actual content, ffmpeg will handle it
    original_save_path = WAVE_OUTPUT_FILENAME 
    try:
        audio_file.save(original_save_path)
        print(f"AUDIO_UPLOAD: Original audio file saved to {original_save_path}")
    except Exception as e:
        print(f"AUDIO_UPLOAD_ERROR: Failed to save audio file: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to save audio: {e}'}), 500

    with app_lock:
        # Allow processing if IDLE or if a previous ASR/NLU attempt failed and reset to these states
        if current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_RECOGNIZING_VOICE, ARM_STATE_UNDERSTANDING_NLU]:
            message = f"机械臂忙 ({current_arm_state}), 无法处理语音。"
            if os.path.exists(original_save_path): # Clean up temp file
                try: os.remove(original_save_path)
                except Exception as e_del: print(f"AUDIO_CLEANUP_WARN: {e_del}")
            return jsonify({'status': 'error', 'message': message}), 409
        
        # State will be set to ARM_STATE_RECOGNIZING_VOICE inside process_audio_and_execute_command_task
        last_voice_command_text = "上传中..." # Indicate ASR is next
        last_nlu_status = "音频已接收，准备识别..."
        message_to_client = "音频已接收，正在后台处理..."

    # Start the full ASR -> NLU -> Action pipeline in a thread
    voice_processing_thread = threading.Thread(
        target=process_audio_and_execute_command_task, 
        args=(original_save_path,), # Pass path of the originally saved (possibly .webm) file
        daemon=True
    )
    voice_processing_thread.start()
    
    return jsonify({'status': 'success', 'message': message_to_client}), 202


@app.route('/control', methods=['POST'])
def control_arm_route():
    global current_arm_state, current_trajectory_index, detected_objects_list, app_lock, \
           last_voice_command_text, last_nlu_status, last_nlu_result_text, \
           was_running_when_homing_triggered, current_pump_status, current_pump_angle, my_arm_instance
           
    data = request.get_json()
    command = data.get('command')
    payload = data.get('payload', {})
    message = f"CMD_RECV: {command}"
    success = True
    status_code = 200
    
    if command == 'process_voice_text': # 模拟文本指令输入
        text_command = payload.get('text', '').strip()
        if not text_command:
            return jsonify({'status': 'error', 'message': '模拟指令文本不能为空。'}), 400
        
        print(f"  SIM_NLU_CTRL: Received simulated text: '{text_command}'")
        
        # --- 开始NLU处理和动作执行 (与 process_audio_and_execute_command_task 后半部分类似) ---
        # 这个处理过程也可能耗时，理想情况下也应放入线程，但为简化，这里暂时同步处理
        # 如果需要异步，可以创建一个新的线程函数包装以下逻辑

        with app_lock:
            if current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_UNDERSTANDING_NLU]: # 允许从这些状态开始
                message_to_client = f"机械臂忙 ({current_arm_state}), 无法处理模拟指令。"
                return jsonify({'status': 'error', 'message': message_to_client}), 409
            
            last_voice_command_text = text_command # 显示用户输入的模拟文本
            current_arm_state = ARM_STATE_UNDERSTANDING_NLU
            last_nlu_status = "正在理解模拟指令..."
            last_nlu_result_text = "处理中..." # 清空之前的NLU结果

        nlu_output = call_deepseek_nlu(text_command)

        if not nlu_output: # NLU 调用本身失败或返回None
            with app_lock:
                if current_arm_state == ARM_STATE_UNDERSTANDING_NLU: current_arm_state = ARM_STATE_IDLE
                # last_nlu_status 和 last_nlu_result_text 已在 call_deepseek_nlu 中设置
                message_to_client = last_nlu_result_text or "NLU处理失败"
            return jsonify({'status': 'error', 'message': message_to_client}), 500

        # 从NLU结果获取信息
        action = nlu_output.get("action")
        pick_object_name_from_nlu = nlu_output.get("pick_object")
        place_coordinates_from_nlu = nlu_output.get("place_coordinates")
        
        # NLU的feedback_text已经在call_deepseek_nlu中更新到last_nlu_result_text
        # 我们将用它作为成功时的 message_to_client
        with app_lock:
            message_to_client = last_nlu_result_text # 使用NLU的反馈作为给客户端的消息

        if action == "pick_and_place" or action == "pick":
            if not pick_object_name_from_nlu:
                with app_lock:
                    last_nlu_status = "NLU结果: 未指定拾取物体"
                    # message_to_client 已被 last_nlu_result_text 更新，可能包含更详细的NLU反馈
                    current_arm_state = ARM_STATE_IDLE
                return jsonify({'status': 'error', 'message': message_to_client}), 400 # Bad request from NLU perspective

            object_instance_to_pick = None
            with app_lock:
                # 确保在NLU之后，执行实际动作前臂是空闲的 (或仍在NLU状态，将转为PICKING_SUBMITTED)
                if current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_UNDERSTANDING_NLU]:
                    print(f"  SIM_NLU_CTRL_WARN: Arm not IDLE/UNDERSTANDING_NLU ({current_arm_state}) before pick action.")
                    last_n_status_temp = f"机械臂忙 ({current_arm_state}),无法执行" # 临时变量
                    last_nlu_status = last_n_status_temp # 更新全局状态
                    return jsonify({'status': 'error', 'message': last_n_status_temp}), 409

                best_conf = -1.0
                temp_list_detected_sim = list(detected_objects_list) # Iterate over a copy
                for obj_detected in temp_list_detected_sim:
                    if pick_object_name_from_nlu.lower() in obj_detected['name'].lower() or \
                       obj_detected['name'].lower() in pick_object_name_from_nlu.lower():
                        if obj_detected['confidence'] > best_conf:
                            best_conf = obj_detected['confidence']
                            object_instance_to_pick = obj_detected
                
                if object_instance_to_pick:
                    current_arm_state = ARM_STATE_PICKING_SUBMITTED # 标记任务已提交
                    last_nlu_status = f"准备拾取'{object_instance_to_pick['name']}' (模拟)"
                    print(f"  SIM_NLU_CTRL: Found '{object_instance_to_pick['name']}' via TPU for simulated command. Submitting task.")
                else:
                    last_nlu_status = f"未在画面中找到'{pick_object_name_from_nlu}' (模拟)"
                    # message_to_client 应该已经是NLU的反馈了，可以补充
                    # message_to_client = f"未在画面中找到'{pick_object_name_from_nlu}'。{last_nlu_result_text}"
                    current_arm_state = ARM_STATE_IDLE # 回到空闲
                    return jsonify({'status': 'error', 'message': f"未在画面中找到'{pick_object_name_from_nlu}'。 " + message_to_client}), 404
            
            # --- 确定放置位置 (与 process_audio_and_execute_command_task 中相同) ---
            final_place_coords_for_arm = None
            if action == "pick_and_place":
                if place_coordinates_from_nlu and isinstance(place_coordinates_from_nlu, dict) and \
                   "x" in place_coordinates_from_nlu and "y" in place_coordinates_from_nlu:
                    place_x, place_y = place_coordinates_from_nlu["x"], place_coordinates_from_nlu["y"]
                    # min_x, min_y, max_x, max_y = -180, -150, 180, 150 # 桌子边界
                    # place_x = max(min_x, min(place_x, max_x))
                    # place_y = max(min_y, min(place_y, max_y))
                    place_z = TABLE_Z_PLANE + DROP_HEIGHT_OFFSET
                    final_place_coords_for_arm = {"x": place_x, "y": place_y, "z": place_z}
                else:
                    default_loc = PREDEFINED_PLACE_LOCATIONS["默认放置区"]
                    final_place_coords_for_arm = {"x": default_loc["x"], "y": default_loc["y"], "z": default_loc["z"]}
                    with app_lock: message_to_client += " (将放置在默认区域)" # 附加到NLU的反馈上
            
            if object_instance_to_pick:
                # 启动拾取（和放置）任务线程
                task_thread = threading.Thread(
                    target=threaded_pick_and_place_task, 
                    args=(
                        object_instance_to_pick['name'], 
                        object_instance_to_pick['box_center_x'], 
                        object_instance_to_pick['box_center_y'],
                        final_place_coords_for_arm 
                    ), daemon=True)
                task_thread.start()
                status_code = 202 # Accepted for processing
                # message_to_client 已被 NLU 的 feedback_text 设置
                success = True
        
        elif action == "inform" or action is None:
            print(f"  SIM_NLU_CTRL: Informational or unhandled action from simulated text. Feedback: {last_nlu_result_text}")
            with app_lock: current_arm_state = ARM_STATE_IDLE
            success = True # NLU处理了，但不是动作
            # message_to_client 已被 NLU 的 feedback_text 设置
        
        else: # 未知动作
            print(f"  SIM_NLU_CTRL_WARN: Unknown action '{action}' from NLU for simulated text.")
            with app_lock:
                last_nlu_status = f"未知NLU动作: {action} (模拟)"
                current_arm_state = ARM_STATE_IDLE
            message_to_client = f"我不理解这个动作：'{action}'。"
            success = False
            status_code = 400 # Bad request / unhandled action

        return jsonify({'status': 'success' if success else 'error', 'message': message_to_client}), status_code
        # --- End of synchronous NLU and action for simulated text ---

    elif command.startswith('set_'): # STM32 Light controls (原样保留)
        with app_lock:
            if command == 'set_green_on': success = send_command_to_stm32(stm32_serial_conn, "SET_GREEN_ON")
            elif command == 'set_red_on': success = send_command_to_stm32(stm32_serial_conn, "SET_RED_ON")
            elif command == 'set_yellow_on': success = send_command_to_stm32(stm32_serial_conn, "SET_YELLOW_ON")
            elif command == 'set_lights_off': success = send_command_to_stm32(stm32_serial_conn, "SET_LIGHTS_OFF")
            else: success = False; message = "Unknown set command"
        if not success: status_code = 500; message = f"Failed to send {command}"
        else: message = f"Command {command} sent to STM32."
        return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

    elif not my_arm_instance:
        return jsonify({'status': 'error', 'message': '机械臂未初始化!'}), 503

    elif command == 'pick_object': # UI initiated pick (bypasses NLU)
        pixel_x, pixel_y = payload.get('pixel_x'), payload.get('pixel_y')
        obj_name = payload.get('name', '未知物体')
        if pixel_x is None or pixel_y is None:
            return jsonify({'status': 'error', 'message': "未提供有效像素坐标."}), 400
        
        with app_lock:
            if current_arm_state != ARM_STATE_IDLE:
                return jsonify({'status': 'error', 'message': f"机械臂忙 ({current_arm_state})."}), 409
            current_arm_state = ARM_STATE_PICKING_SUBMITTED
            message = f"UI拾取 '{obj_name}' 已提交."
            last_nlu_status = "UI拾取" # Update status for UI pick
            last_nlu_result_text = f"正在通过UI指令拾取 {obj_name}。"

        # For UI pick, assume it's pick and drop at default location
        place_coords_ui = PREDEFINED_PLACE_LOCATIONS["默认放置区"]
        threading.Thread(target=threaded_pick_and_place_task, 
                         args=(obj_name, pixel_x, pixel_y, place_coords_ui), daemon=True).start()
        status_code = 202
        return jsonify({'status': 'success', 'message': message}), status_code
        
    else: # Manual arm controls and other tasks (原样保留，注意状态检查)
        action_taken = False
        with app_lock:
            # More states might prevent manual control if NLU/ASR is active
            arm_busy_for_manual = current_arm_state not in [
                ARM_STATE_IDLE, ARM_STATE_PAUSED_TRAJECTORY, 
                ARM_STATE_RECOGNIZING_VOICE, ARM_STATE_UNDERSTANDING_NLU # Allow manual ops even if ASR/NLU is stuck
            ] 
            if arm_busy_for_manual and command.startswith('manual_') and command != 'manual_reset':
                message = f"机械臂忙 ({current_arm_state}), 无法手动操作."
                return jsonify({'status': 'error', 'message': message}), 409

        if command == 'manual_reset':
            with app_lock:
                if current_arm_state != ARM_STATE_HOMING:
                    # Cancel any ongoing ASR/NLU if reset is hit
                    if current_arm_state in [ARM_STATE_RECOGNIZING_VOICE, ARM_STATE_UNDERSTANDING_NLU]:
                        last_nlu_status = "操作已取消 (复位)"
                        last_nlu_result_text = "用户触发复位。"
                    current_arm_state = ARM_STATE_HOMING
                    was_running_when_homing_triggered = False; current_trajectory_index = 0
                    message = "复位指令已提交."
                else: message = "机械臂已在复位中."
            action_taken = True
        elif command.startswith('manual_'):
            # (Manual move logic: x_plus, x_minus, etc. - 原样保留)
            original_state_before_manual = ""
            with app_lock:
                original_state_before_manual = current_arm_state
                current_arm_state = ARM_STATE_MANUAL_MOVE
            try:
                if command == 'manual_x_plus': my_arm_instance.move_axis_point(0, ARM_TRANSLATE_X_STEP, ARM_MOVE_TIME); message = "Arm X+"
                elif command == 'manual_x_minus': my_arm_instance.move_axis_point(0, -ARM_TRANSLATE_X_STEP, ARM_MOVE_TIME); message = "Arm X-"
                elif command == 'manual_y_plus': my_arm_instance.move_axis_point(1, ARM_TRANSLATE_Y_STEP, ARM_MOVE_TIME); message = "Arm Y+"
                elif command == 'manual_y_minus': my_arm_instance.move_axis_point(1, -ARM_TRANSLATE_Y_STEP, ARM_MOVE_TIME); message = "Arm Y-"
                elif command == 'manual_z_plus': my_arm_instance.move_axis_point(2, ARM_TRANSLATE_Z_STEP, ARM_MOVE_TIME); message = "Arm Z+"
                elif command == 'manual_z_minus': my_arm_instance.move_axis_point(2, -ARM_TRANSLATE_Z_STEP, ARM_MOVE_TIME); message = "Arm Z-"
                action_taken = True
            except Exception as e_manual: message = f"手动操作错误: {e_manual}"; success = False; status_code = 500
            finally:
                with app_lock: # Restore previous state if it wasn't IDLE (e.g. PAUSED)
                    if current_arm_state == ARM_STATE_MANUAL_MOVE:
                        current_arm_state = ARM_STATE_IDLE if original_state_before_manual not in [ARM_STATE_PAUSED_TRAJECTORY] else original_state_before_manual
        elif command.startswith('pump_'):
            # (Pump control logic - 原样保留)
            with app_lock:
                if command == 'pump_on': my_arm_instance.suck_up(); current_pump_status = "ON"
                elif command == 'pump_off': my_arm_instance.suck_release(); current_pump_status = "OFF"
                elif command == 'pump_angle_set':
                    angle = payload.get('angle', current_pump_angle); my_arm_instance.rotate_angle(angle); current_pump_angle = angle
                elif command == 'pump_rotate_left_small':
                    new_angle = max(0, current_pump_angle - PUMP_ROTATE_SMALL_STEP); my_arm_instance.rotate_angle(new_angle); current_pump_angle = new_angle
                elif command == 'pump_rotate_right_small':
                    new_angle = min(180, current_pump_angle + PUMP_ROTATE_SMALL_STEP); my_arm_instance.rotate_angle(new_angle); current_pump_angle = new_angle
                message = f"PUMP_CMD: {command} executed. Angle: {current_pump_angle}, Status: {current_pump_status}"
            action_taken = True
        elif command == 'task_pause': # Soft pause from UI (原样保留)
            with app_lock:
                if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY: current_arm_state = ARM_STATE_PAUSED_TRAJECTORY; message = "TASK_PAUSED"
                else: success = False; message = "无法暂停: 非轨迹运行中."
            action_taken = True
        elif command == 'task_stop': # Soft stop from UI (原样保留)
            with app_lock:
                if current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_HOMING]:
                    current_arm_state = ARM_STATE_HOMING
                    was_running_when_homing_triggered = False; current_trajectory_index = 0
                    message = "TASK_STOP_HOMING: 停止并归位."
                else: message = "任务已停止或归位中."
            action_taken = True
        elif command == 'task_continue': # Soft continue from UI (原样保留)
            with app_lock:
                if current_arm_state == ARM_STATE_PAUSED_TRAJECTORY: current_arm_state = ARM_STATE_RUNNING_TRAJECTORY; message = "TASK_CONTINUED"
                else: success = False; message = "无法继续: 非暂停状态."
            action_taken = True
        
        if not action_taken and success: # Default for unhandled commands
            message = f"未知指令或状态不允许: {command}"
            success = False; status_code = 400
        return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

# --- /status Route (更新以包含NLU状态) ---
@app.route('/status')
def get_status():
    global last_voice_command_text, last_nlu_result_text, last_nlu_status, current_arm_state, \
           stm32_current_light_status, current_trajectory_index, TOTAL_TRAJECTORY_POINTS, \
           current_pump_status, current_pump_angle, detected_objects_list, my_arm_instance, app_lock
           
    with app_lock:
        arm_coords_str = "[N/A]"
        if my_arm_instance:
            try:
                pos_data = my_arm_instance.get_axis_point('3') # Assuming '3' gets [x,y,z]
                if pos_data and isinstance(pos_data, (list, tuple)) and len(pos_data) >= 3:
                    arm_coords_str = f"[{pos_data[0]:.1f}, {pos_data[1]:.1f}, {pos_data[2]:.1f}]"
            except Exception: arm_coords_str = "[Error Reading]"
        
        # Ensure detected_objects_list contains serializable data
        objects_to_send = []
        for obj in detected_objects_list:
            # Create a new dict with only necessary, serializable fields
            serializable_obj = {
                "id": obj.get("id"), "name": obj.get("name"),
                "confidence": obj.get("confidence"),
                "box_center_x": obj.get("box_center_x"), "box_center_y": obj.get("box_center_y"),
                "box_width": obj.get("box_width"), "box_height": obj.get("box_height")
            }
            objects_to_send.append(serializable_obj)

        status_data = {
            'arm_state': current_arm_state, 
            'stm32_light_status': stm32_current_light_status,
            'current_trajectory_index': current_trajectory_index, 
            'total_trajectory_points': TOTAL_TRAJECTORY_POINTS,
            'pump_status': current_pump_status, 
            'pump_angle': current_pump_angle, 
            'arm_coords': arm_coords_str,
            'detected_objects': objects_to_send, # Use the sanitized list
            'voice_command_text': last_voice_command_text, # Original ASR text
            'nlu_result_text': last_nlu_result_text,       # Feedback/main content from NLU
            'nlu_status': last_nlu_status,                 # Status of NLU processing
            'timestamp': time.time()
        }
    return jsonify(status_data)

# --- init_hardware_and_camera (更新状态变量名) ---
def init_hardware_and_camera():
    global my_arm_instance, stm32_serial_conn, stm32_reader_thread_obj, arm_bg_thread_obj, camera, \
           current_pump_status, current_pump_angle, tpu_instance, object_detection_thread_obj, \
           current_arm_state, current_trajectory_index, was_running_when_homing_triggered, pil_font, \
           last_voice_command_text, last_nlu_result_text, last_nlu_status # Updated state vars

    if pil_font is None:
        try:
            pil_font = ImageFont.truetype(CHINESE_FONT_PATH, CHINESE_FONT_SIZE)
            print(f"字体初始化: 成功加载中文字体: {CHINESE_FONT_PATH}")
        except Exception as e:
            print(f"!!! 字体初始化警告: 加载字体时发生错误: {e}")
            pil_font = "FONT_LOAD_FAILED"
    
    last_voice_command_text = "待识别..."
    last_nlu_result_text = "待理解..."
    last_nlu_status = "就绪"

    print("Initializing Arm for WebUI...")
    try:
        my_arm_instance = Arm()
        my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, ARM_RESET_TIME)
        with app_lock:
            current_arm_state = ARM_STATE_IDLE
            current_trajectory_index = 0
            was_running_when_homing_triggered = False
            current_pump_status = "OFF"
            current_pump_angle = 0
        print("Arm initialized and at home (WebUI).")
    except Exception as e:
        print(f"!!! Error initializing Arm (WebUI): {e} !!!"); my_arm_instance = None
    
    print(f"Connecting to STM32 ({STM32_SERIAL_PORT}) for WebUI...")
    try:
        stm32_serial_conn = serial.Serial(STM32_SERIAL_PORT, STM32_SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
        stm32_reader_thread_obj = threading.Thread(target=stm32_reader_thread_func, args=(stm32_serial_conn,), daemon=True)
        stm32_reader_thread_obj.start()
        print("STM32 serial connected and reader thread started for WebUI.")
    except serial.SerialException as e:
        print(f"!!! Failed to connect to STM32 (WebUI): {e} !!!"); stm32_serial_conn = None
    
    print(f"Initializing Camera (index: {CAMERA_INDEX}) for WebUI...")
    with camera_lock:
        try:
            camera = cv2.VideoCapture(CAMERA_INDEX)
            if not camera.isOpened():
                print(f"!!! Error: Could not open camera {CAMERA_INDEX} !!!"); camera = None
            else:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Camera {CAMERA_INDEX} initialized successfully.")
        except Exception as e: print(f"!!! Error initializing camera: {e} !!!"); camera = None
            
    if DETECTION_ENABLED:
        print("Initializing Control_TPU for WebUI detection...")
        try:
            tpu_instance = Control_TPU # Assuming Control_TPU is a class that can be instantiated
            print("Control_TPU instance created/assigned.")
            object_detection_thread_obj = threading.Thread(target=object_detection_task, daemon=True)
            object_detection_thread_obj.start()
            print("Object detection thread started for WebUI.")
        except Exception as e:
            print(f"!!! Error creating/starting Control_TPU or detection thread (WebUI): {e} !!!")
            tpu_instance = None; object_detection_thread_obj = None
    else:
        print("Object detection is DISABLED.")
        
    arm_bg_thread_obj = threading.Thread(target=arm_background_task, daemon=True)
    arm_bg_thread_obj.start()
    print("Arm background task thread started for WebUI.")

# --- Main Execution (原样保留，添加API Key检查) ---
if __name__ == '__main__':
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("错误：Pillow库未安装。请运行 'pip install Pillow' 来安装。"); exit(1)

    # API Key Checks
    if YOUDAO_APP_KEY == "YOUR_YOUDAO_APP_KEY" or YOUDAO_APP_SECRET == "YOUR_YOUDAO_APP_SECRET":
        print("\n警告: 请在 test.py 文件中设置您的有道智云 APP_KEY 和 APP_SECRET。\n")
    if DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY":
        print("警告: 请在 test.py 文件中设置您的 DEEPSEEK_API_KEY。\n")
        
    init_hardware_and_camera()
    print("Starting Flask web server on http://0.0.0.0:5000 ...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        print("Flask server shutting down...")
        stop_threads_flag = True
        print("Releasing camera...")
        with camera_lock:
            if camera and camera.isOpened(): camera.release(); camera = None
            print("Camera released.")
        
        threads_to_join = [
            (stm32_reader_thread_obj, "STM32 reader thread"), 
            (arm_bg_thread_obj, "Arm background task"), 
            (object_detection_thread_obj, "Object detection task") # Ensure this is joined
        ]
        for thread, name in threads_to_join:
            if thread and thread.is_alive():
                print(f"Waiting for {name}...")
                thread.join(timeout=3) # Increased timeout slightly
                if thread.is_alive(): print(f"Warning: {name} did not terminate gracefully.")
                else: print(f"{name} joined.")
        
        if stm32_serial_conn and stm32_serial_conn.is_open:
            try:
                send_command_to_stm32(stm32_serial_conn, "SET_LIGHTS_OFF"); time.sleep(0.1)
                stm32_serial_conn.close(); print("STM32 serial closed.")
            except Exception as e: print(f"Error closing STM32 serial: {e}")
        print("WebUI Application finished.")