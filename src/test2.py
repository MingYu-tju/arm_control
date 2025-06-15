from flask import Flask, render_template, request, jsonify, Response
import time
import serial
import threading
import subprocess
import cv2
import wave
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import requests # 用于 API 调用
import base64   # 用于 Base64 编码
import hashlib  # 用于 SHA256 签名
import uuid     # 用于生成 salt


# 导入您自己的模块
from arm_module import Arm
from tpu_module import Control_TPU

# --- 有道智云短语音识别配置 ---
YOUDAO_ASR_URL = "https://openapi.youdao.com/asrapi"
YOUDAO_APP_KEY = "739d5abf63f230b6"  # 应用ID
YOUDAO_APP_SECRET = "oPgrJvXIN1LCjvACCsj1VeP7Fqf8QGnG" # 应用密钥
# 录音参数
YOUDAO_ASR_RATE = 16000
WAVE_OUTPUT_FILENAME = "temp_recorded_audio.wav" # 临时文件名 for uploaded audio

# --- TPU Model Configuration ---
TPU_MODEL_NAME = "rcModel_3"
TPU_CONF_THRES = 0.5
TPU_IOU_THRES = 0.45
TPU_IMAGE_WIDTH = 640
TPU_IMAGE_HEIGHT = 640
TPU_REMOTE_MODE = False
TPU_REMOTE_URL = '192.168.137.1'

# --- 其他全局配置 ---
STM32_SERIAL_PORT = "/dev/ttyUSB0"
STM32_SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1
DETECTION_ENABLED = True
PICK_HEIGHT = 62
ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z = 0, -180, 200
ARM_RESET_TIME = 1500
ARM_TRANSLATE_X_STEP, ARM_TRANSLATE_Y_STEP, ARM_TRANSLATE_Z_STEP = 15, 15, 15
ARM_MOVE_TIME = 300
PUMP_ROTATE_SMALL_STEP = 15

CHINESE_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CHINESE_FONT_SIZE = 16
pil_font = None

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
ARM_STATE_PICKING_MOVE_TO_DROP = "PICKING_MOVE_TO_DROP"
ARM_STATE_PICKING_RELEASING = "PICKING_RELEASING"
ARM_STATE_PICKING_HOMING_AFTER = "PICKING_HOMING_AFTER"
ARM_STATE_LISTENING = "LISTENING"
ARM_STATE_RECOGNIZING_VOICE = "RECOGNIZING_VOICE"

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
detected_objects_list = []
tpu_instance = None
stop_threads_flag = False
stm32_serial_conn = None
my_arm_instance = None
stm32_reader_thread_obj = None
arm_bg_thread_obj = None
object_detection_thread_obj = None
camera = None
camera_lock = threading.Lock()
CAMERA_INDEX = 0
last_voice_command_text = "待识别..."
last_voice_recognition_status = "就绪"

app = Flask(__name__)

# --- 有道智云语音识别辅助函数 ---
def truncate_q_for_sign(q_base64):
    """
    根据有道文档，为签名计算 'q' 的特殊形式：
    q=q前10个字符 + q长度 + q后10个字符（当q长度大于20）或 input=q字符串（当q长度小于等于20）
    """
    q_len = len(q_base64)
    if q_len <= 20:
        return q_base64
    else:
        return q_base64[:10] + str(q_len) + q_base64[q_len-10:]

def calculate_youdao_sign(app_key, q_for_sign, salt, curtime, app_secret):
    """
    计算有道API签名: sha256(应用ID+q_for_sign+salt+curtime+应用密钥)
    """
    sign_str = app_key + q_for_sign + salt + curtime + app_secret
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(sign_str.encode('utf-8'))
    return hash_algorithm.hexdigest()

# --- FFmpeg 转换函数 ---
def convert_audio_to_youdao_wav(input_filepath, output_filepath):
    """
    使用ffmpeg将音频转换为16kHz, 16-bit, 单声道 WAV (PCM)
    """
    # 确保ffmpeg已安装并在系统PATH中，或者提供完整路径
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-i', input_filepath,
        '-ar', str(YOUDAO_ASR_RATE),    # 采样率 16000
        '-ac', '1',                     # 单声道
        '-sample_fmt', 's16',           # 16-bit PCM (ffmpeg s16le for signed 16-bit little-endian)
        '-acodec', 'pcm_s16le',         # 编码器
        output_filepath
    ]
    try:
        print(f"执行FFmpeg转换: {' '.join(command)}")
        # 增加 stderr=subprocess.PIPE 来捕获 ffmpeg 的输出/错误信息
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=15) # 设置超时，例如15秒

        if process.returncode != 0:
            print(f"FFmpeg 转换失败 (返回码 {process.returncode}):")
            print(f"  FFmpeg STDOUT: {stdout.decode(errors='ignore')}")
            print(f"  FFmpeg STDERR: {stderr.decode(errors='ignore')}")
            return False
        
        # 再次检查输出文件是否存在且不为空
        if not os.path.exists(output_filepath) or os.path.getsize(output_filepath) == 0:
            print(f"FFmpeg 转换后输出文件不存在或为空: {output_filepath}")
            # 即使返回码为0，也可能因为某些原因没有生成有效文件
            if stdout: print(f"  FFmpeg STDOUT (on empty output): {stdout.decode(errors='ignore')}")
            if stderr: print(f"  FFmpeg STDERR (on empty output): {stderr.decode(errors='ignore')}")
            return False
            
        print(f"音频成功转换为: {output_filepath}")
        return True
    except FileNotFoundError:
        print("FFmpeg错误：找不到ffmpeg命令。请确保ffmpeg已安装并配置在系统PATH中。")
        return False
    except subprocess.TimeoutExpired:
        print("FFmpeg错误：转换超时。")
        if 'process' in locals() and process.poll() is None: # 确保 process 对象存在且仍在运行
             process.kill()
             process.wait() # 等待进程终止
        return False
    except Exception as e:
        print(f"FFmpeg转换时发生未知错误: {e}")
        return False

# --- 修改后的有道语音识别函数 ---
def recognize_speech_youdao(audio_filepath_original): # 参数名表示这是原始上传的文件路径
    global last_voice_command_text, last_voice_recognition_status, app_lock, current_arm_state

    # 检查AppKey和Secret配置 (这部分逻辑不变)
    if YOUDAO_APP_KEY == "YOUR_YOUDAO_APP_KEY" or YOUDAO_APP_SECRET == "YOUR_YOUDAO_APP_SECRET":
        error_msg = "有道智云 AppKey 或 AppSecret 未配置。"
        print(f"YOUDAO_ASR_ERROR: {error_msg}")
        with app_lock:
            last_voice_recognition_status = f"识别失败: {error_msg}"
            last_voice_command_text = "配置错误"
            if current_arm_state == ARM_STATE_RECOGNIZING_VOICE:
                current_arm_state = ARM_STATE_IDLE
        return None

    if not os.path.exists(audio_filepath_original):
        print(f"YOUDAO_ASR_ERROR: 原始音频文件 {audio_filepath_original} 未找到。")
        with app_lock:
            last_voice_recognition_status = "识别失败: 原始音频丢失"
            last_voice_command_text = "音频文件错误"
            if current_arm_state == ARM_STATE_RECOGNIZING_VOICE:
                current_arm_state = ARM_STATE_IDLE
        return None

    # --- 进行音频转换 ---
    # 定义转换后文件的路径，可以包含一个随机部分以避免并发冲突（如果需要）
    # 但由于每个请求在自己的线程中，固定临时名通常也可以，只要注意清理
    converted_audio_path = os.path.splitext(audio_filepath_original)[0] + "_converted.wav"
    
    print(f"准备转换音频: 从 '{audio_filepath_original}' 到 '{converted_audio_path}'")

    if not convert_audio_to_youdao_wav(audio_filepath_original, converted_audio_path):
        with app_lock: # 更新状态并返回
            last_voice_recognition_status = "识别失败: 音频转换错误"
            last_voice_command_text = "音频处理错误"
            if current_arm_state == ARM_STATE_RECOGNIZING_VOICE:
                current_arm_state = ARM_STATE_IDLE
        # 原始文件 audio_filepath_original 的清理由调用者 (process_audio_and_execute_command_task) 负责
        return None
    
    # --- 使用转换后的文件进行后续操作 ---
    audio_to_process_path = converted_audio_path 
    # 此时 audio_to_process_path 应该是符合有道要求的 WAV 文件

    try:
        if not os.path.exists(audio_to_process_path) or os.path.getsize(audio_to_process_path) == 0:
            print(f"YOUDAO_ASR_ERROR: 转换后的音频文件 {audio_to_process_path} 不存在或为空。")
            with app_lock:
                last_voice_recognition_status = "识别失败: 转换后音频丢失"
                last_voice_command_text = "音频处理错误"
            # current_arm_state 会在 finally 中处理
            return None

        with open(audio_to_process_path, 'rb') as audio_file:
            audio_content_binary = audio_file.read()
        
        q_base64 = base64.b64encode(audio_content_binary).decode('utf-8')
        
        # 由于我们已经强制转换为WAV，所以这里的format固定为"wav"
        audio_format_for_api = "wav" 
        
        curtime = str(int(time.time()))
        salt = str(uuid.uuid4())
        q_for_sign_calc = truncate_q_for_sign(q_base64)
        sign = calculate_youdao_sign(YOUDAO_APP_KEY, q_for_sign_calc, salt, curtime, YOUDAO_APP_SECRET)

        payload = {
            'q': q_base64,
            'langType': 'zh-CHS',
            'appKey': YOUDAO_APP_KEY,
            'salt': salt,
            'curtime': curtime,
            'sign': sign,
            'signType': 'v3', 
            'format': audio_format_for_api, # 固定为 "wav"
            'rate': str(YOUDAO_ASR_RATE),
            'channel': '1',
            'type': '1'
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        api_call_start_time = time.time()
        response = requests.post(YOUDAO_ASR_URL, data=payload, headers=headers, timeout=20)
        response.raise_for_status()
        api_call_end_time = time.time()
        print(f"YOUDAO_ASR: API call duration: {api_call_end_time - api_call_start_time:.2f}s")

        result_data = response.json()
        print(f"YOUDAO_ASR_RESPONSE: {result_data}")

        error_code = result_data.get("errorCode")
        if error_code == "0" and "result" in result_data and result_data["result"]:
            recognized_text_list = result_data["result"]
            # 有道API的result字段是一个列表，通常取第一个元素
            recognized_text = recognized_text_list[0] if isinstance(recognized_text_list, list) and recognized_text_list else str(recognized_text_list)
            with app_lock:
                last_voice_command_text = recognized_text
                last_voice_recognition_status = "识别成功 (有道)"
                print(f"YOUDAO_ASR_SUCCESS: Recognized text: '{recognized_text}'")
            return recognized_text
        else:
            error_map = { # (保持之前的错误映射)
                "101": "缺少必填参数", "102": "不支持的语言类型", "108": "应用ID无效",
                "110": "无相关服务的有效应用", "113": "q不能为空", "202": "签名检验失败",
                "206": "时间戳无效导致签名失败", "3001": "不支持的语音格式",
                "3007": "识别音频文件过大", "3008": "识别音频时长过长",
                "4304": "识别结果为空", "9301": "ASR识别失败",
            }
            err_msg_detail = error_map.get(error_code, f"未知错误")
            print(f"YOUDAO_ASR_ERROR: Recognition failed. ErrorCode: {error_code}, Detail: {err_msg_detail}, RawResponse: {result_data}")
            with app_lock:
                last_voice_recognition_status = f"识别失败({error_code}): {err_msg_detail}"
                last_voice_command_text = f"错误码: {error_code}"
            return None

    except requests.exceptions.RequestException as e:
        print(f"YOUDAO_ASR_ERROR: Request failed during speech recognition: {e}")
        with app_lock:
            last_voice_recognition_status = "识别请求失败 (网络)"
            last_voice_command_text = "网络错误"
        return None
    except json.JSONDecodeError:
        resp_text = response.text if 'response' in locals() and hasattr(response, 'text') else 'No response text available'
        print(f"YOUDAO_ASR_ERROR: Failed to decode JSON response: {resp_text}")
        with app_lock:
            last_voice_recognition_status = "识别失败: 响应格式错误"
            last_voice_command_text = "响应错误"
        return None
    except Exception as e:
        print(f"YOUDAO_ASR_ERROR: Unexpected error in recognize_speech_youdao: {e}")
        import traceback
        traceback.print_exc()
        with app_lock:
            last_voice_recognition_status = "识别失败: 未知内部错误"
            last_voice_command_text = "内部错误"
        return None
    finally:
        # 清理转换后的临时文件
        if os.path.exists(converted_audio_path): # converted_audio_path 在 try 块外部定义，所以这里可用
            try:
                os.remove(converted_audio_path)
                print(f"已删除转换后的临时文件: {converted_audio_path}")
            except Exception as e_del:
                print(f"删除转换后文件 {converted_audio_path} 失败: {e_del}")
        
        # 原始上传文件 (audio_filepath_original) 的清理由调用者 
        # (process_audio_and_execute_command_task) 的 finally 块负责。
        
        with app_lock: # 确保在所有路径下都重置状态（如果当前是识别中）
            if current_arm_state == ARM_STATE_RECOGNIZING_VOICE:
                current_arm_state = ARM_STATE_IDLE
# --- 任务线程 (process_audio_and_execute_command_task会调用新的recognize_speech_youdao) ---
def threaded_pick_action_task(target_object_name, pixel_x, pixel_y_original):
    global current_arm_state, current_pump_status, my_arm_instance, app_lock
    print(f"  THREAD_PICK_START: Task for '{target_object_name}' started.")
    pixel_y_adjusted = pixel_y_original - 20
    try:
        with app_lock:
            current_arm_state = ARM_STATE_PICKING_MOVE_TO_SUCK
        print(f"  THREAD_PICK: Moving to suck at ({pixel_x}, {pixel_y_adjusted}) at height {PICK_HEIGHT}")
        _, final_coord = my_arm_instance.move_pixel(pixel_x, pixel_y_adjusted, PICK_HEIGHT)
        time.sleep(0.5)
        with app_lock:
            current_arm_state = ARM_STATE_PICKING_SUCKING
        my_arm_instance.suck_up()
        with app_lock:
            current_pump_status = "ON"
        time.sleep(0.8)
        with app_lock:
            current_arm_state = ARM_STATE_PICKING_LIFTING
        print(f"  THREAD_PICK: Lifting object from ({final_coord.x:.2f}, {final_coord.y:.2f})")
        my_arm_instance.move_point(final_coord.x, final_coord.y, final_coord.z + 80, 800)
        time.sleep(0.5)
        with app_lock:
            current_arm_state = ARM_STATE_PICKING_MOVE_TO_DROP
        print(f"  THREAD_PICK: Moving to drop-off area (near HOME)")
        current_pos_after_lift = my_arm_instance.get_point()
        my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y - 30, current_pos_after_lift.z, 1200)
        my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y - 30, ARM_HOME_POS_Z + 50, 500)
        time.sleep(0.2)
        with app_lock:
            current_arm_state = ARM_STATE_PICKING_RELEASING
        print(f"  THREAD_PICK: Releasing object.")
        my_arm_instance.suck_release()
        with app_lock:
            current_pump_status = "OFF"
        time.sleep(0.5)
        with app_lock:
            current_arm_state = ARM_STATE_PICKING_HOMING_AFTER
        my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, 1000)
        print(f"  THREAD_PICK_SUCCESS: '{target_object_name}' 拾取完成并归位.")
    except Exception as e_pick_action:
        print(f"  THREAD_PICK_ERROR: 拾取 '{target_object_name}' 过程中发生错误: {e_pick_action}")
        try:
            my_arm_instance.suck_release()
            with app_lock:
                current_pump_status = "OFF"
        except:
            pass
    finally:
        with app_lock:
            current_arm_state = ARM_STATE_IDLE
            print(f"  THREAD_PICK_FINALLY: Task for '{target_object_name}' finished. Arm state IDLE.")

# test.py

def process_audio_and_execute_command_task(audio_filepath): # audio_filepath is the path to the original uploaded file
    global current_arm_state, app_lock, last_voice_command_text, last_voice_recognition_status, detected_objects_list
    
    recognized_text = None # 初始化
    try:
        # 调用 recognize_speech_youdao, 它会处理转换和API调用
        recognized_text = recognize_speech_youdao(audio_filepath) 
        
        # --- 后续的NLU和拾取逻辑 ---
        if recognized_text:
            print(f"  VOICE_THREAD_NLU: Recognized text: '{recognized_text}'")
            target_object_to_pick = None
            unique_detected_classes = []
            with app_lock:
                unique_detected_classes = sorted(list(set(obj['name'] for obj in detected_objects_list)), key=len, reverse=True)
            
            for class_name in unique_detected_classes:
                if class_name in recognized_text:
                    target_object_to_pick = class_name
                    break
            
            if target_object_to_pick:
                object_instance_to_pick = None
                with app_lock:
                    # 检查 arm state 是否允许操作
                    if current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_RECOGNIZING_VOICE]: # 允许从识别状态直接进入拾取
                        print(f"  VOICE_THREAD_NLU_WARN: Arm not IDLE/RECOGNIZING ({current_arm_state}), cannot pick '{target_object_to_pick}'.")
                        last_voice_recognition_status = f"机械臂忙,无法拾取'{target_object_to_pick}'"
                        # recognize_speech_youdao 的 finally 块会处理 current_arm_state
                        return 

                    best_conf = -1.0
                    for obj in detected_objects_list:
                        if obj['name'] == target_object_to_pick and obj['confidence'] > best_conf:
                            best_conf = obj['confidence']
                            object_instance_to_pick = obj
                    
                    if object_instance_to_pick:
                        current_arm_state = ARM_STATE_PICKING_SUBMITTED # 进入拾取流程
                        print(f"  VOICE_THREAD_NLU: Found '{object_instance_to_pick['name']}', submitting pick task.")
                        last_voice_recognition_status = f"准备拾取'{object_instance_to_pick['name']}'"
                    else:
                        print(f"  VOICE_THREAD_NLU_WARN: Object '{target_object_to_pick}' not found in detected list.")
                        last_voice_recognition_status = f"未找到'{target_object_to_pick}'"
                        # 如果之前是 RECOGNIZING_VOICE，recognize_speech_youdao 的 finally 块会设为 IDLE
                        # 如果这里是因为找不到物体而失败，且状态是IDLE，则保持IDLE
                        # 如果状态因其他原因改变，则不修改
                        # (recognize_speech_youdao 的 finally 块已经处理了从 RECOGNIZING_VOICE 到 IDLE 的转换)
                        return 
                
                if object_instance_to_pick: # 确保在锁外启动线程
                     pick_thread = threading.Thread(target=threaded_pick_action_task, args=(object_instance_to_pick['name'], object_instance_to_pick['box_center_x'], object_instance_to_pick['box_center_y']), daemon=True)
                     pick_thread.start()
            else: # 未识别到可拾取对象
                print(f"  VOICE_THREAD_NLU_INFO: No pickable object in '{recognized_text}'.")
                with app_lock: # 确保在锁内更新状态
                    last_voice_recognition_status = "未识别到拾取指令"
                    # recognize_speech_youdao 的 finally 块已处理 RECOGNIZING_VOICE -> IDLE
                    # 如果这里是因为没有NLU匹配，且状态是IDLE，则保持IDLE

        # 如果 recognized_text 为 None (识别失败), recognize_speech_youdao 内部的 finally 块
        # 已经将 current_arm_state (如果原先是 RECOGNIZING_VOICE) 设置为 ARM_STATE_IDLE。
        # 此处无需额外处理 current_arm_state，除非有特定于此函数的逻辑。
    
    finally:
        # 确保原始上传的临时文件被删除
        if os.path.exists(audio_filepath): # audio_filepath 是原始上传文件的路径
            try:
                os.remove(audio_filepath)
                print(f"已删除原始上传的临时文件: {audio_filepath}")
            except Exception as e_del:
                print(f"删除原始上传文件 {audio_filepath} 失败: {e_del}")
        
        # recognize_speech_youdao 的 finally 块负责将 ARM_STATE_RECOGNIZING_VOICE 重置为 ARM_STATE_IDLE。
        # 此处不需要再次设置，除非有特定于 process_audio_and_execute_command_task 的状态转换逻辑。
        # 比如，如果 NLU 步骤后决定不进行拾取，也应该确保状态回到 IDLE（如果它不是PICKING_SUBMITTED）。
        # 但这部分已由 recognize_speech_youdao 的 finally 和 NLU 逻辑覆盖。

# --- STM32、物体检测、机械臂后台、视频流生成函数 (保持不变) ---
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
                                            if current_arm_state in [ARM_STATE_IDLE, ARM_STATE_PAUSED_TRAJECTORY] or (current_arm_state == ARM_STATE_HOMING and not was_running_when_homing_triggered):
                                                current_arm_state = ARM_STATE_RUNNING_TRAJECTORY
                                                current_trajectory_index = 0
                                                was_running_when_homing_triggered = False
                                        elif new_status == "YELLOW_ON":
                                            if current_arm_state == ARM_STATE_PAUSED_TRAJECTORY:
                                                if current_trajectory_index < TOTAL_TRAJECTORY_POINTS:
                                                    current_arm_state = ARM_STATE_RUNNING_TRAJECTORY
                                                    was_running_when_homing_triggered = False
                                                else:
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
            if not stop_threads_flag:
                print(f"[STM32 RX Thread WebUI] SerialException.")
            break
        except Exception as e:
            if not stop_threads_flag:
                print(f"[STM32 RX Thread WebUI] Error: {e}.")
            break
    print("[STM32 RX Thread WebUI] Stopped.")

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


def arm_background_task():
    global current_arm_state, current_trajectory_index, was_running_when_homing_triggered, app_lock
    print("[Arm BG Task WebUI] Started.")
    while not stop_threads_flag:
        state_to_process = None;
        with app_lock:
            state_to_process = current_arm_state
        if my_arm_instance:
            if state_to_process == ARM_STATE_HOMING:
                print("  BG_TASK: Arm state is HOMING. Executing reset...");
                try:
                    my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, ARM_RESET_TIME)
                    print("  BG_TASK: Arm homing complete.")
                except Exception as e_home:
                    print(f"  BG_TASK_ERROR: Homing failed: {e_home}")
                with app_lock:
                    if was_running_when_homing_triggered and stm32_current_light_status == "RED_ON":
                        current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
                        print(f"  BG_TASK: Homing (RED) done. State set to PAUSED.")
                    else:
                        current_arm_state = ARM_STATE_IDLE
                        current_trajectory_index = 0
                        print("  BG_TASK: Homing done. State set to IDLE.")
                    was_running_when_homing_triggered = False
            elif state_to_process == ARM_STATE_RUNNING_TRAJECTORY:
                can_run_step, point_data, temp_traj_index = False, None, -1
                with app_lock:
                    if current_trajectory_index < TOTAL_TRAJECTORY_POINTS:
                        can_run_step, point_data, temp_traj_index = True, PREDEFINED_TRAJECTORY[current_trajectory_index], current_trajectory_index
                    else:
                        print("  BG_TASK: Trajectory fully completed.");
                        current_arm_state = ARM_STATE_IDLE
                        current_trajectory_index = 0
                if can_run_step and point_data:
                    target_x, target_y, target_z, move_duration = point_data
                    print(f"  BG_TASK_TRAJ: Moving to point {temp_traj_index+1}/{TOTAL_TRAJECTORY_POINTS}: ({target_x}, {target_y}, {target_z})")
                    try:
                        my_arm_instance.move_point(target_x, target_y, target_z, move_duration)
                        with app_lock:
                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY and current_trajectory_index == temp_traj_index:
                                current_trajectory_index += 1
                                if current_trajectory_index >= TOTAL_TRAJECTORY_POINTS:
                                    print("  BG_TASK_TRAJ: Trajectory fully completed (last point).")
                                    current_arm_state = ARM_STATE_IDLE
                                    current_trajectory_index = 0
                    except Exception as e_traj:
                        print(f"  BG_TASK_ERROR: Moving to trajectory point {temp_traj_index + 1} failed: {e_traj}")
                        with app_lock:
                            current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
        time.sleep(0.1)
    print("[Arm BG Task WebUI] Stopped.")

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

# --- Flask Routes (保持不变，除了调用ASR的部分) ---
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_voice_audio', methods=['POST'])
def upload_voice_audio_route():
    global current_arm_state, app_lock, last_voice_command_text, last_voice_recognition_status
    
    if 'audio_data' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio data in request.'}), 400
    
    audio_file = request.files['audio_data']
    
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

    # Ensure the 'uploads' directory exists if you want to save files there permanently
    # For temp files, current directory or a dedicated temp dir is fine.
    # uploads_dir = os.path.join(app.root_path, 'uploads')
    # os.makedirs(uploads_dir, exist_ok=True)
    # save_path = os.path.join(uploads_dir, secure_filename(audio_file.filename))
    
    # Using a fixed temporary filename
    save_path = WAVE_OUTPUT_FILENAME 
    
    try:
        audio_file.save(save_path)
        print(f"AUDIO_UPLOAD: Audio file saved to {save_path}")
    except Exception as e:
        print(f"AUDIO_UPLOAD_ERROR: Failed to save audio file: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to save audio: {e}'}), 500

    with app_lock:
        if current_arm_state != ARM_STATE_IDLE and current_arm_state != ARM_STATE_RECOGNIZING_VOICE :
            message = f"机械臂忙 ({current_arm_state}), 无法处理语音。"
            if os.path.exists(save_path):
                try: os.remove(save_path)
                except Exception as e_del: print(f"AUDIO_CLEANUP_WARN: {e_del}")
            return jsonify({'status': 'error', 'message': message}), 409
        
        current_arm_state = ARM_STATE_RECOGNIZING_VOICE
        last_voice_command_text = "识别中..."
        last_voice_recognition_status = "已收到音频，正在识别..."
        message = "音频已接收，正在后台处理语音识别..."

    voice_processing_thread = threading.Thread(
        target=process_audio_and_execute_command_task, 
        args=(save_path,), 
        daemon=True
    )
    voice_processing_thread.start()
    
    return jsonify({'status': 'success', 'message': message}), 202


@app.route('/control', methods=['POST'])
def control_arm_route():
    global current_arm_state, current_trajectory_index, detected_objects_list, app_lock, \
           last_voice_command_text, last_voice_recognition_status, was_running_when_homing_triggered, \
           current_pump_status, current_pump_angle
    data = request.get_json()
    command = data.get('command')
    payload = data.get('payload', {})
    message = f"CMD_RECV: {command}"
    success = True
    status_code = 200
        
    if command == 'process_voice_text': # Simulated text input
        text_command = payload.get('text', '').strip()
        print(f"  SIM_VOICE_NLU: Received text: '{text_command}'")
        target_object_to_pick = None
        unique_detected_classes = []
        with app_lock:
            unique_detected_classes = sorted(list(set(obj['name'] for obj in detected_objects_list)), key=len, reverse=True)
        for cn in unique_detected_classes:
            if cn in text_command:
                target_object_to_pick = cn
                break
        if target_object_to_pick:
            obj_instance_to_pick = None
            with app_lock:
                if current_arm_state != ARM_STATE_IDLE:
                    message = f"机械臂忙 ({current_arm_state}), 无法执行模拟语音拾取 '{target_object_to_pick}'。"
                    return jsonify({'status': 'error', 'message': message}), 409
                best_conf = -1.0
                for obj in detected_objects_list:
                    if obj['name'] == target_object_to_pick and obj['confidence'] > best_conf:
                        best_conf = obj['confidence']
                        obj_instance_to_pick = obj
                if obj_instance_to_pick:
                    current_arm_state = ARM_STATE_PICKING_SUBMITTED
                    message = f"模拟语音: 准备拾取 {obj_instance_to_pick['name']}."
                    last_voice_command_text = text_command
                    last_voice_recognition_status = "模拟指令已识别"
                else:
                    message = f"模拟语音: 未在画面中找到 '{target_object_to_pick}'。"
                    last_voice_command_text = text_command
                    last_voice_recognition_status = f"未找到'{target_object_to_pick}'"
                    return jsonify({'status': 'error', 'message': message}), 404
            if obj_instance_to_pick:
                pick_thread = threading.Thread(target=threaded_pick_action_task, args=(obj_instance_to_pick['name'], obj_instance_to_pick['box_center_x'], obj_instance_to_pick['box_center_y']), daemon=True)
                pick_thread.start()
                status_code = 202
            return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code
        else:
            message = f"模拟语音: 未识别到指令中的目标 '{text_command}'。"
            last_voice_command_text = text_command
            last_voice_recognition_status = "无匹配目标"
            return jsonify({'status': 'error', 'message': message}), 400
    elif command.startswith('set_'):
        with app_lock:
            if command == 'set_green_on':
                success = send_command_to_stm32(stm32_serial_conn, "SET_GREEN_ON")
            elif command == 'set_red_on':
                success = send_command_to_stm32(stm32_serial_conn, "SET_RED_ON")
            elif command == 'set_yellow_on':
                success = send_command_to_stm32(stm32_serial_conn, "SET_YELLOW_ON")
            elif command == 'set_lights_off':
                success = send_command_to_stm32(stm32_serial_conn, "SET_LIGHTS_OFF")
            else:
                success = False
                message = "Unknown set command"
        if not success:
            status_code = 500
            message = f"Failed to send {command}"
        else:
            message = f"Command {command} sent to STM32."
        return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code
    elif not my_arm_instance:
        return jsonify({'status': 'error', 'message': '机械臂未初始化!'}), 503
    elif command == 'pick_object':
        pixel_x = payload.get('pixel_x')
        pixel_y = payload.get('pixel_y')
        obj_name = payload.get('name', '未知物体')
        if pixel_x is None or pixel_y is None:
            message = f"UI_PICK_ERROR: 未提供有效像素坐标."
            return jsonify({'status': 'error', 'message': message}), 400
        with app_lock:
            if current_arm_state != ARM_STATE_IDLE:
                message = f"机械臂忙 ({current_arm_state}), 无法执行UI拾取."
                return jsonify({'status': 'error', 'message': message}), 409
            current_arm_state = ARM_STATE_PICKING_SUBMITTED
            message = f"UI_PICK_SUBMITTED: 拾取 '{obj_name}' 已提交."
        pick_thread = threading.Thread(target=threaded_pick_action_task, args=(obj_name, pixel_x, pixel_y), daemon=True)
        pick_thread.start()
        status_code = 202
        return jsonify({'status': 'success', 'message': message}), status_code
    else:
        action_taken = False
        with app_lock:
            arm_busy_for_manual = current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_PAUSED_TRAJECTORY]
            if arm_busy_for_manual and command.startswith('manual_') and command != 'manual_reset':
                message = f"机械臂忙 ({current_arm_state}), 无法手动操作."
                success = False
                status_code = 409
                return jsonify({'status': 'error', 'message': message}), status_code
        if command == 'manual_reset':
            with app_lock:
                if current_arm_state != ARM_STATE_HOMING:
                    current_arm_state = ARM_STATE_HOMING
                    was_running_when_homing_triggered = False
                    current_trajectory_index = 0
                    message = "ARM_RESET_CMD: 复位指令已提交."
                else:
                    message = "ARM_RESET_CMD: 机械臂已在复位中."
            action_taken = True
        elif command.startswith('manual_'):
            with app_lock:
                current_arm_state = ARM_STATE_MANUAL_MOVE
            try:
                if command == 'manual_x_plus':
                    my_arm_instance.move_axis_point(0, ARM_TRANSLATE_X_STEP, ARM_MOVE_TIME)
                    message = "Arm X+"
                elif command == 'manual_x_minus':
                    my_arm_instance.move_axis_point(0, -ARM_TRANSLATE_X_STEP, ARM_MOVE_TIME)
                    message = "Arm X-"
                elif command == 'manual_y_plus':
                    my_arm_instance.move_axis_point(1, ARM_TRANSLATE_Y_STEP, ARM_MOVE_TIME)
                    message = "Arm Y+"
                elif command == 'manual_y_minus':
                    my_arm_instance.move_axis_point(1, -ARM_TRANSLATE_Y_STEP, ARM_MOVE_TIME)
                    message = "Arm Y-"
                elif command == 'manual_z_plus':
                    my_arm_instance.move_axis_point(2, ARM_TRANSLATE_Z_STEP, ARM_MOVE_TIME)
                    message = "Arm Z+"
                elif command == 'manual_z_minus':
                    my_arm_instance.move_axis_point(2, -ARM_TRANSLATE_Z_STEP, ARM_MOVE_TIME)
                    message = "Arm Z-"
                action_taken = True
            except Exception as e_manual:
                message = f"手动操作错误: {e_manual}"
                success = False
                status_code = 500
            finally:
                with app_lock:
                    if current_arm_state == ARM_STATE_MANUAL_MOVE:
                        current_arm_state = ARM_STATE_IDLE
        elif command.startswith('pump_'):
            with app_lock:
                if command == 'pump_on':
                    my_arm_instance.suck_up()
                    current_pump_status = "ON"
                elif command == 'pump_off':
                    my_arm_instance.suck_release()
                    current_pump_status = "OFF"
                elif command == 'pump_angle_set':
                    angle = payload.get('angle', current_pump_angle)
                    my_arm_instance.rotate_angle(angle)
                    current_pump_angle = angle
                elif command == 'pump_rotate_left_small':
                    new_angle = max(0, current_pump_angle - PUMP_ROTATE_SMALL_STEP)
                    my_arm_instance.rotate_angle(new_angle)
                    current_pump_angle = new_angle
                elif command == 'pump_rotate_right_small':
                    new_angle = min(180, current_pump_angle + PUMP_ROTATE_SMALL_STEP)
                    my_arm_instance.rotate_angle(new_angle)
                    current_pump_angle = new_angle
                message = f"PUMP_CMD: {command} executed. Angle: {current_pump_angle}, Status: {current_pump_status}"
            action_taken = True
        elif command == 'task_pause':
            with app_lock:
                if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY:
                    current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
                    message = "TASK_PAUSED"
                else:
                    success = False
                    message = "无法暂停: 非轨迹运行中."
            action_taken = True
        elif command == 'task_stop':
            with app_lock:
                if current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_HOMING]:
                    current_arm_state = ARM_STATE_HOMING
                    was_running_when_homing_triggered = False
                    current_trajectory_index = 0
                    message = "TASK_STOP_HOMING: 停止并归位."
                else:
                    message = "任务已停止或归位中."
            action_taken = True
        elif command == 'task_continue':
            with app_lock:
                if current_arm_state == ARM_STATE_PAUSED_TRAJECTORY:
                    current_arm_state = ARM_STATE_RUNNING_TRAJECTORY
                    message = "TASK_CONTINUED"
                else:
                    success = False
                    message = "无法继续: 非暂停状态."
            action_taken = True
        if not action_taken and success: 
            message = f"未知指令或状态不允许: {command}"
            success = False
            status_code = 400
        return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

@app.route('/status')
def get_status():
    global last_voice_command_text, last_voice_recognition_status
    with app_lock:
        arm_coords_str = "[N/A]"
        if my_arm_instance:
            try:
                pos_data = my_arm_instance.get_axis_point('3')
                if pos_data and isinstance(pos_data, (list, tuple)) and len(pos_data) >= 3:
                    arm_coords_str = f"[{pos_data[0]:.1f}, {pos_data[1]:.1f}, {pos_data[2]:.1f}]"
            except Exception:
                arm_coords_str = "[Error Reading]"
        objects_to_send = [obj.copy() for obj in detected_objects_list]
        status_data = {
            'arm_state': current_arm_state, 'stm32_light_status': stm32_current_light_status,
            'current_trajectory_index': current_trajectory_index, 'total_trajectory_points': TOTAL_TRAJECTORY_POINTS,
            'pump_status': current_pump_status, 'pump_angle': current_pump_angle, 'arm_coords': arm_coords_str,
            'detected_objects': objects_to_send, 'voice_command_text': last_voice_command_text,
            'voice_recognition_status': last_voice_recognition_status, 'timestamp': time.time()
        }
    return jsonify(status_data)

# --- 初始化与主运行 (保持不变) ---
def init_hardware_and_camera():
    global my_arm_instance, stm32_serial_conn, stm32_reader_thread_obj, arm_bg_thread_obj, camera, current_pump_status, current_pump_angle, tpu_instance, object_detection_thread_obj, current_arm_state, current_trajectory_index, was_running_when_homing_triggered, pil_font, last_voice_command_text, last_voice_recognition_status
    if pil_font is None:
        try:
            pil_font = ImageFont.truetype(CHINESE_FONT_PATH, CHINESE_FONT_SIZE)
            print(f"字体初始化: 成功加载中文字体: {CHINESE_FONT_PATH}")
        except Exception as e:
            print(f"!!! 字体初始化警告: 加载字体时发生错误: {e}")
            pil_font = "FONT_LOAD_FAILED"
    last_voice_command_text = "待识别..."
    last_voice_recognition_status = "就绪"
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
        print(f"!!! Error initializing Arm (WebUI): {e} !!!")
        my_arm_instance = None
    print(f"Connecting to STM32 ({STM32_SERIAL_PORT}) for WebUI...")
    try:
        stm32_serial_conn = serial.Serial(STM32_SERIAL_PORT, STM32_SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
        stm32_reader_thread_obj = threading.Thread(target=stm32_reader_thread_func, args=(stm32_serial_conn,), daemon=True)
        stm32_reader_thread_obj.start()
        print("STM32 serial connected and reader thread started for WebUI.")
    except serial.SerialException as e:
        print(f"!!! Failed to connect to STM32 (WebUI): {e} !!!")
        stm32_serial_conn = None
    print(f"Initializing Camera (index: {CAMERA_INDEX}) for WebUI...")
    with camera_lock:
        try:
            camera = cv2.VideoCapture(CAMERA_INDEX)
            if not camera.isOpened():
                print(f"!!! Error: Could not open camera {CAMERA_INDEX} !!!")
                camera = None
            else:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Camera {CAMERA_INDEX} initialized successfully.")
        except Exception as e:
            print(f"!!! Error initializing camera: {e} !!!")
            camera = None
    if DETECTION_ENABLED:
        print("Initializing Control_TPU for WebUI detection...")
        try:
            tpu_instance = Control_TPU
            print("Control_TPU instance created. Initialization will occur on first use.")
            object_detection_thread_obj = threading.Thread(target=object_detection_task, daemon=True)
            object_detection_thread_obj.start()
            print("Object detection thread started for WebUI.")
        except Exception as e:
            print(f"!!! Error creating Control_TPU instance (WebUI): {e} !!!")
            tpu_instance = None
    else:
        print("Object detection is DISABLED.")
    arm_bg_thread_obj = threading.Thread(target=arm_background_task, daemon=True)
    arm_bg_thread_obj.start()
    print("Arm background task thread started for WebUI.")

if __name__ == '__main__':
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("错误：Pillow库未安装。请运行 'pip install Pillow' 来安装。")
        exit(1)

    # 替换 YOUDAO_APP_KEY 和 YOUDAO_APP_SECRET
    if YOUDAO_APP_KEY == "YOUR_YOUDAO_APP_KEY" or YOUDAO_APP_SECRET == "YOUR_YOUDAO_APP_SECRET":
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 警告: 请在 test.py 文件中设置您的有道智云 APP_KEY 和 APP_SECRET !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        # 可以选择在这里退出，或者允许程序运行但语音识别会失败
        # exit(1) 

    init_hardware_and_camera()
    print("Starting Flask web server on http://0.0.0.0:5000 ...")
    try:
        # 如果需要HTTPS以便在非localhost IP上测试麦克风，可以取消注释ssl_context
        # context = ('server.crt', 'server.key') # 需要您自己生成这些文件
        # app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False, ssl_context=context)
        # 或者使用 adhoc (浏览器会警告，仅用于临时测试):
        # app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False, ssl_context='adhoc')
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        print("Flask server shutting down...")
        stop_threads_flag = True
        print("Releasing camera...")
        with camera_lock:
            if camera and camera.isOpened():
                camera.release()
                camera = None
                print("Camera released.")
        threads_to_join = [(stm32_reader_thread_obj, "STM32 reader thread"), (arm_bg_thread_obj, "Arm background task"), (object_detection_thread_obj, "Object detection task")]
        for thread, name in threads_to_join:
            if thread and thread.is_alive():
                print(f"Waiting for {name}...")
                thread.join(timeout=2)
                if thread.is_alive():
                    print(f"Warning: {name} did not terminate gracefully.")
                else:
                    print(f"{name} joined.")
        if stm32_serial_conn and stm32_serial_conn.is_open:
            try:
                send_command_to_stm32(stm32_serial_conn, "SET_LIGHTS_OFF")
                time.sleep(0.1)
                stm32_serial_conn.close()
                print("STM32 serial closed.")
            except Exception as e:
                print(f"Error closing STM32 serial: {e}")
        print("WebUI Application finished.")