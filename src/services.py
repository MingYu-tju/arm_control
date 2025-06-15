# services.py
import requests
import base64
import hashlib
import uuid
import subprocess
import os
import json
import time
import traceback

# 导入配置和全局状态
import config
import state

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
        '-ar', str(config.YOUDAO_ASR_RATE), '-ac', '1',
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
    if config.YOUDAO_APP_KEY == "YOUR_YOUDAO_APP_KEY" or config.YOUDAO_APP_SECRET == "YOUR_YOUDAO_APP_SECRET":
        with state.app_lock: state.last_nlu_status = "ASR失败: 配置错误"
        return None

    if not os.path.exists(audio_filepath_original):
        with state.app_lock: state.last_nlu_status = "ASR失败: 原始音频丢失"
        return None

    converted_audio_path = os.path.splitext(audio_filepath_original)[0] + "_converted.wav"
    if not convert_audio_to_youdao_wav(audio_filepath_original, converted_audio_path):
        with state.app_lock: state.last_nlu_status = "ASR失败: 音频转换错误"
        return None
    
    audio_to_process_path = converted_audio_path
    try:
        if not os.path.exists(audio_to_process_path) or os.path.getsize(audio_to_process_path) == 0:
            with state.app_lock: state.last_nlu_status = "ASR失败: 转换后音频丢失"
            return None

        with open(audio_to_process_path, 'rb') as audio_file:
            audio_content_binary = audio_file.read()
        q_base64 = base64.b64encode(audio_content_binary).decode('utf-8')
        
        curtime, salt = str(int(time.time())), str(uuid.uuid4())
        q_for_sign_calc = truncate_q_for_sign(q_base64)
        sign = calculate_youdao_sign(config.YOUDAO_APP_KEY, q_for_sign_calc, salt, curtime, config.YOUDAO_APP_SECRET)

        payload = {
            'q': q_base64, 'langType': 'zh-CHS', 'appKey': config.YOUDAO_APP_KEY, 'salt': salt,
            'curtime': curtime, 'sign': sign, 'signType': 'v3', 'format': "wav",
            'rate': str(config.YOUDAO_ASR_RATE), 'channel': '1', 'type': '1'
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = requests.post(config.YOUDAO_ASR_URL, data=payload, headers=headers, timeout=20)
        response.raise_for_status()
        result_data = response.json()
        print(f"YOUDAO_ASR_RESPONSE: {result_data}")

        error_code = result_data.get("errorCode")
        if error_code == "0" and result_data.get("result"):
            recognized_text = result_data["result"][0] if result_data["result"] else ""
            with state.app_lock: state.last_voice_command_text = recognized_text
            print(f"YOUDAO_ASR_SUCCESS: Recognized text: '{recognized_text}'")
            return recognized_text
        else:
            with state.app_lock: state.last_nlu_status = f"ASR失败({error_code})"
            return None
    except requests.exceptions.RequestException as e:
        print(f"YOUDAO_ASR_ERROR: Request failed: {e}")
        with state.app_lock: state.last_nlu_status = "ASR网络错误"
        return None
    except json.JSONDecodeError:
        print(f"YOUDAO_ASR_ERROR: Failed to decode JSON response")
        with state.app_lock: state.last_nlu_status = "ASR响应格式错误"
        return None
    except Exception as e:
        print(f"YOUDAO_ASR_ERROR: Unexpected error: {e}")
        with state.app_lock: state.last_nlu_status = "ASR未知错误"
        return None
    finally:
        if os.path.exists(converted_audio_path):
            try: os.remove(converted_audio_path)
            except Exception as e_del: print(f"删除转换文件失败: {e_del}")

# --- 有道智云语音合成 (TTS) 辅助函数 ---
def truncate_text_for_tts_sign(text_q):
    """
    根据有道TTS API文档为签名计算截断文本 'q' (称为 'input')。
    input = q前10个字符 + q字符长度 + q后10个字符（当q字符长度大于20）
    或 input = q字符串（当q字符长度小于等于20）
    """
    str_q = str(text_q) 
    q_char_len = len(str_q) 
    
    if q_char_len <= 20:
        return str_q
    return str_q[:10] + str(q_char_len) + str_q[q_char_len-10:]

def synthesize_speech_youdao(text_input):
    """
    调用有道智云TTS API将文本合成为语音 (MP3)。
    返回: MP3音频内容的bytes，如果失败则返回None。
    """
    if config.YOUDAO_APP_KEY == "YOUR_YOUDAO_APP_KEY" or config.YOUDAO_APP_SECRET == "YOUR_YOUDAO_APP_SECRET":
        print("TTS_ERROR: Youdao App Key/Secret not configured.")
        return None

    if not text_input:
        print("TTS_ERROR: No text input provided for synthesis.")
        return None
    
    # 检查文本长度限制 (UTF-8编码后)
    if len(text_input.encode('utf-8')) > 2048:
        print("TTS_ERROR: Text input too long (max 2048 UTF-8 bytes). Will attempt to synthesize first 2048 bytes equivalent.")
        # 尝试截断，但这可能导致UTF-8字符被切断，更稳妥的做法是按字符截断到接近长度
        # 为简单起见，这里先按字符估算截断，实际应用可能需要更精确的UTF-8字节截断
        estimated_max_chars = 2048 // 3 # 假设平均一个中文字符占3字节
        text_input = text_input[:estimated_max_chars]
        if len(text_input.encode('utf-8')) > 2048: # 再次检查
             # 如果仍然超长，粗暴截断字节（可能破坏字符）
            text_input_bytes = text_input.encode('utf-8')[:2048]
            text_input = text_input_bytes.decode('utf-8', errors='ignore')


    curtime = str(int(time.time()))
    salt = str(uuid.uuid4())
    
    # 为签名计算 'input'
    q_for_sign_calc = truncate_text_for_tts_sign(text_input)
    
    sign = calculate_youdao_sign(config.YOUDAO_APP_KEY, q_for_sign_calc, salt, curtime, config.YOUDAO_APP_SECRET)

    payload = {
        'q': text_input, # requests库会自动对data字典中的值进行URL编码
        'appKey': config.YOUDAO_APP_KEY,
        'salt': salt,
        'curtime': curtime,
        'sign': sign,
        'signType': 'v3',
        'format': 'mp3',
        'voiceName': config.YOUDAO_TTS_VOICE_NAME,
        'speed': '0.9',  # 语速，可选
        'volume': '3.00', # 音量，可选
    }
    
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        response = requests.post(config.YOUDAO_TTS_URL, data=payload, headers=headers, timeout=20)
        # 不首先调用 raise_for_status，因为错误时TTS API返回JSON，状态码可能是200但Content-Type是json
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        if response.status_code == 200 and 'audio/mp3' in content_type:
            print(f"TTS_SUCCESS: Synthesized audio for: '{text_input[:30]}...'")
            return response.content
        elif 'application/json' in content_type: # 错误情况，通常也是200 OK但内容是JSON
            error_data = response.json()
            error_code = error_data.get("errorCode")
            print(f"TTS_ERROR: API returned error. HTTP Status: {response.status_code}, ErrorCode: {error_code}. Response: {error_data}")
            return None
        else:
            # 其他非预期的成功状态码或内容类型
            response.raise_for_status() # 现在可以抛出，如果前面没匹配到audio/mp3或json错误
            print(f"TTS_ERROR: Unexpected Content-Type: {content_type} or Status: {response.status_code}. Response: {response.text[:200]}")
            return None
            
    except requests.exceptions.HTTPError as e: # HTTP错误状态码 4xx, 5xx
        print(f"TTS_ERROR: HTTP error during synthesis: {e}. Response: {e.response.text[:200] if e.response else 'No response body'}")
        return None
    except requests.exceptions.RequestException as e: # 网络层或其他请求错误
        print(f"TTS_ERROR: Request failed: {e}")
        return None
    except json.JSONDecodeError: # 如果期望JSON错误但无法解析
        print(f"TTS_ERROR: Failed to decode JSON error response. Raw: {response.text[:200]}")
        return None
    except Exception as e:
        print(f"TTS_ERROR: Unexpected error during synthesis: {e}")
        traceback.print_exc()
        return None

# --- DeepSeek NLU 函数 ---
def call_deepseek_nlu(text_input):
    if config.DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY":
        error_msg = "DeepSeek API Key 未配置。"
        print(f"DEEPSEEK_NLU_ERROR: {error_msg}")
        with state.app_lock:
            state.last_nlu_status = f"NLU失败: {error_msg}"
            state.last_nlu_result_text = "配置错误"
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}"
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
        "{\"action\": \"pick_and_place\", \"pick_object\": \"苹果\", \"place_coordinates\": {\"x\": 0, \"y\": -190}, \"feedback_text\": \"好的，已经帮您把苹果放到桌子中间。\"}\n"
        "如果用户说“拿起那个杯子”，你应该返回：\n"
        "{\"action\": \"pick\", \"pick_object\": \"杯子\", \"place_coordinates\": null, \"feedback_text\": \"好的，已经拿起那个杯子。\"}\n"
        "请确保返回的 JSON 格式严格正确，不要包含任何额外的解释或文本在 JSON 对象之外。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text_input}
    ]
    data = {
        "model": config.DEEPSEEK_MODEL, 
        "messages": messages, 
        "stream": False, 
        "temperature": 0.1,
        "max_tokens": 250
    }

    try:
        with state.app_lock:
            state.last_nlu_status = "正在调用NLU服务..."

        response = requests.post(config.DEEPSEEK_API_URL, headers=headers, json=data, timeout=25)
        response.raise_for_status()
        response_json = response.json()
        print(f"DEEPSEEK_NLU_RESPONSE (RAW): {response_json}")

        if response_json and response_json.get("choices") and \
           len(response_json["choices"]) > 0 and response_json["choices"][0].get("message"):
            content_str = response_json["choices"][0]["message"].get("content")
            if content_str:
                print(f"DEEPSEEK_NLU_CONTENT_STR (Before strip): '{content_str}'")
                
                pure_json_str = content_str.strip()
                if pure_json_str.startswith("```json"):
                    pure_json_str = pure_json_str[len("```json"):].strip()
                elif pure_json_str.startswith("```"):
                    pure_json_str = pure_json_str[len("```"):].strip()
                
                if pure_json_str.endswith("```"):
                    pure_json_str = pure_json_str[:-len("```")].strip()
                
                pure_json_str = pure_json_str.strip()
                print(f"DEEPSEEK_NLU_PURE_JSON_STR (After strip): '{pure_json_str}'")

                try:
                    parsed_nlu_result = json.loads(pure_json_str)
                    
                    required_keys = ["action", "pick_object", "place_coordinates", "feedback_text"]
                    if not all(k in parsed_nlu_result for k in required_keys):
                        raise ValueError(f"NLU JSON缺少必要字段，需要: {required_keys}")

                    action_val = parsed_nlu_result.get("action")
                    place_coords_val = parsed_nlu_result.get("place_coordinates")

                    if action_val in ["pick_and_place", "place"]:
                        if place_coords_val is not None and not (isinstance(place_coords_val, dict) and \
                                  "x" in place_coords_val and "y" in place_coords_val and \
                                  isinstance(place_coords_val["x"], (int, float)) and \
                                  isinstance(place_coords_val["y"], (int, float))):
                            raise ValueError("NLU JSON中 'place_coordinates' 格式不正确。")
                    elif place_coords_val is not None:
                         print(f"DEEPSEEK_NLU_WARN: Action is '{action_val}' but 'place_coordinates' is not null: {place_coords_val}")

                    with state.app_lock:
                        state.last_nlu_result_text = parsed_nlu_result.get("feedback_text", "理解成功，但无反馈文本。")
                        state.last_nlu_status = "指令理解成功"
                    return parsed_nlu_result
                
                except (json.JSONDecodeError, ValueError) as e_parse:
                    print(f"DEEPSEEK_NLU_ERROR: 解析模型输出为JSON失败: {e_parse}. Cleaned content: '{pure_json_str}'")
                    with state.app_lock:
                        state.last_nlu_result_text = f"理解结果格式错误: {e_parse}"
                        state.last_nlu_status = "NLU结果格式错误"
                    # 返回一个inform类型的默认结果，其feedback_text是错误信息，但通常不适合TTS
                    return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": f"抱歉，我解析指令时遇到了格式问题。"}
            else:
                print(f"DEEPSEEK_NLU_ERROR: 'content' 字段为空或不存在。")
                with state.app_lock:
                    state.last_nlu_result_text = "NLU服务未返回有效内容。"
                    state.last_nlu_status = "NLU无有效内容"
                return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": "抱歉，NLU服务未返回有效内容。"}

        error_msg = "NLU响应格式不符合预期 (choices或message字段缺失)。"
        print(f"DEEPSEEK_NLU_ERROR: {error_msg} Response: {response_json}")
        with state.app_lock:
            state.last_nlu_result_text = error_msg
            state.last_nlu_status = "NLU响应格式错误"
        return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": f"抱歉，我暂时无法处理您的请求（响应结构错误）。"}

    except requests.exceptions.RequestException as e:
        print(f"DEEPSEEK_NLU_ERROR: 请求失败: {e}")
        with state.app_lock:
            state.last_nlu_result_text = "网络请求失败。"
            state.last_nlu_status = "NLU网络错误"
        return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": "抱歉，我现在无法连接到理解服务。"}
    except Exception as e:
        print(f"DEEPSEEK_NLU_ERROR: 未知错误: {e}")
        traceback.print_exc() 
        with state.app_lock:
            state.last_nlu_result_text = "发生未知内部错误。"
            state.last_nlu_status = "NLU未知错误"
        return {"action": "inform", "pick_object": None, "place_coordinates": None, "feedback_text": "抱歉，处理您的指令时发生了内部错误。"}