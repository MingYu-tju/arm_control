# state.py
import threading

# 导入配置以获取初始状态
import config

# --- 线程与锁 ---
app_lock = threading.Lock()
camera_lock = threading.Lock()
stop_threads_flag = False

# --- 硬件与系统实例 ---
my_arm_instance = None
stm32_serial_conn = None
tpu_instance = None
camera = None
pil_font = None

# --- 线程对象 ---
stm32_reader_thread_obj = None
arm_bg_thread_obj = None
object_detection_thread_obj = None

# --- 动态状态变量 ---
current_arm_state = config.ARM_STATE_IDLE
current_trajectory_index = 0
was_running_when_homing_triggered = False
stm32_current_light_status = "ALL_OFF"
current_pump_status = "OFF"
current_pump_angle = 0
detected_objects_list = []

# --- 语音与NLU状态 ---
last_voice_command_text = "待识别..."
last_nlu_result_text = "待处理..." # NLU的 feedback_text
last_nlu_status = "就绪"

# --- 新增：TTS 相关状态 ---
# pending_tts_audio_data: bytes = None # 不直接存二进制，避免状态过大
# pending_tts_text: str = "" # 要播放的文本
# Instead of storing audio data in state, we'll use a flag and a text identifier
# This allows frontend to request the specific audio when ready.
# This also makes the /status response lighter.

# Let's use a simpler approach for now: a flag in /status will tell frontend to fetch audio for last_nlu_result_text
# if that text is deemed playable by the frontend.
# The actual TTS audio will be fetched via /get_tts_audio?text=... as before.
# The key is ensuring last_nlu_result_text and last_nlu_status are reliably updated by backend tasks.

# No new state variables needed in state.py for THIS specific backend-driven TTS triggering approach,
# if we rely on frontend polling last_nlu_result_text and last_nlu_status.
# The core issue is ensuring these two are updated correctly and timely after ASR->NLU from recording.

# Let's re-verify the original problem statement:
# "录音操作完成后前端依然显示音频已上传，正在等待后端解析"
# This means last_nlu_status (and possibly last_nlu_result_text) are STUCK.

# If the backend *is* updating them, but frontend isn't reacting, that's a frontend logic issue.
# If the backend *is not* updating them after a recording NLU, that's a backend logic issue.

# Let's assume for a moment the backend *is* updating them, and re-focus on frontend trigger for simplicity.
# The previous script.js was designed to react to changes in last_nlu_result_text and last_nlu_status.
# The problem might be more subtle.

# One more check for tasks.py:
# Is it possible that the `tasks.process_audio_and_execute_command_task` thread
# finishes *before* the `services.call_deepseek_nlu` fully updates the state,
# or if an error occurs within NLU that isn't properly setting a final error state?

# Let's ensure `services.call_deepseek_nlu` ALWAYS sets a "final" nlu_status.
# And `tasks.py` correctly reflects this.