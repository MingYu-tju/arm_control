from flask import Flask, render_template, request, jsonify, Response
import time
import serial
import threading
import cv2
import os
import json

from arm_module import Arm
from TPU import TPUFactory # 确保 TPU.py 和 TPUFactory 类存在且可用

# --- 全局配置 ---
STM32_SERIAL_PORT = "/dev/ttyUSB0" # 根据实际情况修改
STM32_SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 0.1
DETECTION_ENABLED = True
ARM_TRANSLATE_X_STEP = 20
ARM_TRANSLATE_Y_STEP = 20
ARM_TRANSLATE_Z_STEP = 15
ARM_MOVE_TIME = 800 # 之前是 400，根据你的代码改回 800
ARM_RESET_TIME = 2000
ARM_HOME_POS_X = 0
ARM_HOME_POS_Y = -115
ARM_HOME_POS_Z = 240
PICK_HEIGHT = 62

PUMP_ROTATE_SMALL_STEP = 15

ARM_STATE_IDLE = "IDLE"
ARM_STATE_RUNNING_TRAJECTORY = "RUNNING_TRAJECTORY"
ARM_STATE_PAUSED_TRAJECTORY = "PAUSED_TRAJECTORY"
ARM_STATE_HOMING = "HOMING"
ARM_STATE_MANUAL_MOVE = "MANUAL_MOVE"
ARM_STATE_PICKING = "PICKING"
ARM_STATE_VOICE_CONTROL_PICKING = "VOICE_CONTROL_PICKING" # 新增状态

# --- 物体检测相关配置 ---
MODEL_CONFIG_DATA = {
    "model_id": "b44a4f80503e405795f1a64e603bc2dd",
    "classes": ["甘蔗", "贝壳", "旧帽子", "砖块", "瓦片", "西梅核", "坏垃圾桶", "煤气罐", "胶水", "饼干", "废电脑", "口服液", "油漆", "旧玩偶", "杀虫剂", "面包", "废电池", "纸盒", "水彩笔", "旧镜子", "西红柿", "扫把", "坏的花盆", "梨", "玻璃瓶", "骨头", "医用手套", "化妆品", "旧书包", "香水瓶", "漆桶", "番茄酱", "过期药片", "蛋糕", "茄子", "过期化妆品", "罐子", "报纸", "雨伞", "渣土", "蓄电池", "旧塑料篮子", "虾壳", "荧光灯", "农药瓶", "废温度计", "菜叶", "牙刷"],
    "conf": { # 与 refuse_classify.py 的参数对齐
        "imgsz": [640, 640],
        "confThres": 0.7, # 修改为 0.7
        "iouThres": 0.6   # 修改为 0.6
    },
    "model_version": "V3",
    "modelType": 1
}

CLASSES_FROM_MODEL = MODEL_CONFIG_DATA["classes"]
NUM_CLASSES_FROM_MODEL = len(CLASSES_FROM_MODEL)
DET_IMGSZ = MODEL_CONFIG_DATA["conf"]["imgsz"]
DET_CONF = MODEL_CONFIG_DATA["conf"]["confThres"] # 使用上面修改后的值
DET_IOU = MODEL_CONFIG_DATA["conf"]["iouThres"]   # 使用上面修改后的值

AI_APP_NAME_LIST = ['det_app', 'cls_app']
MODEL_INIT_PARAMS = {
    "ai_app_name": AI_APP_NAME_LIST[MODEL_CONFIG_DATA["modelType"] - 1],
    "num_classes": NUM_CLASSES_FROM_MODEL,
    "model_name": 'refuse_classify.lfdata',     # <--- 确保这是你正确的模型文件名
    "model_conf_name": 'refuse_classify.conf', # <--- 确保这是你正确的配置文件名
    "model_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "./model"),
    "model_size": (3, DET_IMGSZ[1], DET_IMGSZ[0]),
}

# --- 全局变量 ---
app_lock = threading.Lock()
current_arm_state = ARM_STATE_IDLE
# ... (其他全局变量保持不变) ...
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
CAMERA_INDEX = 0 # 摄像头索引，根据实际情况调整

app = Flask(__name__)

# --- 辅助函数：执行拾取动作 ---
def perform_pick_action(target_object_name, pixel_x, pixel_y_original):
    global current_arm_state, current_pump_status, my_arm_instance
    
    pixel_y_adjusted = pixel_y_original - 20

    print(f"  ACTION: Attempting to pick '{target_object_name}' at pixel_x={pixel_x}, pixel_y_adjusted={pixel_y_adjusted}")
    message = ""
    success = True
    try:
        # 注意：这里的机械臂动作是阻塞的
        print(f"  ACTION_PICK: Moving to suck at ({pixel_x}, {pixel_y_adjusted}) at height {PICK_HEIGHT}")
        _, final_coord = my_arm_instance.move_pixel(pixel_x, pixel_y_adjusted, PICK_HEIGHT)
        my_arm_instance.suck_up()
        time.sleep(0.8) 

        print(f"  ACTION_PICK: Lifting object from ({final_coord.x:.2f}, {final_coord.y:.2f})")
        my_arm_instance.move_point(final_coord.x, final_coord.y, final_coord.z + 80, 800)
        time.sleep(0.5)
        
        print(f"  ACTION_PICK: Moving to drop-off area (near HOME)")
        my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y - 30, ARM_HOME_POS_Z + 50, 1500)
        time.sleep(0.2)
        
        print(f"  ACTION_PICK: Releasing object.")
        my_arm_instance.suck_release()
        time.sleep(0.5)
        
        my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, 1000)
        message = f"'{target_object_name}' 拾取完成。"
        print(f"  ACTION_PICK_SUCCESS: {message}")
        
    except Exception as e_pick_action:
        message = f"拾取 '{target_object_name}' 过程中发生错误: {e_pick_action}"
        print(f"  ACTION_PICK_ERROR: {message}")
        success = False
        try: my_arm_instance.suck_release()
        except: pass
    finally:
        # 这一部分必须在 perform_pick_action 完成后，且能够访问全局状态时执行
        # 所以它仍然需要锁
        with app_lock:
            current_arm_state = ARM_STATE_IDLE
            current_pump_status = "OFF"
            print(f"  ACTION_FINALLY: Arm state set to IDLE, pump OFF. Lock released by perform_pick_action's finally.")
    return success, message


# --- STM32 串口读取线程 (保持不变) ---
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
                                        print(f"  [STATUS UPDATE WebUI] STM32 Light: '{stm32_current_light_status}' -> '{new_status}'")
                                        stm32_current_light_status = new_status
                                        if new_status == "RED_ON":
                                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY: was_running_when_homing_triggered = True
                                            else: was_running_when_homing_triggered = False
                                            current_arm_state = ARM_STATE_HOMING
                                        elif new_status == "GREEN_ON":
                                            if current_arm_state != ARM_STATE_HOMING:
                                                if current_arm_state in [ARM_STATE_IDLE, ARM_STATE_PAUSED_TRAJECTORY, ARM_STATE_RUNNING_TRAJECTORY, ARM_STATE_PICKING, ARM_STATE_VOICE_CONTROL_PICKING]:
                                                    current_arm_state = ARM_STATE_RUNNING_TRAJECTORY; current_trajectory_index = 0; was_running_when_homing_triggered = False
                                        elif new_status == "YELLOW_ON":
                                            if current_arm_state == ARM_STATE_PAUSED_TRAJECTORY:
                                                if current_trajectory_index < TOTAL_TRAJECTORY_POINTS: current_arm_state = ARM_STATE_RUNNING_TRAJECTORY; was_running_when_homing_triggered = False
                                                else: current_arm_state = ARM_STATE_IDLE; current_trajectory_index = 0
                                        elif new_status == "ALL_OFF":
                                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY: current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
                        except Exception as decode_err: print(f"[STM32 RX Thread WebUI] Decode Error: {decode_err}, Raw: {line_bytes!r}")
            else: time.sleep(0.02)
        except serial.SerialException:
            if not stop_threads_flag: print(f"[STM32 RX Thread WebUI] SerialException.")
            break
        except Exception as e:
            if not stop_threads_flag: print(f"[STM32 RX Thread WebUI] Error: {e}.")
            break
    print("[STM32 RX Thread WebUI] Stopped.")

# --- 发送指令到 STM32 (保持不变) ---
def send_command_to_stm32(ser, command_str):
    try:
        full_command = command_str + "\n"
        # print(f"PY TX (WebUI) -> STM32: {command_str}") # 减少打印
        if ser and ser.is_open: ser.write(full_command.encode('ascii'))
        else: print("  错误 (WebUI): STM32串口未连接。"); return False
        return True
    except Exception as e: print(f"发送指令到STM32错误 (WebUI): {e}"); return False

# --- 物体检测线程 (保持不变) ---
def object_detection_task():
    global detected_objects_list, tpu_instance, camera
    print("[Object Detection Task WebUI] Started.")
    frame_counter = 0
    DETECTION_INTERVAL = 5
    while not stop_threads_flag:
        if not DETECTION_ENABLED or tpu_instance is None: time.sleep(0.5); continue
        img_for_detection = None
        with camera_lock:
            if camera and camera.isOpened():
                ret, frame = camera.read()
                if ret: img_for_detection = frame.copy()
            else: time.sleep(1); continue
        if img_for_detection is None: time.sleep(0.1); continue
        frame_counter += 1
        if frame_counter % DETECTION_INTERVAL != 0: time.sleep(0.04); continue
        try:
            # print(f"DEBUG app.py: camera.read() img_for_detection shape: {img_for_detection.shape if img_for_detection is not None else 'None'}")
            # print(f"DEBUG app.py: tpu_instance.predict params - DET_IMGSZ: {DET_IMGSZ}, DET_CONF: {DET_CONF}, DET_IOU: {DET_IOU}")
            boxes = tpu_instance.predict(img_for_detection, DET_IMGSZ, DET_CONF, DET_IOU)
            current_detected = []
            if boxes:
                # if बॉक्सेस: print(f"DEBUG app.py: tpu_instance.predict output box_data[0] pixelX,Y: {boxes[0][0]}, {boxes[0][1]}")
                for i, box_data in enumerate(boxes):
                    x_center, y_center, w, h, score, label_id_int = box_data
                    label_id = int(label_id_int)
                    label_name = CLASSES_FROM_MODEL[label_id] if 0 <= label_id < len(CLASSES_FROM_MODEL) else f"未知ID({label_id})"
                    current_detected.append({
                        "id": f"obj_{time.time_ns()}_{i}", "name": label_name, "label_id": label_id,
                        "confidence": float(f"{score:.2f}"), "box_center_x": int(x_center),
                        "box_center_y": int(y_center), "box_width": int(w), "box_height": int(h),
                        "timestamp": time.time()
                    })
            with app_lock: detected_objects_list = current_detected
        except Exception as e:
            print(f"[Object Detection Task WebUI] Error during prediction: {e}")
            with app_lock: detected_objects_list = []
        time.sleep(0.1)
    print("[Object Detection Task WebUI] Stopped.")

# --- 机械臂后台任务 (保持不变) ---
def arm_background_task():
    global current_arm_state, current_trajectory_index, was_running_when_homing_triggered
    print("[Arm BG Task WebUI] Started.")
    while not stop_threads_flag:
        state_to_process = None
        with app_lock: state_to_process = current_arm_state
        if my_arm_instance:
            if state_to_process == ARM_STATE_HOMING:
                print("  BG_TASK: Arm state is HOMING. Executing reset...")
                try:
                    my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, ARM_RESET_TIME)
                    print("  BG_TASK: Arm homing complete.")
                except Exception as e_home: print(f"  BG_TASK_ERROR: Homing failed: {e_home}")
                with app_lock:
                    if was_running_when_homing_triggered and stm32_current_light_status == "RED_ON": current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
                    else: current_arm_state = ARM_STATE_IDLE; current_trajectory_index = 0
                    was_running_when_homing_triggered = False
            elif state_to_process == ARM_STATE_RUNNING_TRAJECTORY:
                can_run_step, point_data, temp_traj_index = False, None, -1
                with app_lock:
                    if current_trajectory_index < TOTAL_TRAJECTORY_POINTS:
                        can_run_step, point_data, temp_traj_index = True, PREDEFINED_TRAJECTORY[current_trajectory_index], current_trajectory_index
                    else: current_arm_state = ARM_STATE_IDLE; current_trajectory_index = 0
                if can_run_step and point_data:
                    target_x, target_y, target_z, move_duration = point_data
                    try:
                        my_arm_instance.move_point(target_x, target_y, target_z, move_duration)
                        with app_lock:
                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY and current_trajectory_index == temp_traj_index:
                                current_trajectory_index += 1
                                if current_trajectory_index >= TOTAL_TRAJECTORY_POINTS: current_arm_state = ARM_STATE_IDLE; current_trajectory_index = 0
                    except Exception as e_traj:
                        print(f"  BG_TASK_ERROR: Moving to trajectory point {temp_traj_index + 1} failed: {e_traj}")
                        with app_lock: current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
        time.sleep(0.1)
    print("[Arm BG Task WebUI] Stopped.")

# --- 摄像头帧生成器 (保持不变) ---
def generate_frames():
    global camera, detected_objects_list
    while not stop_threads_flag:
        frame_to_yield, current_boxes_for_overlay = None, []
        with app_lock:
            current_boxes_for_overlay = [(obj['box_center_x'], obj['box_center_y'], obj['box_width'], obj['box_height'], obj['name'], obj['confidence']) for obj in detected_objects_list]
        with camera_lock:
            if camera is None or not camera.isOpened():
                try:
                    camera = cv2.VideoCapture(CAMERA_INDEX)
                    if not camera.isOpened(): time.sleep(0.5); continue
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                except Exception as e_cam: print(f"Camera re-init error: {e_cam}"); camera = None; time.sleep(1); continue
            success, frame = camera.read()
            if success:
                for (x_center, y_center, w, h, name, score) in current_boxes_for_overlay:
                    x1, y1, x2, y2 = int(x_center - w / 2), int(y_center - h / 2), int(x_center + w / 2), int(y_center + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret: frame_to_yield = buffer.tobytes()
        if frame_to_yield: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')
        time.sleep(1/25)

# --- Flask 路由 ---
@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control_arm_route():
    global current_arm_state, current_trajectory_index, was_running_when_homing_triggered, current_pump_status, current_pump_angle, detected_objects_list
    
    data = request.get_json()
    command = data.get('command')
    payload = data.get('payload', {})
    message = f"Received command: {command}"
    success = True
    status_code = 200

    # --- Voice Command Processing (NLU) ---
    if command == 'process_voice_text':
        text_command = payload.get('text', '').strip()
        print(f"VOICE_NLU: Received text: '{text_command}'")
        target_object_to_pick = None
        
        for class_name in CLASSES_FROM_MODEL:
            if class_name in text_command:
                target_object_to_pick = class_name
                print(f"VOICE_NLU: Found target '{target_object_to_pick}' in command.")
                break
        
        if target_object_to_pick:
            # 准备执行拾取的参数
            object_instance_to_pick = None
            with app_lock: # 只需要在查找 detected_objects_list 和检查/设置初始状态时加锁
                if current_arm_state != ARM_STATE_IDLE:
                    message = f"机械臂当前状态为 {current_arm_state}, 不能执行语音拾取。"
                    return jsonify({'status': 'error', 'message': message}), 400
                
                best_confidence = -1.0
                temp_detected_list = list(detected_objects_list) # 复制以在锁外安全迭代（如果需要的话，但这里迭代在锁内）
                for obj_instance in temp_detected_list:
                    if obj_instance['name'] == target_object_to_pick:
                        if obj_instance['confidence'] > best_confidence:
                            best_confidence = obj_instance['confidence']
                            object_instance_to_pick = obj_instance # 保存整个对象实例
                
                if object_instance_to_pick:
                    current_arm_state = ARM_STATE_VOICE_CONTROL_PICKING # 设置状态
                    message = f"语音指令识别: 准备拾取 {object_instance_to_pick['name']}."
                    print(message)
                else:
                    message = f"语音指令: 未在当前画面中找到物体 '{target_object_to_pick}'。"
                    success = False; status_code = 404
                    print(message)
                    # 直接返回，因为没有找到物体，锁会自动释放
                    return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code
            
            # ---- 如果找到了物体并且状态已设置，现在在锁之外调用阻塞的拾取动作 ----
            if object_instance_to_pick and current_arm_state == ARM_STATE_VOICE_CONTROL_PICKING: # 再次确认状态
                print(f"VOICE_NLU: Invoking perform_pick_action for {object_instance_to_pick['name']} outside main lock.")
                # perform_pick_action 内部会处理自己的锁和最终状态重置
                success, pick_message = perform_pick_action(
                    object_instance_to_pick['name'],
                    object_instance_to_pick['box_center_x'],
                    object_instance_to_pick['box_center_y']
                )
                message = pick_message # 更新主消息
                print(f"VOICE_NLU: perform_pick_action completed. Success: {success}, Message: {message}")
            
            # 对于语音命令，无论 perform_pick_action 成功与否，都在这里返回
            return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code
            
        else: # target_object_to_pick is None
            message = f"语音指令: 未识别到有效拾取对象在指令 '{text_command}' 中。"
            success = False; status_code = 400
            print(message)
            return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

    # --- 其他命令的处理逻辑 ---
    # (STM32 LED set commands)
    if command.startswith('set_'):
        with app_lock:
            if command == 'set_green_on': success = send_command_to_stm32(stm32_serial_conn, "SET_GREEN_ON")
            elif command == 'set_red_on': success = send_command_to_stm32(stm32_serial_conn, "SET_RED_ON")
            elif command == 'set_yellow_on': success = send_command_to_stm32(stm32_serial_conn, "SET_YELLOW_ON")
            elif command == 'set_lights_off': success = send_command_to_stm32(stm32_serial_conn, "SET_LIGHTS_OFF")
            else: success = False; message = "Unknown set command"
        if not success: status_code = 500; message = f"Failed to send {command}"
        return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

    if not my_arm_instance:
        return jsonify({'status': 'error', 'message': '机械臂未初始化!'}), 500

    # ---- UI Triggered Pick Object ----
    if command == 'pick_object':
        pixel_x = payload.get('pixel_x')
        pixel_y = payload.get('pixel_y')
        obj_name = payload.get('name', '未知物体')
        
        can_pick_ui = False
        with app_lock: # 检查状态
            if current_arm_state == ARM_STATE_IDLE:
                current_arm_state = ARM_STATE_PICKING # 设置状态
                can_pick_ui = True
                message = f"UI指令: 准备拾取 {obj_name}."
                print(message)
            else:
                message = f"机械臂当前状态为 {current_arm_state}, 不能执行UI拾取。"
                success = False; status_code = 400
        
        if can_pick_ui:
            if pixel_x is None or pixel_y is None:
                message = "UI拾取失败：未提供像素坐标。" ; success = False; status_code = 400
                with app_lock: current_arm_state = ARM_STATE_IDLE # 出错恢复
            else:
                print(f"UI_PICK: Invoking perform_pick_action for {obj_name} outside main lock.")
                success, pick_message = perform_pick_action(obj_name, pixel_x, pixel_y)
                message = pick_message
                print(f"UI_PICK: perform_pick_action completed. Success: {success}, Message: {message}")
        
        return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

    # ---- Manual, Pump, Task Commands ----
    # 这些命令通常很快，或者它们的状态转换由后台任务处理
    # 它们可以在一个try-except-finally块中处理，确保锁和状态的正确性
    original_manual_state = None # 用于恢复手动操作后的状态
    try:
        with app_lock:
            if command.startswith('manual_'):
                if current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_PICKING, ARM_STATE_VOICE_CONTROL_PICKING]: # 允许从这些完成状态调整
                    if command != 'manual_reset':
                        return jsonify({'status': 'error', 'message': f'机械臂当前状态为 {current_arm_state}, 不能执行手动操作。'}), 400
                original_manual_state = current_arm_state # 记录进入手动前的状态
                current_arm_state = ARM_STATE_MANUAL_MOVE
            elif command == 'manual_reset':
                current_arm_state = ARM_STATE_HOMING
                was_running_when_homing_triggered = False; current_trajectory_index = 0
                message = "Arm Reset command submitted."
                return jsonify({'status': 'success', 'message': message}), 200 # Reset提交后直接返回
        
        # 执行动作 (这些应该相对较快，或者它们不应该长时间阻塞此请求)
        if command == 'manual_x_plus': my_arm_instance.move_axis_point(0, ARM_TRANSLATE_X_STEP, ARM_MOVE_TIME); message = "Arm X+"
        elif command == 'manual_x_minus': my_arm_instance.move_axis_point(0, -ARM_TRANSLATE_X_STEP, ARM_MOVE_TIME); message = "Arm X-"
        elif command == 'manual_y_plus': my_arm_instance.move_axis_point(1, ARM_TRANSLATE_Y_STEP, ARM_MOVE_TIME); message = "Arm Y+"
        elif command == 'manual_y_minus': my_arm_instance.move_axis_point(1, -ARM_TRANSLATE_Y_STEP, ARM_MOVE_TIME); message = "Arm Y-"
        elif command == 'manual_z_plus': my_arm_instance.move_axis_point(2, ARM_TRANSLATE_Z_STEP, ARM_MOVE_TIME); message = "Arm Z+"
        elif command == 'manual_z_minus': my_arm_instance.move_axis_point(2, -ARM_TRANSLATE_Z_STEP, ARM_MOVE_TIME); message = "Arm Z-"
        
        elif command == 'pump_on':
            my_arm_instance.suck_up()
            with app_lock: current_pump_status = "ON"; message = "Pump ON"
        elif command == 'pump_off':
            my_arm_instance.suck_release()
            with app_lock: current_pump_status = "OFF"; message = "Pump OFF"
        elif command == 'pump_angle_set':
            angle = payload.get('angle', current_pump_angle)
            my_arm_instance.rotate_angle(angle)
            with app_lock: current_pump_angle = angle; message = f"Pump Angle set to {angle}°"
        elif command == 'pump_rotate_left_small':
            with app_lock: new_angle = max(0, current_pump_angle - PUMP_ROTATE_SMALL_STEP)
            my_arm_instance.rotate_angle(new_angle)
            with app_lock: current_pump_angle = new_angle; message = f"Pump Rotated Left to {new_angle}°"
        elif command == 'pump_rotate_right_small':
            with app_lock: new_angle = min(180, current_pump_angle + PUMP_ROTATE_SMALL_STEP)
            my_arm_instance.rotate_angle(new_angle)
            with app_lock: current_pump_angle = new_angle; message = f"Pump Rotated Right to {new_angle}°"
        
        elif command == 'task_pause':
            with app_lock:
                if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY: current_arm_state = ARM_STATE_PAUSED_TRAJECTORY; message = "Task Paused"
                else: success = False; message = "Cannot pause, not running trajectory"
        elif command == 'task_stop': 
            with app_lock:
                if current_arm_state not in [ARM_STATE_IDLE, ARM_STATE_HOMING]:
                    current_arm_state = ARM_STATE_HOMING; was_running_when_homing_triggered = False; current_trajectory_index = 0; message = "Task Stop (Homing initiated)"
                else: message = "Task already stopped or homing."
        elif command == 'task_continue':
            with app_lock:
                if current_arm_state == ARM_STATE_PAUSED_TRAJECTORY: current_arm_state = ARM_STATE_RUNNING_TRAJECTORY; message = "Task Continued"
                else: success = False; message = "Cannot continue, not paused"
        
        elif command == 'start_hardware_voice_recognition':
            print("Simulating hardware voice recognition start...")
            message = "Hardware voice recognition process initiated (simulated)."
        else:
            # 仅当命令不是已知类型时才标记为未知
            known_prefixes = ('set_', 'manual_', 'pump_', 'pick_', 'task_', 'process_voice_text', 'start_hardware_voice_recognition')
            if not any(command.startswith(p) for p in known_prefixes):
                message = f"Unknown arm command: {command}"; success = False; status_code = 400
    
    except Exception as e:
        message = f"Error executing command '{command}': {e}"; print(f"ERROR_CONTROL_ROUTE: {message}"); success = False; status_code = 500
        with app_lock: # General error recovery for state if action was in progress
            if current_arm_state in [ARM_STATE_PICKING, ARM_STATE_VOICE_CONTROL_PICKING, ARM_STATE_MANUAL_MOVE]:
                 current_arm_state = ARM_STATE_IDLE
    finally:
        if command.startswith('manual_') and command != 'manual_reset':
            with app_lock:
                if current_arm_state == ARM_STATE_MANUAL_MOVE:
                    # 恢复到手动操作之前的状态，如果那个状态是IDLE的话，或者直接设为IDLE
                    current_arm_state = ARM_STATE_IDLE # 简化：手动操作后总是回到 IDLE
                    # if original_manual_state == ARM_STATE_IDLE or original_manual_state == ARM_STATE_PICKING:
                    #    current_arm_state = ARM_STATE_IDLE
                    # else: # 保留之前的状态可能导致问题，不如直接IDLE
                    #    current_arm_state = original_manual_state 

    return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

# --- /status route (保持不变) ---
@app.route('/status')
def get_status():
    with app_lock:
        arm_coords_str = "[N/A]"
        if my_arm_instance:
            try:
                pos_data = my_arm_instance.get_axis_point('3')
                if pos_data and isinstance(pos_data, (list, tuple)) and len(pos_data) >= 3:
                     arm_coords_str = f"[{pos_data[0]:.1f}, {pos_data[1]:.1f}, {pos_data[2]:.1f}]"
            except Exception: arm_coords_str = "[Error Reading]"
        objects_to_send = [obj.copy() for obj in detected_objects_list]
        status_data = {
            'arm_state': current_arm_state, 'stm32_light_status': stm32_current_light_status,
            'current_trajectory_index': current_trajectory_index, 'total_trajectory_points': TOTAL_TRAJECTORY_POINTS,
            'pump_status': current_pump_status, 'pump_angle': current_pump_angle,
            'arm_coords': arm_coords_str, 'detected_objects': objects_to_send, 'timestamp': time.time()
        }
    return jsonify(status_data)

# --- init_hardware_and_camera (TPU init part should be correct now) ---
def init_hardware_and_camera():
    global my_arm_instance, stm32_serial_conn, stm32_reader_thread_obj, arm_bg_thread_obj
    global camera, current_pump_status, current_pump_angle, tpu_instance, object_detection_thread_obj
    global current_arm_state, current_trajectory_index, was_running_when_homing_triggered

    print("Initializing Arm for WebUI...")
    try:
        my_arm_instance = Arm()
        my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, ARM_RESET_TIME)
        with app_lock:
            current_arm_state = ARM_STATE_IDLE; current_trajectory_index = 0
            was_running_when_homing_triggered = False; current_pump_status = "OFF"; current_pump_angle = 0
        print("Arm initialized and at home (WebUI).")
    except Exception as e: print(f"!!! Error initializing Arm (WebUI): {e} !!!"); my_arm_instance = None

    print(f"Connecting to STM32 ({STM32_SERIAL_PORT}) for WebUI...")
    try:
        stm32_serial_conn = serial.Serial(STM32_SERIAL_PORT, STM32_SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT)
        stm32_reader_thread_obj = threading.Thread(target=stm32_reader_thread_func, args=(stm32_serial_conn,), daemon=True)
        stm32_reader_thread_obj.start()
        print("STM32 serial connected and reader thread started for WebUI.")
    except serial.SerialException as e: print(f"!!! Failed to connect to STM32 (WebUI): {e} !!!"); stm32_serial_conn = None

    print(f"Initializing Camera (index: {CAMERA_INDEX}) for WebUI...")
    with camera_lock:
        try:
            camera = cv2.VideoCapture(CAMERA_INDEX)
            if not camera.isOpened(): print(f"!!! Error: Could not open camera {CAMERA_INDEX} !!!"); camera = None
            else:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Camera {CAMERA_INDEX} initialized successfully.")
        except Exception as e: print(f"!!! Error initializing camera: {e} !!!"); camera = None
    
    if DETECTION_ENABLED:
        print("Initializing TPU for WebUI detection...")
        try:
            tpu_instance = TPUFactory(algorithm="detection")
            if not tpu_instance.link(): raise Exception("TPU link failed. Check TPU connection.")
            print(f"TPU: Initializing model with params: {MODEL_INIT_PARAMS}")
            tpu_instance.model_init(MODEL_INIT_PARAMS, host=True)
            print("TPU and model initialized successfully for WebUI.")
            object_detection_thread_obj = threading.Thread(target=object_detection_task, daemon=True)
            object_detection_thread_obj.start()
            print("Object detection thread started for WebUI.")
        except Exception as e:
            print(f"!!! Error initializing TPU or detection model (WebUI): {e} !!!")
            tpu_instance = None
    else: print("Object detection is DISABLED.")

    arm_bg_thread_obj = threading.Thread(target=arm_background_task, daemon=True)
    arm_bg_thread_obj.start()
    print("Arm background task thread started for WebUI.")

# --- Main execution and cleanup (保持不变) ---
if __name__ == '__main__':
    init_hardware_and_camera()
    print("Starting Flask web server on http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    print("Flask server shutting down...")
    stop_threads_flag = True
    with camera_lock:
        if camera and camera.isOpened(): camera.release(); print("Camera released.")
    if stm32_reader_thread_obj and stm32_reader_thread_obj.is_alive(): stm32_reader_thread_obj.join(timeout=1)
    if arm_bg_thread_obj and arm_bg_thread_obj.is_alive(): arm_bg_thread_obj.join(timeout=1)
    if object_detection_thread_obj and object_detection_thread_obj.is_alive(): object_detection_thread_obj.join(timeout=1)
    if tpu_instance:
        try:
            if hasattr(tpu_instance, 'release') and callable(getattr(tpu_instance, 'release')):
                tpu_instance.release(); print("TPU released.")
            else: print("TPU instance does not have a callable 'release' method.")
        except Exception as e_tpu_rel: print(f"Error releasing TPU: {e_tpu_rel}")
    if stm32_serial_conn and stm32_serial_conn.is_open:
        try: send_command_to_stm32(stm32_serial_conn, "SET_LIGHTS_OFF"); stm32_serial_conn.close()
        except: pass
    print("WebUI Application finished.")