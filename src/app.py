# app.py

from flask import Flask, render_template, request, jsonify, Response
import threading
import time
import serial
import cv2
import os

# 导入您自己的外部模块
from arm_module import Arm
from tpu_module import Control_TPU 

# 导入拆分后的项目模块
import config
import state
import utils
import tasks
import services 
from PIL import ImageFont

app = Flask(__name__)

# --- Flask Routes (Web接口定义) ---

@app.route('/')
def index_page():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """提供视频流的路由"""
    return Response(tasks.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_voice_audio', methods=['POST'])
def upload_voice_audio_route():
    """处理上传的语音文件，并启动后台处理任务"""
    if 'audio_data' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio data in request.'}), 400
    audio_file = request.files['audio_data']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file.'}), 400

    original_save_path = config.WAVE_OUTPUT_FILENAME 
    try:
        audio_file.save(original_save_path)
        print(f"AUDIO_UPLOAD: Original audio file saved to {original_save_path}")
    except Exception as e:
        print(f"AUDIO_UPLOAD_ERROR: Failed to save audio file: {e}")
        return jsonify({'status': 'error', 'message': f'Failed to save audio: {e}'}), 500

    with state.app_lock:
        if state.current_arm_state not in [
            config.ARM_STATE_IDLE, 
            config.ARM_STATE_RECOGNIZING_VOICE, 
            config.ARM_STATE_UNDERSTANDING_NLU
        ]:
            message = f"机械臂忙 ({state.current_arm_state}), 无法处理语音。"
            if os.path.exists(original_save_path):
                try: os.remove(original_save_path)
                except Exception as e_del: print(f"AUDIO_CLEANUP_WARN: {e_del}")
            return jsonify({'status': 'error', 'message': message}), 409
        
        # 与 tasks.py 中 ASR 前设置的状态一致或兼容
        state.last_voice_command_text = "上传识别中..." 
        state.last_nlu_status = "音频已接收，准备处理..." # tasks.py 会覆盖为 "正在识别语音..."
        state.last_nlu_result_text = "待处理..." 
    
    # ... (启动线程) ...

    voice_processing_thread = threading.Thread(
        target=tasks.process_audio_and_execute_command_task, 
        args=(original_save_path,),
        daemon=True
    )
    voice_processing_thread.start()
    
    return jsonify({'status': 'success', 'message': "音频已上传，正在后台处理..."}), 202 # 这个message是给前端fetch的即时响应


@app.route('/get_tts_audio', methods=['GET'])
def get_tts_audio_route():
    text_to_synthesize = request.args.get('text', '').strip()
    if not text_to_synthesize:
        return jsonify({'status': 'error', 'message': 'No text provided for TTS.'}), 400

    audio_content = services.synthesize_speech_youdao(text_to_synthesize)

    if audio_content:
        return Response(audio_content, mimetype='audio/mp3')
    else:
        return jsonify({'status': 'error', 'message': 'TTS synthesis failed.'}), 500


@app.route('/control', methods=['POST'])
def control_arm_route():
    data = request.get_json()
    command = data.get('command')
    payload = data.get('payload', {})
    message_to_client = f"CMD_RECV: {command}" 
    success = True
    status_code = 200
    response_data = {} 

    if command == 'process_voice_text': 
        text_command = payload.get('text', '').strip()
        if not text_command:
            response_data = {'status': 'error', 'message': '模拟指令文本不能为空。'}
            return jsonify(response_data), 400
        
        print(f"  SIM_NLU_CTRL: Received simulated text: '{text_command}'")
        
        with state.app_lock:
            if state.current_arm_state not in [config.ARM_STATE_IDLE, config.ARM_STATE_UNDERSTANDING_NLU, config.ARM_STATE_RECOGNIZING_VOICE]:
                message_to_client = f"机械臂忙 ({state.current_arm_state}), 无法处理模拟指令。"
                response_data = {'status': 'error', 'message': message_to_client}
                return jsonify(response_data), 409
            
            state.last_voice_command_text = text_command 
            state.current_arm_state = config.ARM_STATE_UNDERSTANDING_NLU
            state.last_nlu_status = "正在理解模拟指令..."
            state.last_nlu_result_text = "处理中..."

        nlu_output = services.call_deepseek_nlu(text_command)

        with state.app_lock: 
            message_to_client = state.last_nlu_result_text 
            current_nlu_status = state.last_nlu_status

        if not nlu_output or current_nlu_status != "指令理解成功":
            success = False
            status_code = 500 if not nlu_output else 400 
            if "格式错误" in current_nlu_status or "无有效内容" in current_nlu_status or "响应格式错误" in current_nlu_status :
                 status_code = 400
            with state.app_lock:
                if state.current_arm_state == config.ARM_STATE_UNDERSTANDING_NLU:
                    state.current_arm_state = config.ARM_STATE_IDLE
            response_data = {'status': 'error', 'message': message_to_client} 
            return jsonify(response_data), status_code

        action = nlu_output.get("action")
        pick_object_name_from_nlu = nlu_output.get("pick_object")
        place_coordinates_from_nlu = nlu_output.get("place_coordinates")
        
        if action in ["pick_and_place", "pick"]:
            if not pick_object_name_from_nlu:
                with state.app_lock: 
                    state.last_nlu_status = "NLU结果: 未指定拾取物体" 
                    state.current_arm_state = config.ARM_STATE_IDLE
                success = False
                status_code = 400
            else:
                object_instance_to_pick = None
                with state.app_lock:
                    if state.current_arm_state not in [config.ARM_STATE_IDLE, config.ARM_STATE_UNDERSTANDING_NLU]:
                        err_msg = f"机械臂忙 ({state.current_arm_state}),无法执行"
                        state.last_nlu_status = err_msg
                        message_to_client = err_msg 
                        success = False
                        status_code = 409
                    else:
                        best_conf = -1.0
                        for obj_detected in list(state.detected_objects_list):
                            if pick_object_name_from_nlu.lower() in obj_detected['name'].lower() or \
                               obj_detected['name'].lower() in pick_object_name_from_nlu.lower():
                                if obj_detected['confidence'] > best_conf:
                                    best_conf = obj_detected['confidence']
                                    object_instance_to_pick = obj_detected
                        
                        if object_instance_to_pick:
                            state.current_arm_state = config.ARM_STATE_PICKING_SUBMITTED
                        else: 
                            current_feedback = message_to_client 
                            state.last_nlu_status = f"未在画面中找到'{pick_object_name_from_nlu}'" 
                            message_to_client = f"虽然我明白了您的意思（“{current_feedback}”），但在画面中没有找到'{pick_object_name_from_nlu}'。"
                            with state.app_lock: state.last_nlu_result_text = message_to_client 
                            state.current_arm_state = config.ARM_STATE_IDLE
                            success = False
                            status_code = 404
                
                if success and object_instance_to_pick:
                    final_place_coords_for_arm = None
                    if action == "pick_and_place":
                        if place_coordinates_from_nlu and isinstance(place_coordinates_from_nlu, dict) and "x" in place_coordinates_from_nlu and "y" in place_coordinates_from_nlu:
                            place_x, place_y = place_coordinates_from_nlu["x"], place_coordinates_from_nlu["y"]
                            place_z = config.TABLE_Z_PLANE + config.DROP_HEIGHT_OFFSET
                            final_place_coords_for_arm = {"x": place_x, "y": place_y, "z": place_z}
                        else:
                            default_loc = config.PREDEFINED_PLACE_LOCATIONS["默认放置区"]
                            final_place_coords_for_arm = {"x": default_loc["x"], "y": default_loc["y"], "z": default_loc["z"]}
                    
                    task_thread = threading.Thread(
                        target=tasks.threaded_pick_and_place_task, 
                        args=(object_instance_to_pick['name'], object_instance_to_pick['box_center_x'], object_instance_to_pick['box_center_y'], final_place_coords_for_arm), 
                        daemon=True)
                    task_thread.start()
                    status_code = 202 
        
        elif action == "inform" or action is None:
            with state.app_lock: state.current_arm_state = config.ARM_STATE_IDLE
        else: 
            with state.app_lock:
                state.last_nlu_status = f"未知NLU动作: {action} (模拟)"
                state.current_arm_state = config.ARM_STATE_IDLE
            message_to_client = f"我不理解这个动作：'{action}'。" 
            with state.app_lock: state.last_nlu_result_text = message_to_client
            success = False
            status_code = 400
        
        response_data = {'status': 'success' if success else 'error', 'message': message_to_client}
        return jsonify(response_data), status_code

    elif command.startswith('set_'): 
        with state.app_lock:
            if command == 'set_green_on': success = utils.send_command_to_stm32(state.stm32_serial_conn, "SET_GREEN_ON")
            elif command == 'set_red_on': success = utils.send_command_to_stm32(state.stm32_serial_conn, "SET_RED_ON")
            elif command == 'set_yellow_on': success = utils.send_command_to_stm32(state.stm32_serial_conn, "SET_YELLOW_ON")
            elif command == 'set_lights_off': success = utils.send_command_to_stm32(state.stm32_serial_conn, "SET_LIGHTS_OFF")
            else: success = False; message_to_client = "Unknown set command"
        if not success and message_to_client == f"CMD_RECV: {command}":
             message_to_client = f"Failed to send {command}"
        elif success:
            message_to_client = f"Command {command} sent to STM32."
        if not success : status_code = 500
        response_data = {'status': 'success' if success else 'error', 'message': message_to_client}

    elif not state.my_arm_instance:
        response_data = {'status': 'error', 'message': '机械臂未初始化!'}
        return jsonify(response_data), 503

    elif command == 'pick_object': 
        pixel_x, pixel_y = payload.get('pixel_x'), payload.get('pixel_y')
        obj_name = payload.get('name', '未知物体')
        if pixel_x is None or pixel_y is None:
            response_data = {'status': 'error', 'message': "未提供有效像素坐标."}
            return jsonify(response_data), 400
        
        with state.app_lock:
            if state.current_arm_state != config.ARM_STATE_IDLE:
                message_to_client = f"机械臂忙 ({state.current_arm_state})."
                response_data = {'status': 'error', 'message': message_to_client}
                return jsonify(response_data), 409
            state.current_arm_state = config.ARM_STATE_PICKING_SUBMITTED
            message_to_client = f"好的，准备通过界面指令拾取{obj_name}。" 
            state.last_nlu_status = "UI拾取" 
            state.last_nlu_result_text = message_to_client 

        place_coords_ui = config.PREDEFINED_PLACE_LOCATIONS["默认放置区"]
        threading.Thread(target=tasks.threaded_pick_and_place_task, 
                         args=(obj_name, pixel_x, pixel_y, place_coords_ui), daemon=True).start()
        status_code = 202
        response_data = {'status': 'success', 'message': message_to_client}
        
    else: 
        action_taken = False
        with state.app_lock:
            arm_busy_for_manual = state.current_arm_state not in [
                config.ARM_STATE_IDLE, config.ARM_STATE_PAUSED_TRAJECTORY, 
                config.ARM_STATE_RECOGNIZING_VOICE, config.ARM_STATE_UNDERSTANDING_NLU
            ] 
            if arm_busy_for_manual and command.startswith('manual_') and command != 'manual_reset':
                message_to_client = f"机械臂忙 ({state.current_arm_state}), 无法手动操作."
                response_data = {'status': 'error', 'message': message_to_client}
                return jsonify(response_data), 409

        if command == 'manual_reset':
            with state.app_lock:
                if state.current_arm_state != config.ARM_STATE_HOMING:
                    if state.current_arm_state in [config.ARM_STATE_RECOGNIZING_VOICE, config.ARM_STATE_UNDERSTANDING_NLU]:
                        state.last_nlu_status = "操作已取消 (复位)"
                        state.last_nlu_result_text = "用户触发复位，机械臂正在归位。" 
                    else:
                        state.last_nlu_result_text = "机械臂正在归位。"

                    state.current_arm_state = config.ARM_STATE_HOMING
                    state.was_running_when_homing_triggered = False; state.current_trajectory_index = 0
                    message_to_client = state.last_nlu_result_text 
                else: message_to_client = "机械臂已在复位中."
            action_taken = True
        elif command.startswith('manual_'):
            original_state_before_manual = ""
            with state.app_lock:
                original_state_before_manual = state.current_arm_state
                state.current_arm_state = config.ARM_STATE_MANUAL_MOVE
            try:
                if command == 'manual_x_plus': state.my_arm_instance.move_axis_point(0, config.ARM_TRANSLATE_X_STEP, config.ARM_MOVE_TIME)
                elif command == 'manual_x_minus': state.my_arm_instance.move_axis_point(0, -config.ARM_TRANSLATE_X_STEP, config.ARM_MOVE_TIME)
                elif command == 'manual_y_plus': state.my_arm_instance.move_axis_point(1, config.ARM_TRANSLATE_Y_STEP, config.ARM_MOVE_TIME)
                elif command == 'manual_y_minus': state.my_arm_instance.move_axis_point(1, -config.ARM_TRANSLATE_Y_STEP, config.ARM_MOVE_TIME)
                elif command == 'manual_z_plus': state.my_arm_instance.move_axis_point(2, config.ARM_TRANSLATE_Z_STEP, config.ARM_MOVE_TIME)
                elif command == 'manual_z_minus': state.my_arm_instance.move_axis_point(2, -config.ARM_TRANSLATE_Z_STEP, config.ARM_MOVE_TIME)
                message_to_client = f"机械臂沿 {command.split('_')[1].upper()} 轴移动。"
                with state.app_lock: state.last_nlu_result_text = message_to_client 
                action_taken = True
            except Exception as e_manual: message_to_client = f"手动操作错误: {e_manual}"; success = False; status_code = 500
            finally:
                with state.app_lock:
                    if state.current_arm_state == config.ARM_STATE_MANUAL_MOVE:
                        state.current_arm_state = original_state_before_manual if original_state_before_manual == config.ARM_STATE_PAUSED_TRAJECTORY else config.ARM_STATE_IDLE
        elif command.startswith('pump_'):
            with state.app_lock:
                if command == 'pump_on': state.my_arm_instance.suck_up(); state.current_pump_status = "ON"; message_to_client = "吸泵已启动。"
                elif command == 'pump_off': state.my_arm_instance.suck_release(); state.current_pump_status = "OFF"; message_to_client = "吸泵已停止。"
                elif command == 'pump_angle_set':
                    angle = payload.get('angle', state.current_pump_angle); state.my_arm_instance.rotate_angle(angle); state.current_pump_angle = angle; message_to_client = f"旋转头角度已设为 {angle}°。"
                elif command == 'pump_rotate_left_small':
                    new_angle = max(0, state.current_pump_angle - config.PUMP_ROTATE_SMALL_STEP); state.my_arm_instance.rotate_angle(new_angle); state.current_pump_angle = new_angle; message_to_client = f"旋转头左旋至 {new_angle}°。"
                elif command == 'pump_rotate_right_small':
                    new_angle = min(180, state.current_pump_angle + config.PUMP_ROTATE_SMALL_STEP); state.my_arm_instance.rotate_angle(new_angle); state.current_pump_angle = new_angle; message_to_client = f"旋转头右旋至 {new_angle}°。"
                else: success = False; message_to_client = f"未知吸泵指令: {command}"
                state.last_nlu_result_text = message_to_client
            action_taken = True 
        elif command == 'task_pause':
            with state.app_lock:
                if state.current_arm_state == config.ARM_STATE_RUNNING_TRAJECTORY: state.current_arm_state = config.ARM_STATE_PAUSED_TRAJECTORY; message_to_client = "任务已暂停。"
                else: success = False; message_to_client = "无法暂停: 非轨迹运行中."
                state.last_nlu_result_text = message_to_client
            action_taken = True
        elif command == 'task_stop':
            with state.app_lock:
                if state.current_arm_state not in [config.ARM_STATE_IDLE, config.ARM_STATE_HOMING]:
                    state.current_arm_state = config.ARM_STATE_HOMING
                    state.was_running_when_homing_triggered = False; state.current_trajectory_index = 0
                    message_to_client = "任务已停止并归位。"
                else: message_to_client = "任务已停止或归位中."
                state.last_nlu_result_text = message_to_client
            action_taken = True
        elif command == 'task_continue':
            with state.app_lock:
                if state.current_arm_state == config.ARM_STATE_PAUSED_TRAJECTORY: state.current_arm_state = config.ARM_STATE_RUNNING_TRAJECTORY; message_to_client = "任务已继续。"
                else: success = False; message_to_client = "无法继续: 非暂停状态."
                state.last_nlu_result_text = message_to_client
            action_taken = True

        if not action_taken and success : 
            message_to_client = f"未知指令或状态不允许: {command}"
            success = False; status_code = 400
            with state.app_lock: state.last_nlu_result_text = message_to_client #确保未知指令也有反馈
        
        if not response_data: 
            response_data = {'status': 'success' if success else 'error', 'message': message_to_client}
            
    return jsonify(response_data), status_code

@app.route('/status')
def get_status():
    with state.app_lock:
        arm_coords_str = "[N/A]"
        if state.my_arm_instance:
            try:
                pos_data = state.my_arm_instance.get_axis_point('3')
                if pos_data and isinstance(pos_data, (list, tuple)) and len(pos_data) >= 3:
                    arm_coords_str = f"[{pos_data[0]:.1f}, {pos_data[1]:.1f}, {pos_data[2]:.1f}]"
            except Exception: arm_coords_str = "[Error Reading]"
        
        objects_to_send = [obj.copy() for obj in state.detected_objects_list]

        status_data = {
            'arm_state': state.current_arm_state, 
            'stm32_light_status': state.stm32_current_light_status,
            'current_trajectory_index': state.current_trajectory_index, 
            'total_trajectory_points': config.TOTAL_TRAJECTORY_POINTS,
            'pump_status': state.current_pump_status, 
            'pump_angle': state.current_pump_angle, 
            'arm_coords': arm_coords_str,
            'detected_objects': objects_to_send,
            'voice_command_text': state.last_voice_command_text,    
            'nlu_result_text': state.last_nlu_result_text,        
            'nlu_status': state.last_nlu_status,                  
            'timestamp': time.time()
        }
    return jsonify(status_data)

def init_hardware_and_camera():
    try:
        state.pil_font = ImageFont.truetype(config.CHINESE_FONT_PATH, config.CHINESE_FONT_SIZE)
        print(f"字体初始化: 成功加载中文字体: {config.CHINESE_FONT_PATH}")
    except Exception as e:
        print(f"!!! 字体初始化警告: 加载字体时发生错误: {e}")
        state.pil_font = "FONT_LOAD_FAILED"
    
    print("Initializing Arm...")
    try:
        state.my_arm_instance = Arm()
        state.my_arm_instance.move_point(config.ARM_HOME_POS_X, config.ARM_HOME_POS_Y, config.ARM_HOME_POS_Z, config.ARM_RESET_TIME)
        with state.app_lock:
            state.current_arm_state = config.ARM_STATE_IDLE
        print("Arm initialized and at home.")
    except Exception as e:
        print(f"!!! Error initializing Arm: {e} !!!"); state.my_arm_instance = None
    
    print(f"Connecting to STM32 ({config.STM32_SERIAL_PORT})...")
    try:
        state.stm32_serial_conn = serial.Serial(config.STM32_SERIAL_PORT, config.STM32_SERIAL_BAUDRATE, timeout=config.SERIAL_TIMEOUT)
        state.stm32_reader_thread_obj = threading.Thread(target=tasks.stm32_reader_thread_func, args=(state.stm32_serial_conn,), daemon=True)
        state.stm32_reader_thread_obj.start()
        print("STM32 serial connected and reader thread started.")
    except serial.SerialException as e:
        print(f"!!! Failed to connect to STM32: {e} !!!"); state.stm32_serial_conn = None
    
    print(f"Initializing Camera (index: {config.CAMERA_INDEX})...")
    with state.camera_lock:
        try:
            state.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            if not state.camera.isOpened():
                raise IOError(f"Could not open camera {config.CAMERA_INDEX}")
            state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"Camera {config.CAMERA_INDEX} initialized successfully.")
        except Exception as e: print(f"!!! Error initializing camera: {e} !!!"); state.camera = None
            
    if config.DETECTION_ENABLED:
        print("Initializing Control_TPU for detection...")
        try:
            state.tpu_instance = Control_TPU 
            state.object_detection_thread_obj = threading.Thread(target=tasks.object_detection_task, daemon=True)
            state.object_detection_thread_obj.start()
            print("Object detection thread started.")
        except Exception as e:
            print(f"!!! Error creating/starting Control_TPU or detection thread: {e} !!!")
            state.tpu_instance = None
            
    state.arm_bg_thread_obj = threading.Thread(target=tasks.arm_background_task, daemon=True)
    state.arm_bg_thread_obj.start()
    print("Arm background task thread started.")

if __name__ == '__main__':
    if config.YOUDAO_APP_KEY == "YOUR_YOUDAO_APP_KEY" or config.YOUDAO_APP_SECRET == "YOUR_YOUDAO_APP_SECRET":
        print("\n警告: 请在 config.py 文件中设置您的有道智云 APP_KEY 和 APP_SECRET。\n")
    if config.DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY":
        print("警告: 请在 config.py 文件中设置您的 DEEPSEEK_API_KEY。\n")
        
    init_hardware_and_camera()
    print("Starting Flask web server on http://0.0.0.0:5000 ...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    finally:
        print("Flask server shutting down...")
        state.stop_threads_flag = True
        
        print("Releasing camera...")
        with state.camera_lock:
            if state.camera and state.camera.isOpened():
                state.camera.release()
                state.camera = None
            print("Camera released.")
        
        threads_to_join = [
            (state.stm32_reader_thread_obj, "STM32 reader thread"), 
            (state.arm_bg_thread_obj, "Arm background task"), 
            (state.object_detection_thread_obj, "Object detection task")
        ]
        for thread, name in threads_to_join:
            if thread and thread.is_alive():
                print(f"Waiting for {name} to terminate...")
                thread.join(timeout=3)
                if thread.is_alive(): print(f"Warning: {name} did not terminate gracefully.")
                else: print(f"{name} joined.")
        
        if state.stm32_serial_conn and state.stm32_serial_conn.is_open:
            try:
                utils.send_command_to_stm32(state.stm32_serial_conn, "SET_LIGHTS_OFF")
                time.sleep(0.1)
                state.stm32_serial_conn.close()
                print("STM32 serial closed.")
            except Exception as e:
                print(f"Error closing STM32 serial: {e}")
        
        print("Application finished.")