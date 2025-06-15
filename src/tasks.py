# tasks.py

import threading
import time
import os
import cv2 # Keep cv2 import if generate_frames needs it
import numpy as np # Keep numpy import if generate_frames needs it
from PIL import Image, ImageDraw, ImageFont # Keep PIL imports if generate_frames needs them
import serial # Keep serial import if stm32_reader_thread_func needs it
import traceback # Make sure traceback is imported

# 导入配置、状态、服务和工具函数
import config
import state
import services
# 注意：utils.py 中的函数在此文件中没有直接使用，但在其他模块中可能被调用

# --- 核心业务任务线程 ---

def threaded_pick_and_place_task(pick_object_name_from_tpu, pixel_x, pixel_y_original, place_location_coords_xyz=None):
    """
    一个完整的拾取并放置任务的线程函数。
    - pick_object_name_from_tpu: TPU确认的物体名称 (用于日志)。
    - pixel_x, pixel_y_original: 物体在摄像头画面中的中心像素坐标。
    - place_location_coords_xyz: 包含x,y,z的目标放置坐标字典，或为None则放置在默认区域。
    """
    print(f"  PICK_PLACE_TASK_START: For '{pick_object_name_from_tpu}' to {place_location_coords_xyz if place_location_coords_xyz else 'default drop'}.")
    
    pixel_y_adjusted = pixel_y_original - 20 
    try:
        with state.app_lock: state.current_arm_state = config.ARM_STATE_PICKING_MOVE_TO_SUCK
        print(f"  PICK_PLACE_TASK: Moving to suck '{pick_object_name_from_tpu}' at ({pixel_x}, {pixel_y_adjusted}), height {config.PICK_HEIGHT}")
        _, final_coord_suck = state.my_arm_instance.move_pixel(pixel_x, pixel_y_adjusted, config.PICK_HEIGHT)
        time.sleep(0.5)

        with state.app_lock: state.current_arm_state = config.ARM_STATE_PICKING_SUCKING
        state.my_arm_instance.suck_up()
        with state.app_lock: state.current_pump_status = "ON"
        time.sleep(0.8) 

        with state.app_lock: state.current_arm_state = config.ARM_STATE_PICKING_LIFTING
        lift_height = final_coord_suck.z + 80
        state.my_arm_instance.move_point(final_coord_suck.x, final_coord_suck.y, lift_height, 800)
        time.sleep(0.5)

        if place_location_coords_xyz and all(k in place_location_coords_xyz for k in ["x", "y", "z"]):
            target_drop_x, target_drop_y, target_drop_z = place_location_coords_xyz["x"], place_location_coords_xyz["y"], place_location_coords_xyz["z"]
            print(f"  PICK_PLACE_TASK: Target place (from NLU/default): ({target_drop_x}, {target_drop_y}, {target_drop_z})")
        else:
            default_loc = config.PREDEFINED_PLACE_LOCATIONS["默认放置区"]
            target_drop_x, target_drop_y, target_drop_z = default_loc["x"], default_loc["y"], default_loc["z"]
            print(f"  PICK_PLACE_TASK: No specific place coords. Using default drop-off.")

        with state.app_lock: state.current_arm_state = config.ARM_STATE_MOVING_TO_PLACE
        print(f"  PICK_PLACE_TASK: Moving to drop-off point ({target_drop_x}, {target_drop_y}, {target_drop_z})")
        state.my_arm_instance.move_point(target_drop_x, target_drop_y, max(lift_height, target_drop_z + 50) , 1200) 
        time.sleep(0.2)
        state.my_arm_instance.move_point(target_drop_x, target_drop_y, target_drop_z, 800)
        time.sleep(0.5)
        
        with state.app_lock: state.current_arm_state = config.ARM_STATE_PLACING_OBJECT
        print(f"  PICK_PLACE_TASK: Releasing object at ({target_drop_x}, {target_drop_y}, {target_drop_z}).")
        state.my_arm_instance.suck_release()
        with state.app_lock: state.current_pump_status = "OFF"
        time.sleep(0.8)

        print(f"  PICK_PLACE_TASK: Lifting after release and homing.")
        state.my_arm_instance.move_point(target_drop_x, target_drop_y, target_drop_z + 80, 500)
        time.sleep(0.2)

        with state.app_lock: state.current_arm_state = config.ARM_STATE_PICKING_HOMING_AFTER
        state.my_arm_instance.move_point(config.ARM_HOME_POS_X, config.ARM_HOME_POS_Y, config.ARM_HOME_POS_Z, 1000)
        print(f"  PICK_PLACE_TASK_SUCCESS: '{pick_object_name_from_tpu}' pick and place completed, arm homed.")
    except Exception as e_pick_place:
        print(f"  PICK_PLACE_TASK_ERROR: {e_pick_place}")
        traceback.print_exc()
        try: 
            state.my_arm_instance.suck_release()
            with state.app_lock: state.current_pump_status = "OFF"
            state.my_arm_instance.move_point(config.ARM_HOME_POS_X, config.ARM_HOME_POS_Y, config.ARM_HOME_POS_Z, 1500)
        except Exception as e_recovery:
            print(f"  PICK_PLACE_TASK_RECOVERY_ERROR: {e_recovery}")
    finally:
        with state.app_lock:
            state.current_arm_state = config.ARM_STATE_IDLE
        print(f"  PICK_PLACE_TASK_FINALLY: Task finished. Arm state IDLE.")

def process_audio_and_execute_command_task(audio_filepath_original):
    print(f"TASK_PROCESS_AUDIO: Started for '{audio_filepath_original}'")
    nlu_output_for_finally_check = None # To check if NLU was successful in finally
    try:
        print("TASK_PROCESS_AUDIO: Attempting ASR...")
        with state.app_lock:
            state.current_arm_state = config.ARM_STATE_RECOGNIZING_VOICE
            state.last_nlu_status = "正在识别语音..." 
        
        recognized_text = services.recognize_speech_youdao(audio_filepath_original)
        
        if not recognized_text:
            with state.app_lock: # Ensure we read the status set by the service
                print(f"TASK_PROCESS_AUDIO: ASR failed or returned no text. Current state.last_nlu_status='{state.last_nlu_status}'")
                if state.current_arm_state == config.ARM_STATE_RECOGNIZING_VOICE:
                    state.current_arm_state = config.ARM_STATE_IDLE
            return

        with state.app_lock:
            current_asr_text = state.last_voice_command_text # Should be same as recognized_text
            print(f"TASK_PROCESS_AUDIO: ASR successful. Recognized text='{recognized_text}'. state.last_voice_command_text='{current_asr_text}'")

        print("TASK_PROCESS_AUDIO: Attempting NLU...")
        with state.app_lock:
            state.current_arm_state = config.ARM_STATE_UNDERSTANDING_NLU
            state.last_nlu_status = "正在理解指令..."
        
        nlu_output = services.call_deepseek_nlu(recognized_text)
        nlu_output_for_finally_check = nlu_output # Store for finally block

        if not nlu_output:
            with state.app_lock:
                 print(f"TASK_PROCESS_AUDIO: NLU failed or returned no output. Current state.last_nlu_status='{state.last_nlu_status}', state.last_nlu_result_text='{state.last_nlu_result_text}'")
                 if state.current_arm_state == config.ARM_STATE_UNDERSTANDING_NLU:
                    state.current_arm_state = config.ARM_STATE_IDLE
            return
        
        with state.app_lock:
            current_nlu_status = state.last_nlu_status
            current_nlu_result_text = state.last_nlu_result_text
            print(f"TASK_PROCESS_AUDIO: NLU processing completed. NLU Output: {nlu_output}. Current state.last_nlu_status='{current_nlu_status}', state.last_nlu_result_text='{current_nlu_result_text}'")

        action = nlu_output.get("action")
        pick_object_name_from_nlu = nlu_output.get("pick_object")
        # place_coordinates_from_nlu = nlu_output.get("place_coordinates") # Already got this
        
        print(f"TASK_PROCESS_AUDIO: NLU Action='{action}', Object='{pick_object_name_from_nlu}'")

        if action in ["pick_and_place", "pick"]:
            if not pick_object_name_from_nlu:
                with state.app_lock:
                    # Assuming NLU service itself sets a good feedback_text for this case.
                    # We can make the status more specific here if needed.
                    if state.last_nlu_status == "指令理解成功": # If NLU thought it was a success but object is missing
                         state.last_nlu_status = "NLU结果: 未指定拾取物体"
                    state.current_arm_state = config.ARM_STATE_IDLE
                print("TASK_PROCESS_AUDIO: NLU action requires object, but none specified by NLU.")
                return

            object_instance_to_pick = None
            current_detected_objects_snapshot = []
            with state.app_lock:
                 current_detected_objects_snapshot = list(state.detected_objects_list)
                 if state.current_arm_state not in [config.ARM_STATE_IDLE, config.ARM_STATE_UNDERSTANDING_NLU]:
                    err_msg = f"机械臂忙 ({state.current_arm_state}),无法执行拾取"
                    state.last_nlu_status = err_msg 
                    state.last_nlu_result_text = err_msg 
                    print(f"TASK_PROCESS_AUDIO: Arm busy: {err_msg}")
                    return

            best_conf = -1.0
            for obj_detected in current_detected_objects_snapshot:
                if pick_object_name_from_nlu.lower() in obj_detected['name'].lower() or \
                   obj_detected['name'].lower() in pick_object_name_from_nlu.lower():
                    if obj_detected['confidence'] > best_conf:
                        best_conf = obj_detected['confidence']
                        object_instance_to_pick = obj_detected
            
            if object_instance_to_pick:
                with state.app_lock:
                    state.current_arm_state = config.ARM_STATE_PICKING_SUBMITTED
                    # NLU service should have set an appropriate feedback_text like "好的，正在拾取..."
                    # state.last_nlu_status should remain "指令理解成功"
                print(f"TASK_PROCESS_AUDIO: Found '{object_instance_to_pick['name']}' via TPU. Submitting pick task.")
            else: 
                with state.app_lock:
                    original_feedback = state.last_nlu_result_text 
                    new_feedback = f"虽然明白了您的指令（例如“{original_feedback.split('。')[0]}”），但在画面中没有找到'{pick_object_name_from_nlu}'。"
                    state.last_nlu_status = f"未在画面中找到'{pick_object_name_from_nlu}'"
                    state.last_nlu_result_text = new_feedback
                    state.current_arm_state = config.ARM_STATE_IDLE
                print(f"TASK_PROCESS_AUDIO: Object '{pick_object_name_from_nlu}' not found in detected list. Updated NLU status and result text.")
                return # Return here as no action can be taken
            
            final_place_coords_for_arm = None # Default for 'pick' only action
            if action == "pick_and_place":
                place_coordinates_from_nlu_val = nlu_output.get("place_coordinates") # Re-get from nlu_output
                if place_coordinates_from_nlu_val and isinstance(place_coordinates_from_nlu_val, dict) and "x" in place_coordinates_from_nlu_val and "y" in place_coordinates_from_nlu_val:
                    place_x = place_coordinates_from_nlu_val["x"]
                    place_y = place_coordinates_from_nlu_val["y"]
                    place_z = config.TABLE_Z_PLANE + config.DROP_HEIGHT_OFFSET
                    final_place_coords_for_arm = {"x": place_x, "y": place_y, "z": place_z}
                else:
                    print(f"TASK_PROCESS_AUDIO: NLU (pick_and_place) did not provide valid place_coordinates. Using default.")
                    default_loc = config.PREDEFINED_PLACE_LOCATIONS["默认放置区"]
                    final_place_coords_for_arm = {"x": default_loc["x"], "y": default_loc["y"], "z": default_loc["z"]}
            
            if object_instance_to_pick: # This check is a bit redundant due to above, but safe
                task_thread = threading.Thread(
                    target=threaded_pick_and_place_task, 
                    args=(object_instance_to_pick['name'], object_instance_to_pick['box_center_x'], object_instance_to_pick['box_center_y'], final_place_coords_for_arm),
                    daemon=True)
                task_thread.start()
                print(f"TASK_PROCESS_AUDIO: Started threaded_pick_and_place_task for '{object_instance_to_pick['name']}'.")

        elif action == "inform" or action is None:
            # NLU service already set state.last_nlu_result_text and state.last_nlu_status = "指令理解成功"
            print(f"TASK_PROCESS_AUDIO: NLU action is 'inform'. No arm action.")
            with state.app_lock:
                state.current_arm_state = config.ARM_STATE_IDLE
        else: 
            with state.app_lock:
                state.last_nlu_status = f"未知NLU动作: {action}"
                state.last_nlu_result_text = f"我不理解这个动作：'{action}'。" 
                state.current_arm_state = config.ARM_STATE_IDLE
            print(f"TASK_PROCESS_AUDIO: Unknown NLU action '{action}'. Updated NLU status and result text.")
    
    except Exception as e_task_main:
        print(f"TASK_PROCESS_AUDIO_ERROR: Exception in main task logic: {e_task_main}")
        traceback.print_exc()
        with state.app_lock:
            state.last_nlu_status = "任务处理异常"
            state.last_nlu_result_text = "抱歉，处理您的指令时发生了内部错误。"
            # Reset arm state only if it's not already in a stable or ongoing critical state
            if state.current_arm_state not in [config.ARM_STATE_IDLE, config.ARM_STATE_HOMING, 
                                              config.ARM_STATE_PICKING_SUBMITTED, # These are states where another thread might be active
                                              config.ARM_STATE_PICKING_MOVE_TO_SUCK, config.ARM_STATE_PICKING_SUCKING,
                                              config.ARM_STATE_PICKING_LIFTING, config.ARM_STATE_PICKING_HOMING_AFTER,
                                              config.ARM_STATE_MOVING_TO_PLACE, config.ARM_STATE_PLACING_OBJECT
                                              ]:
                 state.current_arm_state = config.ARM_STATE_IDLE
    finally:
        # This finally block executes regardless of return statements in the try block.
        if os.path.exists(audio_filepath_original):
            try: os.remove(audio_filepath_original)
            except Exception as e_del: print(f"TASK_PROCESS_AUDIO:删除原始上传文件失败: {e_del}")
        
        # Final check of states before thread exits
        with state.app_lock:
            final_nlu_s = state.last_nlu_status
            final_nlu_r_t = state.last_nlu_result_text
            final_voice_c_t = state.last_voice_command_text
            final_arm_s = state.current_arm_state
            print(f"TASK_PROCESS_AUDIO_FINALLY: Thread ending for '{audio_filepath_original}'.")
            print(f"  Final NLU Status: '{final_nlu_s}'")
            print(f"  Final NLU Result Text: '{final_nlu_r_t}'")
            print(f"  Final Voice Command Text: '{final_voice_c_t}'")
            print(f"  Final Arm State: '{final_arm_s}'")

            # If the NLU process didn't lead to an action that sets its own arm state (like PICKING_SUBMITTED)
            # and the arm is still in an NLU-related intermediate state, reset it to IDLE.
            # This is particularly for cases like "inform" action, or NLU errors.
            nlu_action_from_output = nlu_output_for_finally_check.get("action") if nlu_output_for_finally_check else "error_or_no_nlu_output"
            
            if final_arm_s in [config.ARM_STATE_RECOGNIZING_VOICE, config.ARM_STATE_UNDERSTANDING_NLU]:
                # If still in these states, it means the process aborted or NLU was 'inform'
                print(f"  TASK_PROCESS_AUDIO_FINALLY: Arm was in intermediate state '{final_arm_s}', NLU action was '{nlu_action_from_output}'. Resetting to IDLE.")
                state.current_arm_state = config.ARM_STATE_IDLE
            elif nlu_action_from_output == "inform" and final_arm_s != config.ARM_STATE_IDLE:
                 print(f"  TASK_PROCESS_AUDIO_FINALLY: NLU action was 'inform', ensuring arm state is IDLE (was '{final_arm_s}').")
                 state.current_arm_state = config.ARM_STATE_IDLE


# --- 后台硬件与系统任务 (保持不变) ---
def stm32_reader_thread_func(ser):
    # ... (保持不变)
    print("[STM32 RX Thread] Started.")
    buffer = b""
    while not state.stop_threads_flag:
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
                                with state.app_lock:
                                    if new_status != state.stm32_current_light_status:
                                        state.stm32_current_light_status = new_status
                                        print(f"[STM32 Event] Light status changed to {new_status}")
                                        if new_status == "RED_ON": 
                                            if state.current_arm_state != config.ARM_STATE_HOMING:
                                                state.was_running_when_homing_triggered = (state.current_arm_state == config.ARM_STATE_RUNNING_TRAJECTORY)
                                                state.current_arm_state = config.ARM_STATE_HOMING
                                        elif new_status == "GREEN_ON": 
                                            idle_like_states = [config.ARM_STATE_IDLE, config.ARM_STATE_PAUSED_TRAJECTORY, config.ARM_STATE_RECOGNIZING_VOICE, config.ARM_STATE_UNDERSTANDING_NLU]
                                            if state.current_arm_state in idle_like_states or (state.current_arm_state == config.ARM_STATE_HOMING and not state.was_running_when_homing_triggered):
                                                state.current_arm_state = config.ARM_STATE_RUNNING_TRAJECTORY
                                                state.current_trajectory_index = 0
                                        elif new_status == "YELLOW_ON": 
                                            if state.current_arm_state == config.ARM_STATE_PAUSED_TRAJECTORY:
                                                state.current_arm_state = config.ARM_STATE_RUNNING_TRAJECTORY
                                        elif new_status == "ALL_OFF": 
                                            if state.current_arm_state == config.ARM_STATE_RUNNING_TRAJECTORY:
                                                state.current_arm_state = config.ARM_STATE_PAUSED_TRAJECTORY
                        except Exception as decode_err:
                            print(f"[STM32 RX Thread] Decode Error: {decode_err}, Raw: {line_bytes!r}")
            else:
                time.sleep(0.02)
        except serial.SerialException:
            if not state.stop_threads_flag: print(f"[STM32 RX Thread] SerialException.")
            break
        except Exception as e:
            if not state.stop_threads_flag: print(f"[STM32 RX Thread] Error: {e}.")
            break
    print("[STM32 RX Thread] Stopped.")

def object_detection_task():
    # ... (保持不变)
    print("[Object Detection Task] Started.")
    frame_counter = 0
    DETECTION_INTERVAL = 3 
    
    while not state.stop_threads_flag:
        if not config.DETECTION_ENABLED or state.tpu_instance is None:
            time.sleep(0.5)
            continue
        
        with state.app_lock:
            is_arm_idle_for_detection = state.current_arm_state == config.ARM_STATE_IDLE
        if not is_arm_idle_for_detection:
            time.sleep(0.2)
            continue

        original_frame = None
        with state.camera_lock:
            if state.camera and state.camera.isOpened():
                ret, frame = state.camera.read()
                if ret: original_frame = frame
        if original_frame is None:
            time.sleep(0.1)
            continue

        frame_counter += 1
        if frame_counter % DETECTION_INTERVAL != 0:
            time.sleep(0.03)
            continue

        try:
            boxes = state.tpu_instance.predict(
                original_frame, mode=config.TPU_REMOTE_MODE, model_name=config.TPU_MODEL_NAME,
                iou=config.TPU_IOU_THRES, conf=config.TPU_CONF_THRES,
                w=config.TPU_IMAGE_WIDTH, h=config.TPU_IMAGE_HEIGHT,
            )
            
            current_detected = []
            if boxes:
                actual_height, actual_width, _ = original_frame.shape
                target_h, target_w = config.TPU_IMAGE_HEIGHT, config.TPU_IMAGE_WIDTH
                pad_y = (target_h - actual_height) // 2
                pad_x = (target_w - actual_width) // 2
                
                for i, box_data in enumerate(boxes):
                    cx_padded, cy_padded, w_padded, h_padded, score, label_name = box_data
                    cx_corrected, cy_corrected = cx_padded - pad_x, cy_padded - pad_y
                    
                    if not (0 <= cx_corrected < actual_width and 0 <= cy_corrected < actual_height):
                        continue

                    current_detected.append({
                        "id": f"obj_{time.time_ns()}_{i}", "name": label_name, "confidence": float(score),
                        "box_center_x": int(cx_corrected), "box_center_y": int(cy_corrected),
                        "box_width": int(w_padded), "box_height": int(h_padded)
                    })
            
            with state.app_lock:
                state.detected_objects_list = current_detected
        except Exception as e:
            print(f"[Object Detection Task] Error during prediction: {e}")
            traceback.print_exc()
            with state.app_lock: state.detected_objects_list = []
    print("[Object Detection Task] Stopped.")


def arm_background_task():
    # ... (保持不变)
    print("[Arm BG Task] Started.")
    while not state.stop_threads_flag:
        state_to_process = None
        with state.app_lock:
            state_to_process = state.current_arm_state
        
        if state.my_arm_instance:
            if state_to_process == config.ARM_STATE_HOMING:
                print("  BG_TASK: Arm state is HOMING. Executing reset...");
                try:
                    state.my_arm_instance.move_point(config.ARM_HOME_POS_X, config.ARM_HOME_POS_Y, config.ARM_HOME_POS_Z, config.ARM_RESET_TIME)
                except Exception as e_home:
                    print(f"  BG_TASK_ERROR: Homing failed: {e_home}")
                with state.app_lock:
                    if state.was_running_when_homing_triggered:
                        state.current_arm_state = config.ARM_STATE_PAUSED_TRAJECTORY
                    else:
                        state.current_arm_state = config.ARM_STATE_IDLE
                        state.current_trajectory_index = 0
                    state.was_running_when_homing_triggered = False
            
            elif state_to_process == config.ARM_STATE_RUNNING_TRAJECTORY:
                can_run_step, point_data, temp_traj_index = False, None, -1
                with state.app_lock:
                    if state.current_trajectory_index < config.TOTAL_TRAJECTORY_POINTS:
                        can_run_step = True
                        point_data = config.PREDEFINED_TRAJECTORY[state.current_trajectory_index]
                        temp_traj_index = state.current_trajectory_index
                    else:
                        print("  BG_TASK: Trajectory fully completed.");
                        state.current_arm_state = config.ARM_STATE_IDLE
                        state.current_trajectory_index = 0
                
                if can_run_step and point_data:
                    target_x, target_y, target_z, move_duration = point_data
                    print(f"  BG_TASK_TRAJ: Moving to point {temp_traj_index+1}/{config.TOTAL_TRAJECTORY_POINTS}")
                    try:
                        state.my_arm_instance.move_point(target_x, target_y, target_z, move_duration)
                        with state.app_lock:
                            if state.current_arm_state == config.ARM_STATE_RUNNING_TRAJECTORY and state.current_trajectory_index == temp_traj_index:
                                state.current_trajectory_index += 1
                                if state.current_trajectory_index >= config.TOTAL_TRAJECTORY_POINTS:
                                    state.current_arm_state = config.ARM_STATE_IDLE
                                    state.current_trajectory_index = 0
                    except Exception as e_traj:
                        print(f"  BG_TASK_ERROR: Moving to trajectory point {temp_traj_index + 1} failed: {e_traj}")
                        with state.app_lock: state.current_arm_state = config.ARM_STATE_PAUSED_TRAJECTORY
        
        time.sleep(0.1)
    print("[Arm BG Task] Stopped.")

def generate_frames():
    # ... (保持不变)
    while not state.stop_threads_flag:
        frame_to_yield = None
        try:
            with state.camera_lock:
                if state.camera is None or not state.camera.isOpened():
                    time.sleep(0.5); continue
                success, frame_bgr = state.camera.read()
                if not success or frame_bgr is None:
                    time.sleep(0.05); continue
            processed_frame = frame_bgr.copy()
            with state.app_lock: current_boxes = list(state.detected_objects_list)
            if state.pil_font and state.pil_font != "FONT_LOAD_FAILED":
                try:
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    draw = ImageDraw.Draw(pil_image)
                    for obj in current_boxes:
                        x_center, y_center, w, h = obj['box_center_x'], obj['box_center_y'], obj['box_width'], obj['box_height']
                        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
                        text_y = y1 - config.CHINESE_FONT_SIZE - 2 if y1 > config.CHINESE_FONT_SIZE else y1 + 2
                        draw.text((x1, text_y), f"{obj['name']}: {obj['confidence']:.2f}", font=state.pil_font, fill=(0, 255, 0))
                    processed_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as e_pil_draw:
                    print(f"ERROR during Pillow drawing: {e_pil_draw}")
            else: 
                for obj in current_boxes:
                    x_center, y_center, w, h = obj['box_center_x'], obj['box_center_y'], obj['box_width'], obj['box_height']
                    x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                    cv2.rectangle(processed_frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"{obj['name']}: {obj['confidence']:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret: frame_to_yield = buffer.tobytes()
        except Exception as e_outer:
            print(f"CRITICAL ERROR in generate_frames loop: {e_outer}")
            traceback.print_exc(); time.sleep(0.1)
        if frame_to_yield:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')
        time.sleep(1/30) 