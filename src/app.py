from flask import Flask, render_template, request, jsonify, Response
import time
import serial
import threading
import cv2

# 假设你的模块在同一目录下
from arm_module import Arm # 确保这个模块存在且Arm类已定义

# --- 全局配置 ---
STM32_SERIAL_PORT = "/dev/ttyUSB0"
STM32_SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 0.1

ARM_TRANSLATE_X_STEP = 20
ARM_TRANSLATE_Y_STEP = 20
ARM_TRANSLATE_Z_STEP = 15
ARM_MOVE_TIME = 400 # 用于 move_axis_point
ARM_RESET_TIME = 2000
ARM_HOME_POS_X = 0
ARM_HOME_POS_Y = -115
ARM_HOME_POS_Z = 240

PUMP_ROTATE_SMALL_STEP = 15 # 旋转头小幅调整的步长 (度)

ARM_STATE_IDLE = "IDLE"
ARM_STATE_RUNNING_TRAJECTORY = "RUNNING_TRAJECTORY"
ARM_STATE_PAUSED_TRAJECTORY = "PAUSED_TRAJECTORY"
ARM_STATE_HOMING = "HOMING"
ARM_STATE_MANUAL_MOVE = "MANUAL_MOVE" # 新增手动移动中的状态

# --- 全局变量 ---
app_lock = threading.Lock() # 保护共享变量

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
was_running_when_homing_triggered = False # 用于红灯复位后的状态判断

# 负压泵相关状态
current_pump_status = "OFF" # "ON" 或 "OFF"
current_pump_angle = 0      # 角度值

stop_threads_flag = False
stm32_serial_conn = None
my_arm_instance = None # Arm 类的实例
stm32_reader_thread_obj = None
arm_bg_thread_obj = None # 持有后台任务线程对象

camera = None
camera_lock = threading.Lock()
CAMERA_INDEX = 0

# --- 初始化 Flask 应用 ---
app = Flask(__name__)

# --- STM32 串口读取线程 (与之前版本类似) ---
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
                                        # 根据灯光状态更新机械臂状态 (逻辑与之前修复bug版类似)
                                        if new_status == "RED_ON":
                                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY: was_running_when_homing_triggered = True
                                            else: was_running_when_homing_triggered = False
                                            current_arm_state = ARM_STATE_HOMING
                                        elif new_status == "GREEN_ON":
                                            if current_arm_state != ARM_STATE_HOMING: # 避免在Homing时被Green打断
                                                if current_arm_state in [ARM_STATE_IDLE, ARM_STATE_PAUSED_TRAJECTORY, ARM_STATE_RUNNING_TRAJECTORY]:
                                                    current_arm_state = ARM_STATE_RUNNING_TRAJECTORY; current_trajectory_index = 0; was_running_when_homing_triggered = False
                                        elif new_status == "YELLOW_ON":
                                            if current_arm_state == ARM_STATE_PAUSED_TRAJECTORY:
                                                if current_trajectory_index < TOTAL_TRAJECTORY_POINTS: current_arm_state = ARM_STATE_RUNNING_TRAJECTORY; was_running_when_homing_triggered = False
                                                else: current_arm_state = ARM_STATE_IDLE; current_trajectory_index = 0
                                        elif new_status == "ALL_OFF":
                                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY: current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
                        except Exception as decode_err: print(f"[STM32 RX Thread WebUI] Decode Error: {decode_err}")
            else: time.sleep(0.02)
        except serial.SerialException:
            if not stop_threads_flag: print(f"[STM32 RX Thread WebUI] SerialException.")
            break
        except Exception as e:
            if not stop_threads_flag: print(f"[STM32 RX Thread WebUI] Error: {e}.")
            break
    print("[STM32 RX Thread WebUI] Stopped.")

# --- 发送指令到 STM32 (与之前版本类似) ---
def send_command_to_stm32(ser, command_str):
    try:
        full_command = command_str + "\n"
        print(f"PY TX (WebUI) -> STM32: {command_str}")
        if ser and ser.is_open: ser.write(full_command.encode('ascii'))
        else: print("  错误 (WebUI): STM32串口未连接。"); return False
        return True
    except Exception as e: print(f"发送指令到STM32错误 (WebUI): {e}"); return False

# --- 机械臂后台任务 (状态机和轨迹执行) ---
def arm_background_task():
    global current_arm_state, current_trajectory_index, was_running_when_homing_triggered, current_pump_angle, current_pump_status
    print("[Arm BG Task WebUI] Started.")
    while not stop_threads_flag:
        # 在这个版本中，耗时操作会直接在这个线程中执行，从而阻塞这个后台任务的循环
        # 也可能间接影响Flask服务器的其他部分，特别是如果GIL被长时间占用
        
        state_to_process = None
        with app_lock: # 读取当前状态
            state_to_process = current_arm_state

        if my_arm_instance:
            if state_to_process == ARM_STATE_HOMING:
                print("  BG_TASK: Arm state is HOMING. Executing reset...")
                try:
                    my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, ARM_RESET_TIME)
                    print("  BG_TASK: Arm homing complete.")
                except Exception as e_home:
                    print(f"  BG_TASK_ERROR: Homing failed: {e_home}")
                with app_lock:
                    if was_running_when_homing_triggered and stm32_current_light_status == "RED_ON":
                        current_arm_state = ARM_STATE_PAUSED_TRAJECTORY
                        print(f"  BG_TASK: Homing (RED) done. PAUSED.")
                    else:
                        current_arm_state = ARM_STATE_IDLE
                        current_trajectory_index = 0
                        print("  BG_TASK: Homing done. IDLE.")
                    was_running_when_homing_triggered = False
            
            elif state_to_process == ARM_STATE_RUNNING_TRAJECTORY:
                # 检查是否还有轨迹点需要执行
                can_run_step = False
                point_data = None
                temp_traj_index = -1

                with app_lock:
                    if current_trajectory_index < TOTAL_TRAJECTORY_POINTS:
                        can_run_step = True
                        point_data = PREDEFINED_TRAJECTORY[current_trajectory_index]
                        temp_traj_index = current_trajectory_index # 记录当前要执行的索引
                    else: # 轨迹已完成
                        print("  BG_TASK: Trajectory was RUNNING but all points done. Setting IDLE.")
                        current_arm_state = ARM_STATE_IDLE
                        current_trajectory_index = 0

                if can_run_step and point_data:
                    target_x, target_y, target_z, move_duration = point_data
                    print(f"  BG_TASK_TRAJ: Moving to point {temp_traj_index + 1}/{TOTAL_TRAJECTORY_POINTS}")
                    try:
                        my_arm_instance.move_point(target_x, target_y, target_z, move_duration)
                        # 移动成功后，在锁内更新状态
                        with app_lock:
                            # 再次检查状态，防止在move_point执行期间被外部改变（如红灯）
                            if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY and current_trajectory_index == temp_traj_index:
                                current_trajectory_index += 1
                                if current_trajectory_index >= TOTAL_TRAJECTORY_POINTS:
                                    print("  BG_TASK_TRAJ: Trajectory fully completed.")
                                    current_arm_state = ARM_STATE_IDLE
                                    current_trajectory_index = 0
                            elif current_arm_state != ARM_STATE_RUNNING_TRAJECTORY : #被中断
                                print(f"  BG_TASK_TRAJ: Step {temp_traj_index + 1} done, but state changed to {current_arm_state}. Not advancing index here.")
                                # 让主状态机或STM32中断逻辑处理后续

                    except Exception as e_traj:
                        print(f"  BG_TASK_ERROR: Moving to trajectory point {temp_traj_index + 1} failed: {e_traj}")
                        with app_lock:
                            current_arm_state = ARM_STATE_PAUSED_TRAJECTORY # 出错则暂停
                            # current_trajectory_index 保持在出错的这一点
            
            # 注意：ARM_STATE_MANUAL_MOVE 的处理主要在 /control 路由中，完成后直接设回 IDLE
            # BG task 主要处理 HOMING 和 RUNNING_TRAJECTORY 这种持续性的状态
            
        time.sleep(0.1) # 后台任务轮询间隔
    print("[Arm BG Task WebUI] Stopped.")


# --- 摄像头帧生成器 (与之前版本类似) ---
def generate_frames():
    global camera
    while not stop_threads_flag:
        frame_to_yield = None
        with camera_lock:
            if camera is None or not camera.isOpened():
                # print("摄像头尝试重连...") # 减少打印
                try:
                    camera = cv2.VideoCapture(CAMERA_INDEX)
                    if not camera.isOpened(): time.sleep(1); continue
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                except Exception: camera = None; time.sleep(1); continue
            success, frame = camera.read()
            if success:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret: frame_to_yield = buffer.tobytes()
        
        if frame_to_yield:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')
        time.sleep(1/25) # 控制帧率，例如25fps

# --- Flask 路由 ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control_arm_route(): # 重命名以区分内部函数
    global current_arm_state, current_trajectory_index, was_running_when_homing_triggered, current_pump_status, current_pump_angle
    
    data = request.get_json()
    command = data.get('command')
    payload = data.get('payload', {}) # 获取payload，默认为空字典
    message = f"Received command: {command}"
    success = True
    status_code = 200

    # STM32 灯光控制指令 (快速，直接处理)
    if command.startswith('set_'):
        with app_lock: # 确保STM32指令发送时状态不会混淆
            if command == 'set_green_on': success = send_command_to_stm32(stm32_serial_conn, "SET_GREEN_ON")
            elif command == 'set_red_on': success = send_command_to_stm32(stm32_serial_conn, "SET_RED_ON")
            elif command == 'set_yellow_on': success = send_command_to_stm32(stm32_serial_conn, "SET_YELLOW_ON")
            elif command == 'set_lights_off': success = send_command_to_stm32(stm32_serial_conn, "SET_LIGHTS_OFF")
            else: success = False; message = "Unknown set command"
        
        if not success: status_code = 500; message = f"Failed to send {command}"
        return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code

    # 机械臂相关指令
    if not my_arm_instance:
        return jsonify({'status': 'error', 'message': '机械臂未初始化!'}), 500

    # --- 在这里直接执行机械臂的阻塞操作 ---
    # --- 这会导致 Web UI 在操作完成前无响应 ---
    previous_state_for_manual = None
    try:
        with app_lock: # 获取锁以检查和设置状态
            # 检查是否可以执行手动操作或特定任务
            if command.startswith('manual_') and current_arm_state != ARM_STATE_IDLE:
                if command != 'manual_reset': # 复位可以打断一些状态
                     return jsonify({'status': 'error', 'message': f'机械臂当前状态为 {current_arm_state}, 不能执行手动操作。'}), 400
            
            if command.startswith('manual_') and command != 'manual_reset':
                previous_state_for_manual = current_arm_state # 应该是IDLE
                current_arm_state = ARM_STATE_MANUAL_MOVE # 标记为手动移动中
            elif command == 'manual_reset':
                current_arm_state = ARM_STATE_HOMING # 转交给后台任务处理复位
                was_running_when_homing_triggered = False
                current_trajectory_index = 0
                message = "Arm Reset command submitted to background task."
                return jsonify({'status': 'success', 'message': message}), 200


        # 手动控制指令 (直接执行)
        if command == 'manual_x_plus': my_arm_instance.move_axis_point(0, ARM_TRANSLATE_X_STEP, ARM_MOVE_TIME); message = "Arm X+"
        elif command == 'manual_x_minus': my_arm_instance.move_axis_point(0, -ARM_TRANSLATE_X_STEP, ARM_MOVE_TIME); message = "Arm X-"
        elif command == 'manual_y_plus': my_arm_instance.move_axis_point(1, ARM_TRANSLATE_Y_STEP, ARM_MOVE_TIME); message = "Arm Y+"
        elif command == 'manual_y_minus': my_arm_instance.move_axis_point(1, -ARM_TRANSLATE_Y_STEP, ARM_MOVE_TIME); message = "Arm Y-"
        elif command == 'manual_z_plus': my_arm_instance.move_axis_point(2, ARM_TRANSLATE_Z_STEP, ARM_MOVE_TIME); message = "Arm Z+"
        elif command == 'manual_z_minus': my_arm_instance.move_axis_point(2, -ARM_TRANSLATE_Z_STEP, ARM_MOVE_TIME); message = "Arm Z-"
        
        # 负压泵控制 (假设这些是快速的或arm_module内部非阻塞)
        elif command == 'pump_on':
            my_arm_instance.suck_up()
            print("COMMAND: Pump ON")
            with app_lock: current_pump_status = "ON"; message = "Pump ON"
        elif command == 'pump_off':
            my_arm_instance.suck_release()
            print("COMMAND: Pump OFF")
            with app_lock: current_pump_status = "OFF"; message = "Pump OFF"
        elif command == 'pump_angle_set':
            angle = payload.get('angle', current_pump_angle) # 从payload获取角度
            print(angle)
            my_arm_instance.rotate_angle(angle)
            print(f"COMMAND: Set Pump Angle to {angle}")
            with app_lock: current_pump_angle = angle; message = f"Pump Angle set to {angle}°"
        elif command == 'pump_rotate_left_small':
            new_angle = 0
            with app_lock: new_angle = max(-90, current_pump_angle - PUMP_ROTATE_SMALL_STEP)
            my_arm_instance.rotate_angle(new_angle)
            print(f"COMMAND: Pump Rotate Left Small to {new_angle}")
            with app_lock: current_pump_angle = new_angle; message = f"Pump Rotated Left to {new_angle}°"
        elif command == 'pump_rotate_right_small':
            new_angle = 0
            with app_lock: new_angle = min(90, current_pump_angle + PUMP_ROTATE_SMALL_STEP)
            my_arm_instance.rotate_angle(new_angle)
            print(f"COMMAND: Pump Rotate Right Small to {new_angle}")
            with app_lock: current_pump_angle = new_angle; message = f"Pump Rotated Right to {new_angle}°"

        # 软控制任务状态 (直接修改状态，由arm_background_task响应)
        elif command == 'task_pause':
            with app_lock:
                if current_arm_state == ARM_STATE_RUNNING_TRAJECTORY:
                    current_arm_state = ARM_STATE_PAUSED_TRAJECTORY; message = "Task Paused"
                else: success = False; message = "Cannot pause, not running trajectory"
        elif command == 'task_stop': # 停止通常意味着复位并到IDLE
            with app_lock:
                current_arm_state = ARM_STATE_HOMING; was_running_when_homing_triggered = False; current_trajectory_index = 0; message = "Task Stop (Homing initiated)"
        elif command == 'task_continue':
            with app_lock:
                if current_arm_state == ARM_STATE_PAUSED_TRAJECTORY:
                    current_arm_state = ARM_STATE_RUNNING_TRAJECTORY; message = "Task Continued"
                else: success = False; message = "Cannot continue, not paused"
        
        # 模拟语音指令处理入口（这里只打印，实际NLU和执行在别处）
        elif command == 'process_voice_text':
            text = payload.get('text', '')
            print(f"Simulating NLU for text: {text}")
            message = f"Voice text '{text}' submitted for NLU."
            # 真实的NLU和后续的指令转换会更复杂，可能也需要改变current_arm_state
        elif command == 'start_hardware_voice_recognition':
            print("Simulating hardware voice recognition start...")
            message = "Hardware voice recognition process initiated (simulated)."


        else: # 未知机械臂相关指令
            if not command.startswith('set_'): # 排除已经处理的 set_ 指令
                message = f"Unknown arm command: {command}"
                success = False
                status_code = 400
    
    except Exception as e:
        message = f"Error executing command '{command}': {e}"
        print(message)
        success = False
        status_code = 500
    finally:
        if command.startswith('manual_') and command != 'manual_reset' : # 手动移动完成后，无论成功失败都应恢复状态
            with app_lock:
                if current_arm_state == ARM_STATE_MANUAL_MOVE: # 确保是我们设置的
                    current_arm_state = ARM_STATE_IDLE # 直接回到IDLE

    return jsonify({'status': 'success' if success else 'error', 'message': message}), status_code


@app.route('/status')
def get_status():
    with app_lock:
        # 假设机械臂对象有获取当前位置和角度的方法
        arm_coords_str = "[N/A]"
        #arm_joints_str = "[N/A]"
        pos = my_arm_instance.get_axis_point('3') # 示例方法
        if pos: arm_coords_str = f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}, ...]"
        # if my_arm_instance and hasattr(my_arm_instance, 'get_joint_angles'):
        #     joints = my_arm_instance.get_joint_angles() # 示例方法
        #     if joints: arm_joints_str = ", ".join([f"{j:.1f}" for j in joints])


        status_data = {
            'arm_state': current_arm_state,
            'stm32_light_status': stm32_current_light_status,
            'current_trajectory_index': current_trajectory_index,
            'total_trajectory_points': TOTAL_TRAJECTORY_POINTS,
            'pump_status': current_pump_status,
            'pump_angle': current_pump_angle,
            'arm_coords': arm_coords_str, # 示例
            #'arm_joints': arm_joints_str, # 示例
            # 'recognized_objects': [], # 实际应用中会填充此列表
            # 'target_object': 'None',
            # 'target_location': 'None',
            'timestamp': time.time()
        }
    return jsonify(status_data)

def init_hardware_and_camera():
    global my_arm_instance, stm32_serial_conn, stm32_reader_thread_obj, arm_bg_thread_obj
    global current_arm_state, current_trajectory_index, was_running_when_homing_triggered
    global camera, current_pump_status, current_pump_angle

    print("Initializing Arm for WebUI...")
    try:
        my_arm_instance = Arm() # 假设Arm()内部处理连接
        print("Resetting Arm to home position on startup (WebUI)...")
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
        print("STM32 serial connected (WebUI)."); time.sleep(0.5)
        stm32_reader_thread_obj = threading.Thread(target=stm32_reader_thread_func, args=(stm32_serial_conn,), daemon=True)
        stm32_reader_thread_obj.start()
        print("STM32 reader thread started for WebUI.")
    except serial.SerialException as e:
        print(f"!!! Failed to connect to STM32 (WebUI): {e} !!!"); stm32_serial_conn = None

    print(f"Initializing Camera (index: {CAMERA_INDEX}) for WebUI...")
    with camera_lock:
        try:
            camera = cv2.VideoCapture(CAMERA_INDEX)
            if not camera.isOpened(): print(f"!!! Error: Could not open camera {CAMERA_INDEX} !!!"); camera = None
            else:
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640); camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"Camera {CAMERA_INDEX} initialized successfully.")
        except Exception as e: print(f"!!! Error initializing camera: {e} !!!"); camera = None
    
    # 启动机械臂后台状态机线程
    arm_bg_thread_obj = threading.Thread(target=arm_background_task, daemon=True)
    arm_bg_thread_obj.start()


if __name__ == '__main__':
    init_hardware_and_camera()

    print("Starting Flask web server on http://0.0.0.0:5000 ...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) 
    
    print("Flask server shutting down...")
    stop_threads_flag = True 

    with camera_lock:
        if camera and camera.isOpened(): camera.release(); print("Camera released.")
            
    if stm32_reader_thread_obj and stm32_reader_thread_obj.is_alive():
        print("Waiting for STM32 reader thread..."); stm32_reader_thread_obj.join(timeout=2)
    if arm_bg_thread_obj and arm_bg_thread_obj.is_alive():
        print("Waiting for Arm background task..."); arm_bg_thread_obj.join(timeout=2)
    
    if stm32_serial_conn and stm32_serial_conn.is_open:
        try: send_command_to_stm32(stm32_serial_conn, "SET_LIGHTS_OFF"); stm32_serial_conn.close()
        except: pass
    
    if my_arm_instance: # 尝试最后复位
        print("Final attempt to home arm...")
        try:
            # 注意：如果move_point是阻塞的，这里可能会延迟关闭
            # 理想情况下，机械臂应该有一个非阻塞的停止或复位命令
            # my_arm_instance.move_point(ARM_HOME_POS_X, ARM_HOME_POS_Y, ARM_HOME_POS_Z, ARM_RESET_TIME // 2)
            pass # 实际的复位可能需要更复杂的处理以避免阻塞关闭
        except: pass
        # if hasattr(my_arm_instance, 'disconnect'): my_arm_instance.disconnect()

    print("WebUI Application finished.")