import os
import time
import subprocess
from Sk import *


class Arm():
    # 根据配置文件初始化Arm机械臂
    def __init__(self):

        # 获取当前脚本的目录路径
        # PWD = os.getcwd()
        # print("PWD:", PWD)
        # my_path = os.getenv('ARM_PATH')
        # if PWD != my_path:
        #     subprocess.run(['cp', f"{my_path}/param.ini", f"{PWD}/param.ini"])

#        self.suck = sucker(492, 491)
#        self.pwm1 = pwm(2)
#        self.pwmservo = pwmServo(self.pwm1)
#        self.serial1 = uart("/dev/ttyAMA5")
#        self.serial2 = uart("/dev/ttyAMA2", 9600)
#        self.servo = serialServo(self.serial1)
#        self.horn = voice(self.serial2)
#        self.arm = firstArm(self.servo, self.pwmservo, self.suck, self.horn)

        self.serial = uart("/dev/ttyUSB0")
        self.serial1 = uart("/dev/ttyAMA2", 9600)
        self.horn = voice(self.serial1)
        self.arm = firstArm(self.serial, self.horn)

        # if PWD != my_path:
        #     subprocess.run(['rm', f"{PWD}/param.ini"])

    # 判断x,y,z值是否合理
    def is_value(self, x, y, z):
        return self.arm.is_value(x, y, z)

    # 积木块1："机械臂复位"
    # 参数：times（回到初始位置的时间，默认是2000）
    def reset_position(self, times=2000):
        self.arm.init(times)

    # 积木块2："获取舵机【】的脉冲值"
    # 参数：servo_id(舵机id值: 1 , 2, 3)
    # 返回值：指定舵机的脉冲值
    def servo_get_position(self, servo_id):
        return self.arm.get_servo_position(servo_id)

    # 积木块3：“获取末端执行器坐标【】的值”
    # 参数：c_axis(坐标轴类型x = 0, y = 1, z = 2)
    # 返回值：末端执行器的坐标值
    def get_axis_point(self, c_axis):
        poi = self.arm.get_point()
        if c_axis == 0:
            return int(poi.x)
        elif c_axis == 1:
            return int(poi.y)
        elif c_axis == 2:
            return int(poi.z)
        elif c_axis == "3":
            return [int(poi.x), int(poi.y), int(poi.z)]
        else:
            print("输入正确的axis值")

    # 积木块4：“移动到空间位置”
    # 参数：x、y、z（目标坐标），times（移动时间,默认为0）
    def move_point(self, x, y, z, times=0):
        # 空间限位
        if self.arm.is_value(x, y, z) == False or z < 58:
            print("不合法参数，机械臂无法到达")
        elif times != 0:
            self.arm.move_point(x, y, z, times)
        else:
            self.arm.move_point(x, y, z)

    # 积木块5：“沿【】坐标轴移动【】距离，运动时间【0】ms”
    # 参数：c_axis(坐标轴类型x = 0, y = 1, z = 2), coord_move(移动距离)，times(移动时间)
    def move_axis_point(self, c_axis, coord_move, times):
        poi = self.arm.get_point()
        print(f"poi.x:{poi.x}, poi.y:{poi.y}, poi.z:{poi.z}")
        if c_axis == 0:
            if self.arm.is_value(int(poi.x + coord_move), int(poi.y), int(poi.z)) == False:
                print("不合法参数，机械臂无法到达")
            else:
                self.arm.move_x(coord_move, times)
        elif c_axis == 1:
            if self.arm.is_value(int(poi.x), int(poi.y + coord_move), int(poi.z)) == False:
                print("不合法参数，机械臂无法到达")
            else:
                self.arm.move_y(coord_move, times)
        elif c_axis == 2:
            if self.arm.is_value(int(poi.x), int(poi.y), int(poi.z + coord_move)) == False or int(poi.z + coord_move) < 58:
                print("不合法参数，机械臂无法到达")
            else:
                self.arm.move_z(coord_move, times)
        else:
            print("输入正确的axis值")

    # 积木块6：“移动到图像位置”
    # 参数：pix_x、pix_y (像素值坐标)，pix_z(预设高度), img_w(图像宽), img_h(图像高)
    # 返回值：times（运动所用时间）, position（实际坐标值）(可不接收)
    def move_pixel_wh(self, pix_x, pix_y, pix_z, img_w, img_h):
        if pix_z < 58 or pix_x > img_w or pix_y > img_w or pix_x < 0 or pix_y < 0:
            print("不合法参数，机械臂无法到达")
        else:
            times, positions = self.arm.move_pixels(int((640 / img_w) * pix_x), int((480 / img_h) * pix_y), pix_z)
            print(f"times:{times}, positions.x:{positions.x}, positions.y:{positions.y}, positions.z:{positions.z}")
            # 空间限位
            if self.arm.is_value(int(positions.x), int(positions.y), int(positions.z)) == False:
                print("不合法参数，机械臂无法到达")
            else:
                return times, positions

    # 积木块7：设置舵机【】的脉冲为【】运行【】ms，限位【】
    # 参数：servo_id(舵机id: 1, 2, 3), set_position(目标脉冲值)，times(运行时间)，flag(是否限位)
    def servo_set_position(self, servo_id, set_position, times, flag):
        if flag:
            if servo_id == 1:
                # 获取舵机的脉冲值
                pos2 = self.arm.get_servo_position(2)
                pos3 = self.arm.get_servo_position(3)
                # print("pos = ", pos1, " pos2 = ", set_position, " pos3 = ", pos3)
                poi = self.arm.forward_kinematics(set_position, pos2, pos3)

            elif servo_id == 2:
                # 获取舵机的脉冲值
                pos1 = self.arm.get_servo_position(1)
                pos3 = self.arm.get_servo_position(3)
                # print("pos = ", pos1, " pos2 = ", set_position, " pos3 = ", pos3)
                poi = self.arm.forward_kinematics(pos1, set_position, pos3)

            elif servo_id == 3:
                # 获取舵机的脉冲值
                pos1 = self.arm.get_servo_position(1)
                pos2 = self.arm.get_servo_position(2)
                # print("pos = ", pos1, " pos2 = ", pos2, " pos3 = ", set_position)
                poi = self.arm.forward_kinematics(pos1, pos2, set_position)

            print("positionX = ", poi.x, "positionY = ", poi.y, "positionZ = ", poi.z)

            # r = pow(pow(poi.x, 2) + pow(poi.y, 2), 0.5)
            # print("R :", r)
            if self.arm.is_value(int(poi.x), int(poi.y), int(poi.z)) == False or poi.z < 58:
                print("不合法参数，机械臂无法到达")
            else:
                self.arm.servo_move(servo_id, set_position, times)
#                time.sleep((times+200)/1000)
        else:
            self.arm.servo_move(servo_id, set_position, times)
#            time.sleep((times+200)/1000)

    # 积木块8：气泵【】
    # 参数：mode(模式：吸气 = 0, 放气 = 1, 初始状态 = 2)
    def suck_mode(self, mode):
        if mode == 0:
            self.arm.suck()
        elif mode == 1:
            self.arm.release()
        elif mode == 2:
            self.arm.set_sucker_init()
        else:
            print("错误气泵指令")

    # 积木块9：气泵旋转角度【】
    # 参数：angle(目标角度：0~180)
    def suck_rotate_angle(self, angle):
        if angle in range(181):
            self.arm.rotate_air_pump_angle(angle)
        else:
            print("角度范围不正确, 应为0~180")

    def rotate_angle(self, angle):
        self.arm.rotate_air_pump_angle(angle)


    # 积木块10：夹爪角度设置
    # 参数：switch(打开或关闭), angle(目标角度), times(运动时间)
    def gripper_angle(self, switch, angle, times):
        self.arm.clip(angle, times)


    # 获取图片
    def get_img(self):
        from cap import VideoCapture
        try:
            cap = VideoCapture()
        except Exception as e:
            print(f"初始化摄像头异常, error: {e}")
            exit(0)
        image = cap.read()
        cap.release()
        if image is not None:
            return image

    # 功能：语音播报命令
    # 参数： data为命令内容，data_size为命令长度
    def horn_send(self, data):
        set_data = [0xAA, 0x55, data, 0x55, 0xAA]
        self.arm.send_cmd(set_data, 5)

    # 功能：接收语音命令
    # 参数：data_size为接收数据长度，timeout为超时时间）
    # 返回值：data为接收到的命令内容
    def horn_recv(self, timeout):
        data_size = 3
        return self.arm.recv_cmd(data_size, timeout)

    # 功能：夹爪夹紧(74夹_15松, 900)
    def gripper_clip(self):
        self.arm.clip(74, 900)

    # 功能：夹爪松开
    def gripper_loosen(self):
        self.arm.clip(15, 900)

    # 功能：设置气泵为初始状态(在垃圾分类案例中，气泵放气有时候不能将卡片释放，需要先将气泵吸气先置零)
    def suck_pin_reset(self):
        self.arm.set_sucker_init()

    # 功能：气泵吸气
    def suck_up(self):
        self.arm.suck()

    # 功能：气泵放气
    def suck_release(self):
        self.arm.release()

    # 功能：控制机械臂末端运动到指定像素坐标,并将该像素坐标转换为现实坐标（机械臂面向正前方）
    # 参数：pixelX、pixelY是识别卡片中心点的像素值，H是预设高度。
    # 返回值：time（运动所用时间）, position（实际坐标值）
    def move_pixel(self, pixelX, pixelY, H):
        return self.arm.move_pixels(pixelX, pixelY, H)

    def calculate_pixel(self, pixelX, pixelY, H):
        return self.arm.calculate_pixels(pixelX, pixelY, H)

    # 功能：控制机械臂末端运动到指定像素坐标,并将该像素坐标转换为现实坐标(机械臂面向左侧)
    # 参数：pixelX、pixelY是识别卡片中心点的像素值，H是预设高度。
    # 返回值：time（运动所用时间）, position（实际坐标值）
    def move_left_pixel(self, pixelX, pixelY, H):
        return self.arm.left_move_pixels(pixelX, pixelY, H)

    # 功能：控制机械臂末端运动到指定像素坐标,并将该像素坐标转换为现实坐标(机械臂面向右侧)
    # 参数：pixelX、pixelY是识别卡片中心点的像素值，H是预设高度。
    # 返回值：time（运动所用时间）, position（实际坐标值）
    def move_right_pixel(self, pixelX, pixelY, H):
        return self.arm.right_move_pixels(pixelX, pixelY, H)

    # 功能：获取机械臂末端坐标
    # 返回值：point为机械臂末端坐标，如（point.x, point.y, point.z）
    def get_point(self):
        return self.arm.get_point()

    # 功能：获取机械臂三个舵机的脉冲值
    # 返回值：servo_positions为机械臂三个舵机的脉冲值，顺序为底部、中部、顶部
    def get_position(self):
        pos1 = self.servo_get_position(1)
        pos2 = self.servo_get_position(2)
        pos3 = self.servo_get_position(3)

        return pos1, pos2, pos3

class Horn():
    # 根据配置文件初始化语音模块
    def __init__(self):
        self.suck = sucker(492, 491)
        self.pwm1 = pwm(2)
        self.pwmservo = pwmServo(self.pwm1)
        self.serial1 = uart("/dev/ttyAMA5")
        self.serial2 = uart("/dev/ttyAMA2", 9600)
        self.servo = serialServo(self.serial1)
        self.horn = voice(self.serial2)
        self.arm = firstArm(self.servo, self.pwmservo, self.suck, self.horn)

    # 功能：语音播报命令
    # 参数： data为命令内容，data_size为命令长度
    def send(self, data):
        set_data = [0xAA, 0x55, data, 0x55, 0xAA]
        self.arm.send_cmd(set_data, 5)

    # 功能：接收语音命令
    # 参数：data_size为接收数据长度，timeout为超时时间）
    # 返回值：data为接收到的命令内容
    def rece(self, timeout):
        data_size = 3
        return self.arm.recv_cmd(data_size, timeout)


class Gripper():
    # 根据配置文件初始化夹爪
    def __init__(self):
        self.suck = sucker(492, 491)
        self.pwm1 = pwm(2)
        self.pwmservo = pwmServo(self.pwm1)
        self.serial1 = uart("/dev/ttyAMA5")
        self.serial2 = uart("/dev/ttyAMA2", 9600)
        self.servo = serialServo(self.serial1)
        self.horn = voice(self.serial2)
        self.arm = firstArm(self.servo, self.pwmservo, self.suck, self.horn)

    def angle(self, angle, time):
        self.arm.clip(angle, time)

    # 功能：夹爪夹紧
    def clip(self):
        self.arm.clip(74, 900)

    # 功能：夹爪松开
    def loosen(self):
        self.arm.clip(15, 900)


class Suck():
    # 根据配置文件初始化气泵
    def __init__(self):
        self.suck = sucker(492, 491)
        self.pwm1 = pwm(2)
        self.pwmservo = pwmServo(self.pwm1)
        self.serial1 = uart("/dev/ttyAMA5")
        self.serial2 = uart("/dev/ttyAMA2", 9600)
        self.servo = serialServo(self.serial1)
        self.horn = voice(self.serial2)
        self.arm = firstArm(self.servo, self.pwmservo, self.suck, self.horn)

    # 功能：气泵的吸气、放气引脚都置零
    def pin_reset(self):
        self.arm.set_sucker_init()

    # 功能：气泵吸气
    def up(self):
        self.arm.suck()

    # 功能：气泵放气
    def release(self):
        self.arm.release()

    # 功能：控制气泵旋转角度
    # 参数：angle为目标角度
    def rotate_angle(self, angle):
        self.arm.rotate_air_pump_angle(angle)


class Servo():
    # 根据配置文件初始化舵机
    def __init__(self):
        self.suck = sucker(492, 491)
        self.pwm1 = pwm(2)
        self.pwmservo = pwmServo(self.pwm1)
        self.serial1 = uart("/dev/ttyAMA5")
        self.serial2 = uart("/dev/ttyAMA2", 9600)
        self.servo = serialServo(self.serial1)
        self.horn = voice(self.serial2)
        self.arm = firstArm(self.servo, self.pwmservo, self.suck, self.horn)

    # 功能：控制指定id的舵机旋转
    # 参数：servo_id为舵机id，set_position为目标脉冲值，time为运动时间
    def set_position(self, servo_id, set_position, time):
        self.arm.servo_move(servo_id, set_position, time)

    # 功能：获取指定id的舵机的脉冲值
    # 参数：servo_id为舵机id
    # 返回值：position为指定舵机的脉冲值
    def get_position(self, servo_id):
        return self.arm.get_servo_position(servo_id)

class desktopArm(Arm):
    def __init__(self):
        super().__init__()

    def is_in_range(self, x, y, z):
        if (x <= -180 or x >= 180) or (y >= -110) or (z <= 135):
            return False
        return True

    def move_point(self, x, y, z, times=0):
        if self.is_in_range(x, y, z):
            super().move_point(x, y, z, times)
        else:
            print("不合法参数，机械臂无法到达")

    def move_axis_point(self, c_axis, coord_move, times):
        poi = self.arm.get_point()

        if c_axis == 0:
            if self.is_in_range(int(poi.x + coord_move), int(poi.y), int(poi.z)) == False:
                print("不合法参数，机械臂无法到达")
                return

        elif c_axis == 1:
            if self.is_in_range(int(poi.x), int(poi.y + coord_move), int(poi.z)) == False:
                print("不合法参数，机械臂无法到达")
                return

        elif c_axis == 2:
            if self.is_in_range(int(poi.x), int(poi.y), int(poi.z + coord_move)) == False:
                print("不合法参数，机械臂无法到达")
                return


        super().move_axis_point(c_axis, coord_move, times)


    def move_pixel_wh(self, pix_x, pix_y, pix_z, img_w, img_h):
        if pix_z <= 135 :
            print("不合法参数，机械臂无法到达")
        else:
            return super().move_pixel_wh(pix_x, pix_y, pix_z, img_w, img_h)

    def servo_set_position(self, servo_id, set_position, times, flag):
        if flag:
            if servo_id == 1:
                # 获取舵机的脉冲值
                pos2 = self.arm.get_servo_position(2)
                pos3 = self.arm.get_servo_position(3)
                # print("pos = ", pos1, " pos2 = ", set_position, " pos3 = ", pos3)
                poi = self.arm.forward_kinematics(set_position, pos2, pos3)

            elif servo_id == 2:
                # 获取舵机的脉冲值
                pos1 = self.arm.get_servo_position(1)
                pos3 = self.arm.get_servo_position(3)
                # print("pos = ", pos1, " pos2 = ", set_position, " pos3 = ", pos3)
                poi = self.arm.forward_kinematics(pos1, set_position, pos3)

            elif servo_id == 3:
                # 获取舵机的脉冲值
                pos1 = self.arm.get_servo_position(1)
                pos2 = self.arm.get_servo_position(2)
                # print("pos = ", pos1, " pos2 = ", pos2, " pos3 = ", set_position)
                poi = self.arm.forward_kinematics(pos1, pos2, set_position)

            print("positionX = ", poi.x, "positionY = ", poi.y, "positionZ = ", poi.z)

            # r = pow(pow(poi.x, 2) + pow(poi.y, 2), 0.5)
            # print("R :", r)
            if self.is_in_range(int(poi.x), int(poi.y), int(poi.z)) == False or self.arm.is_value(int(poi.x), int(poi.y), int(poi.z)) == False:
                print("不合法参数，机械臂无法到达")
            else:
                self.arm.servo_move(servo_id, set_position, times)
#                time.sleep((times+200)/1000)
        else:
            self.arm.servo_move(servo_id, set_position, times)
#            time.sleep((times+200)/1000)

    def move_pixel(self, pixelX, pixelY, H):
        if H <= 135:
            print("不合法参数，机械臂无法到达")
        else:
            return super().move_pixel(pixelX, pixelY, H)

    def move_left_pixel(self, pixelX, pixelY, H):
        print('桌面模式不可用')

    def move_right_pixel(self, pixelX, pixelY, H):
        print('桌面模式不可用')

#Control_Arm = Arm()
Control_Arm = desktopArm()

#Control_Arm.move_point(100, -200, 200, 1500)
