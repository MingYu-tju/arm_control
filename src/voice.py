import serial
from tool import has_digit_input
import time


default_app_mapping = {
    '00': 'close_app',          # 退出案例
    '01': 'static_grab',        # 色块分拣
    '02': 'calculate',          # 简易计算
    '03': 'obb_grab',           # 角度矫正
    '04': 'refuse_classify',    # 垃圾分类
    '05': 'tictactoe',          # 井字棋
    'd0': 'enable_wifi',        # 连接WiFi
    'd1': 'disable_wifi',       # 关闭热点或者断开WiFi
    'd2': 'enable_ap',          # 打开热点
    'd3': 'read_wifi_ip',       # WIFI地址
    'd4': 'read_wired_ip',      # 网口地址
    'd5': 'read_usb_ip',        # USB虚拟地址
}


class voice:
    def __init__(self, port="/dev/ttyAMA2", baudrate=9600, timeout=3):
        """
        初始化语音类。
        """
        self.ser = serial.Serial(
            port=port,                      # 串口设备路径
            baudrate=baudrate,              # 波特率
            bytesize=serial.EIGHTBITS,      # 数据位
            parity=serial.PARITY_NONE,      # 无奇偶校验
            stopbits=serial.STOPBITS_ONE,   # 停止位
            timeout=timeout                 # 超时时间
        )
        if self.ser.isOpen():
            print(f"串口 {self.ser.port} 打开成功")
        else:
            print(f"无法打开串口 {self.ser.port}")

    def send(self, *data):
        """发送数据到串口"""
        if data:
            frame_start = bytes([0xAA, 0x55])  # 帧头
            frame_end = bytes([0x55, 0xAA])    # 帧尾

            data_bytes = bytes(data)

            data_frame = frame_start + data_bytes + frame_end

            self.ser.write(data_frame)
#            print(f"发送数据：{data_frame}")
        else:
            print("发送数据为空")

    def recv_data(self):
        """从串口接收数据"""
        count = self.ser.inWaiting()
        if count > 0:
            buf = self.ser.read(count)
            recv_data = str(buf.hex())
            print(f"接收到的数据：{recv_data}")
            return recv_data

    def recv(self):
        """从串口接收数据"""
        count = self.ser.inWaiting()
        if count > 0:
            buf = self.ser.read(count)
            recv_data = buf.hex()
            print(f"接收到的数据：{recv_data}")
            if recv_data[:2] == 'ff' and recv_data[4:] == 'ff':
#                print(f"数据：{recv_data[2:4]}")
                return recv_data[2:4]

    def app_type(self, app_mapping = default_app_mapping):
        data = self.recv()
        if data:
            print('data : ', data)
        if data in app_mapping:
            print(app_mapping[data])
            return app_mapping[data]

    def mode_type(self):
        data = self.recv()
        if data == 'f5':
            print('竞赛模式')
            return 'race'
        elif data == 'f6':
            print('展示模式')
            return 'show'
        if has_digit_input() == 'q':
            self.ser.close()
            return None

    def tpu_type(self):
        while True:
            data = self.recv()
            if data == 'fa':
                print('松科加速棒')
                return 'songke'
            elif data == 'fb':
                print('耐能加速棒')
                return 'kneron'
            if has_digit_input() == 'q':
                self.ser.close()
                return None

    def cur_player_type(self):
        data = self.recv()
        if data == 'f7':
            print('用户先手')
            return -1
        elif data == 'f8':
            print('机器先手')
            return 1
        if has_digit_input() == 'q':
            self.ser.close()
            return None

    def send_ip(self, ip_address):
        for char in ip_address:
            if char.isdigit():
                self.send(200 + int(char))  # 数字
            elif char == '.':
                self.send(210)  # 点
            time.sleep(0.4)

    def switch_app(self, app_mapping=default_app_mapping):
        data = self.recv()
        if data is not None and data in app_mapping:
#            self.send(133) # 语音：“初始化中”
            # print(app_mapping[data])
            return app_mapping[data]

    def close(self):
        """关闭串口"""
        self.ser.close()
        print("串口已关闭")

