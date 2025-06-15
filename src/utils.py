# utils.py
import serial

def send_command_to_stm32(ser, command_str):
    """向 STM32 发送命令"""
    if ser and ser.is_open:
        try:
            ser.write(f"{command_str}\n".encode('utf-8'))
            print(f"[STM32 TX] Sent: {command_str}")
            return True
        except serial.SerialException as e:
            print(f"[STM32 TX Error] Failed to send '{command_str}': {e}")
            return False
    return False