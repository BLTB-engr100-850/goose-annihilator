import serial
import time
import numpy as np

arduino = serial.Serial('COM5', 115200)

last_sent_time = 0
command_interval = 0.01  # Minimum interval between commands (in seconds)

def send_command(command):
  command = command + "\n"
  global last_sent_time
  current_time = time.time()
  if current_time - last_sent_time >= command_interval:
    try:
      arduino.write(command.encode('utf-8'))
      last_sent_time = current_time
    except serial.SerialException as e:
      print(f"Serial communication error: {e}")
  else:
    print("Skipping command to prevent flooding.")


if __name__ == "__main__":
  send_command(f"1020 1135 0")
  arduino.close()