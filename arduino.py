import serial

arduino = serial.Serial('COM5', 115200)

def send_command(command):
  command = command + "\n"
  try:
    arduino.write(command.encode('utf-8'))
  except serial.SerialException as e:
    print(f"Serial communication error: {e}")
  pass