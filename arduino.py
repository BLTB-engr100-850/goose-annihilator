import serial

arduino = serial.Serial('COM5', 9600)

def send_command(command):
  arduino.write(command.encode())

if __name__ == "__main__":
  send_command("10 10 1")
  arduino.close()