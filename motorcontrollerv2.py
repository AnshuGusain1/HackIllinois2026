import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(6, GPIO.OUT)

while True:
    GPIO.output(6, GPIO.HIGH)
    print("GPIO 6 HIGH - measure MOSFET B drain, should be ~0V")
    time.sleep(3)
    GPIO.output(6, GPIO.LOW)
    print("GPIO 6 LOW - measure MOSFET B drain, should be ~5V")
    time.sleep(3)
