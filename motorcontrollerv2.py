import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup([5, 12, 16], GPIO.OUT)

# Turn on standby
GPIO.output(16, GPIO.HIGH)

# Start PWM at 60% speed
pwm = GPIO.PWM(12, 1000)
pwm.start(60)

# FORWARD: Pi LOW -> MOSFET OFF -> 10K pulls to 5V -> AIN1 HIGH
print("FORWARD...")
GPIO.output(5, GPIO.LOW)
time.sleep(3)

# STOP
print("STOPPING...")
pwm.ChangeDutyCycle(0)
time.sleep(2)

# BACKWARD: Pi HIGH -> MOSFET ON -> AIN1 pulled to GND -> AIN1 LOW
print("BACKWARD...")
pwm.ChangeDutyCycle(60)
GPIO.output(5, GPIO.HIGH)
time.sleep(3)

# STOP
print("STOPPING...")
pwm.ChangeDutyCycle(0)
time.sleep(1)

print("DONE")
pwm.stop()
GPIO.cleanup()
