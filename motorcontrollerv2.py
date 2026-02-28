import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup([5, 12, 16], GPIO.OUT)

print("All LOW - motor should stop")
GPIO.output(16, GPIO.LOW)
GPIO.output(5, GPIO.LOW)
GPIO.output(12, GPIO.LOW)
time.sleep(3)

print("Turning STBY on...")
GPIO.output(16, GPIO.HIGH)

pwm = GPIO.PWM(12, 1000)
pwm.start(60)

print("FORWARD...")
GPIO.output(5, GPIO.LOW)
time.sleep(3)

print("STOPPING...")
pwm.ChangeDutyCycle(0)
time.sleep(2)

print("BACKWARD...")
pwm.ChangeDutyCycle(60)
GPIO.output(5, GPIO.HIGH)
time.sleep(3)

print("STOPPING...")
pwm.ChangeDutyCycle(0)

pwm.stop()
GPIO.cleanup()
print("DONE")
