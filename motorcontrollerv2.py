import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup([5, 12, 16], GPIO.OUT)
GPIO.output(16, GPIO.HIGH)  # STBY on

pwm = GPIO.PWM(12, 1000)
pwm.start(60)

print("Forward - AIN1 HIGH (Pi LOW, MOSFET OFF)")
GPIO.output(5, GPIO.LOW)
time.sleep(3)

print("Stop")
pwm.ChangeDutyCycle(0)
time.sleep(1)

print("Backward - AIN1 LOW (Pi HIGH, MOSFET ON)")
pwm.ChangeDutyCycle(60)
GPIO.output(5, GPIO.HIGH)
time.sleep(3)

print("Done")
pwm.stop()
GPIO.cleanup()
