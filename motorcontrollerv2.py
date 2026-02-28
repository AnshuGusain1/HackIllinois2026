import RPi.GPIO as GPIO
import time

IN1 = 5
IN2 = 22
IN3 = 6
IN4 = 27
ENA = 12
ENB = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)

pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# FORWARD
print("BOTH MOTORS FORWARD...")
GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)
GPIO.output(IN3, GPIO.HIGH)
GPIO.output(IN4, GPIO.LOW)
pwm_a.ChangeDutyCycle(60)
pwm_b.ChangeDutyCycle(60)
time.sleep(3)

# STOP
print("STOPPING...")
pwm_a.ChangeDutyCycle(0)
pwm_b.ChangeDutyCycle(0)
time.sleep(2)

# BACKWARD
print("BOTH MOTORS BACKWARD...")
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.HIGH)
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.HIGH)
pwm_a.ChangeDutyCycle(60)
pwm_b.ChangeDutyCycle(60)
time.sleep(3)

# STOP
print("STOPPING...")
pwm_a.ChangeDutyCycle(0)
pwm_b.ChangeDutyCycle(0)
time.sleep(1)

pwm_a.stop()
pwm_b.stop()
GPIO.cleanup()
print("DONE")
