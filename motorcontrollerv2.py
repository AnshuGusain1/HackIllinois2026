import RPi.GPIO as GPIO
import time

PWMA = 12
STBY = 16
AIN1 = 5

def test():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([PWMA, STBY, AIN1], GPIO.OUT)
    GPIO.output(STBY, GPIO.HIGH)
    time.sleep(0.5)

    pwm_a = GPIO.PWM(PWMA, 1000)
    pwm_a.start(0)

    print("Test 1: AIN1 = HIGH")
    GPIO.output(AIN1, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(100)
    time.sleep(3)

    pwm_a.ChangeDutyCycle(0)
    time.sleep(2)

    print("Test 2: AIN1 = LOW")
    GPIO.output(AIN1, GPIO.LOW)
    pwm_a.ChangeDutyCycle(100)
    time.sleep(3)

    pwm_a.ChangeDutyCycle(0)
    GPIO.output(STBY, GPIO.LOW)
    pwm_a.stop()
    GPIO.cleanup()
    print("Done. Did motor A change direction?")

test()
