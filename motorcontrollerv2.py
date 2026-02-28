import RPi.GPIO as GPIO
import time

AIN1_CONTROL = 5
PWMA = 12
STBY = 16

def test():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([AIN1_CONTROL, PWMA, STBY], GPIO.OUT)
    GPIO.output(STBY, GPIO.HIGH)
    time.sleep(0.5)

    pwm_a = GPIO.PWM(PWMA, 1000)
    pwm_a.start(0)

    print("GPIO HIGH - transistor ON - collector pulls LOW")
    GPIO.output(AIN1_CONTROL, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(100)
    time.sleep(3)

    pwm_a.ChangeDutyCycle(0)
    time.sleep(2)

    print("GPIO LOW - transistor OFF - collector pulled to 5V")
    GPIO.output(AIN1_CONTROL, GPIO.LOW)
    pwm_a.ChangeDutyCycle(100)
    time.sleep(3)

    pwm_a.ChangeDutyCycle(0)
    GPIO.output(STBY, GPIO.LOW)
    pwm_a.stop()
    GPIO.cleanup()
    print("Done. Did motor A spin in DIFFERENT directions?")

test()
