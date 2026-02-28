import RPi.GPIO as GPIO
import time

PWMA = 12
PWMB = 13
BIN1 = 6
STBY = 16

def test():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([PWMA, PWMB, BIN1, STBY], GPIO.OUT)
    GPIO.output(STBY, GPIO.HIGH)
    time.sleep(0.5)

    pwm_a = GPIO.PWM(PWMA, 1000)
    pwm_b = GPIO.PWM(PWMB, 1000)
    pwm_a.start(0)
    pwm_b.start(0)

    # AIN1 is physically wired to 5V (not controlled by code)
    # So we only test motor A here

    print("Test: AIN1 wired to 5V - motor A should spin")
    pwm_a.ChangeDutyCycle(100)
    time.sleep(3)

    pwm_a.ChangeDutyCycle(0)
    time.sleep(2)

    # Now test BIN1 with GPIO LOW for comparison
    print("Test: BIN1=LOW - motor B should spin")
    GPIO.output(BIN1, GPIO.LOW)
    pwm_b.ChangeDutyCycle(100)
    time.sleep(3)

    pwm_b.ChangeDutyCycle(0)
    GPIO.output(STBY, GPIO.LOW)
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    print("Done.")

test()
