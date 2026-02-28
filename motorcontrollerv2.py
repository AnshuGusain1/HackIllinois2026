import RPi.GPIO as GPIO
import time

PWMA = 12
PWMB = 13
AIN1 = 5
BIN1 = 6
STBY = 16

def test():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([PWMA, PWMB, AIN1, BIN1, STBY], GPIO.OUT)
    GPIO.output(STBY, GPIO.HIGH)
    time.sleep(0.5)

    pwm_a = GPIO.PWM(PWMA, 1000)
    pwm_b = GPIO.PWM(PWMB, 1000)
    pwm_a.start(0)
    pwm_b.start(0)

    # Test 1: Direct LOW (should work - this was your "backward")
    print("Test 1: AIN1=LOW (should spin)")
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(BIN1, GPIO.LOW)
    pwm_a.ChangeDutyCycle(100)
    pwm_b.ChangeDutyCycle(100)
    time.sleep(3)

    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)
    time.sleep(2)

    # Test 2: Use PWM to simulate HIGH (trick the Schmitt inverter)
    print("Test 2: AIN1=PWM fake HIGH")
    GPIO.setup(AIN1, GPIO.OUT)
    GPIO.setup(BIN1, GPIO.OUT)
    ain_pwm = GPIO.PWM(AIN1, 10000)
    bin_pwm = GPIO.PWM(BIN1, 10000)
    ain_pwm.start(100)
    bin_pwm.start(100)
    pwm_a.ChangeDutyCycle(100)
    pwm_b.ChangeDutyCycle(100)
    time.sleep(3)

    ain_pwm.stop()
    bin_pwm.stop()
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)
    GPIO.output(STBY, GPIO.LOW)
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    print("Done. Did the directions change between Test 1 and Test 2?")

test()
