import RPi.GPIO as GPIO
import time

# L298N pins
VS_PIN = None  # powered by battery directly
IN1 = 5
IN2 = 22
ENA = 12

def test():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([IN1, IN2, ENA], GPIO.OUT)

    # Enable at full power, no PWM
    GPIO.output(ENA, GPIO.HIGH)

    print("Test: IN1=HIGH, IN2=LOW - motor A should spin")
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(5)

    print("Test: IN1=LOW, IN2=HIGH - motor A should spin other way")
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    time.sleep(5)

    GPIO.output(ENA, GPIO.LOW)
    GPIO.cleanup()
    print("Done. Did motor A spin?")

test()
