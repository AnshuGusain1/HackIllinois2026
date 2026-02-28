import RPi.GPIO as GPIO
import time

PWMA = 12
PWMB = 13
AIN1 = 5    # Goes through level shifter to shield
BIN1 = 6    # Goes through level shifter to shield
STBY = 16

SPEED = 60

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([PWMA, PWMB, AIN1, BIN1, STBY], GPIO.OUT)
    GPIO.output(STBY, GPIO.HIGH)
    time.sleep(0.5)

    pwm_a = GPIO.PWM(PWMA, 1000)
    pwm_b = GPIO.PWM(PWMB, 1000)
    pwm_a.start(0)
    pwm_b.start(0)
    return pwm_a, pwm_b

# Transistor inverts: Pi HIGH → Shield LOW, Pi LOW → Shield HIGH
def forward(pwm_a, pwm_b):
    GPIO.output(AIN1, GPIO.LOW)    # Transistor inverts to HIGH on shield
    GPIO.output(BIN1, GPIO.LOW)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def backward(pwm_a, pwm_b):
    GPIO.output(AIN1, GPIO.HIGH)   # Transistor inverts to LOW on shield
    GPIO.output(BIN1, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def turn_left(pwm_a, pwm_b):
    GPIO.output(AIN1, GPIO.HIGH)   # Left motor backward
    GPIO.output(BIN1, GPIO.LOW)    # Right motor forward
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def turn_right(pwm_a, pwm_b):
    GPIO.output(AIN1, GPIO.LOW)    # Left motor forward
    GPIO.output(BIN1, GPIO.HIGH)   # Right motor backward
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def stop(pwm_a, pwm_b):
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

def cleanup(pwm_a, pwm_b):
    stop(pwm_a, pwm_b)
    GPIO.output(STBY, GPIO.LOW)
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()

def main():
    pwm_a, pwm_b = setup()
    print("Starting motor test. Press Ctrl+C to stop.")

    try:
        while True:
            print("Moving forward...")
            forward(pwm_a, pwm_b)
            time.sleep(3)

            print("Stopping...")
            stop(pwm_a, pwm_b)
            time.sleep(1)

            print("Moving backward...")
            backward(pwm_a, pwm_b)
            time.sleep(3)

            print("Stopping...")
            stop(pwm_a, pwm_b)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping motors...")
    finally:
        cleanup(pwm_a, pwm_b)
        print("GPIO cleaned up. Done.")

if __name__ == "__main__":
    main()
