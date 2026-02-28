import RPi.GPIO as GPIO
import time

ENA = 12    # Left motor speed (PWM)
ENB = 13    # Right motor speed (PWM)
IN1 = 5     # Left motor direction 1
IN2 = 22    # Left motor direction 2
IN3 = 6     # Right motor direction 1
IN4 = 27    # Right motor direction 2

SPEED = 60

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([ENA, ENB, IN1, IN2, IN3, IN4], GPIO.OUT)

    pwm_a = GPIO.PWM(ENA, 1000)
    pwm_b = GPIO.PWM(ENB, 1000)
    pwm_a.start(0)
    pwm_b.start(0)
    return pwm_a, pwm_b

def forward(pwm_a, pwm_b):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def backward(pwm_a, pwm_b):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def turn_left(pwm_a, pwm_b):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def turn_right(pwm_a, pwm_b):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def stop(pwm_a, pwm_b):
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

def cleanup(pwm_a, pwm_b):
    stop(pwm_a, pwm_b)
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
