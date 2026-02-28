import RPi.GPIO as GPIO
import time

# Motor A (left track)
IN1 = 5
IN2 = 22
ENA = 12

# Motor B (right track)
IN3 = 6
IN4 = 27
ENB = 13

SPEED = 60

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
    pwm_a = GPIO.PWM(ENA, 1000)
    pwm_b = GPIO.PWM(ENB, 1000)
    pwm_a.start(0)
    pwm_b.start(0)
    return pwm_a, pwm_b

def forward(pwm_a, pwm_b, speed=SPEED):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def backward(pwm_a, pwm_b, speed=SPEED):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def turn_left(pwm_a, pwm_b, speed=SPEED):
    # Left track backward, right track forward
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def turn_right(pwm_a, pwm_b, speed=SPEED):
    # Left track forward, right track backward
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def stop(pwm_a, pwm_b):
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

def cleanup(pwm_a, pwm_b):
    stop(pwm_a, pwm_b)
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()

if __name__ == "__main__":
    pwm_a, pwm_b = setup()

    try:
        print("FORWARD...")
        forward(pwm_a, pwm_b)
        time.sleep(3)

        print("STOP...")
        stop(pwm_a, pwm_b)
        time.sleep(1)

        print("BACKWARD...")
        backward(pwm_a, pwm_b)
        time.sleep(3)

        print("STOP...")
        stop(pwm_a, pwm_b)
        time.sleep(1)

        print("TURN LEFT...")
        turn_left(pwm_a, pwm_b)
        time.sleep(2)

        print("STOP...")
        stop(pwm_a, pwm_b)
        time.sleep(1)

        print("TURN RIGHT...")
        turn_right(pwm_a, pwm_b)
        time.sleep(2)

        print("STOP...")
        stop(pwm_a, pwm_b)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cleanup(pwm_a, pwm_b)
        print("DONE")
