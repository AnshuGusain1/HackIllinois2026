import RPi.GPIO as GPIO
import time

# Motor pins (L298N)
IN1 = 5
IN2 = 22
IN3 = 6
IN4 = 27
ENA = 12
ENB = 13

# IR pin
IR_PIN = 20

# IR codes from Elegoo remote
FORWARD  = 0xff629d
BACKWARD = 0xffa857
LEFT     = 0xff22dd
RIGHT    = 0xffc23d
STOP     = 0xff02fd

SPEED = 70

GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(21, GPIO.OUT)
GPIO.output(21, GPIO.HIGH)

pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

def forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def backward():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def turn_left():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def turn_right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def stop():
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

def read_nec_code():
    timeout = 0.02
    while GPIO.input(IR_PIN) == GPIO.HIGH:
        return None
    
    start = time.time()
    while GPIO.input(IR_PIN) == GPIO.LOW:
        if time.time() - start > timeout:
            return None
    while GPIO.input(IR_PIN) == GPIO.HIGH:
        if time.time() - start > timeout:
            return None
    
    code = 0
    for i in range(32):
        t_start = time.time()
        while GPIO.input(IR_PIN) == GPIO.LOW:
            if time.time() - t_start > timeout:
                return None
        t_start = time.time()
        while GPIO.input(IR_PIN) == GPIO.HIGH:
            if time.time() - t_start > timeout:
                return None
        code = (code << 1) | (1 if time.time() - t_start > 0.001 else 0)
    
    return code

print("ELEGOO TANK - IR REMOTE CONTROL")
print("Up=Forward  Down=Backward  Left/Right=Turn  OK=Stop")
print("Ctrl+C to quit")
print()

try:
    while True:
        code = read_nec_code()
        if code == FORWARD:
            print("FORWARD")
            forward()
        elif code == BACKWARD:
            print("BACKWARD")
            backward()
        elif code == LEFT:
            print("LEFT")
            turn_left()
        elif code == RIGHT:
            print("RIGHT")
            turn_right()
        elif code == STOP:
            print("STOP")
            stop()
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopping...")
    stop()
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    print("Done")
