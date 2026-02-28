import RPi.GPIO as GPIO
import time

AIN1 = 5    # MOSFET A gate
BIN1 = 6    # MOSFET B gate
PWMA = 12   # Motor A speed
PWMB = 13   # Motor B speed
STBY = 16   # Standby

GPIO.setmode(GPIO.BCM)
GPIO.setup([AIN1, BIN1, PWMA, PWMB, STBY], GPIO.OUT)

# Everything off first
GPIO.output(STBY, GPIO.LOW)
GPIO.output(AIN1, GPIO.LOW)
GPIO.output(BIN1, GPIO.LOW)
GPIO.output(PWMA, GPIO.LOW)
GPIO.output(PWMB, GPIO.LOW)
print("All off. Motors should be stopped.")
time.sleep(2)

# Enable shield
GPIO.output(STBY, GPIO.HIGH)

# Start PWM
pwm_a = GPIO.PWM(PWMA, 1000)
pwm_b = GPIO.PWM(PWMB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# FORWARD: Pi LOW -> MOSFET OFF -> 5V to shield -> AIN1/BIN1 HIGH
print("BOTH MOTORS FORWARD...")
GPIO.output(AIN1, GPIO.LOW)
GPIO.output(BIN1, GPIO.LOW)
pwm_a.ChangeDutyCycle(60)
pwm_b.ChangeDutyCycle(60)
time.sleep(3)

# STOP
print("STOPPING...")
pwm_a.ChangeDutyCycle(0)
pwm_b.ChangeDutyCycle(0)
time.sleep(2)

# BACKWARD: Pi HIGH -> MOSFET ON -> GND to shield -> AIN1/BIN1 LOW
print("BOTH MOTORS BACKWARD...")
GPIO.output(AIN1, GPIO.HIGH)
GPIO.output(BIN1, GPIO.HIGH)
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
