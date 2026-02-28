import RPi.GPIO as GPIO
import time

# Pin definitions (BCM numbering)
PWMA = 12   # Left motor speed (hardware PWM0)
PWMB = 13   # Right motor speed (hardware PWM1)
AIN1 = 5    # Left motor direction
BIN1 = 6    # Right motor direction
STBY = 16   # Standby enable

# Motor speed (0-100 duty cycle percentage)
SPEED = 60

def setup():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup([PWMA, PWMB, AIN1, BIN1, STBY], GPIO.OUT)
    
    # Enable the H-bridge
    GPIO.output(STBY, GPIO.HIGH)
    
    # Set up PWM on both motor channels at 1kHz
    pwm_a = GPIO.PWM(PWMA, 1000)
    pwm_b = GPIO.PWM(PWMB, 1000)
    pwm_a.start(0)
    pwm_b.start(0)
    
    return pwm_a, pwm_b

def forward(pwm_a, pwm_b):
    GPIO.output(AIN1, GPIO.HIGH)
    GPIO.output(BIN1, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(SPEED)
    pwm_b.ChangeDutyCycle(SPEED)

def backward(pwm_a, pwm_b):
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(BIN1, GPIO.LOW)
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
    
    print("Starting motor loop. Press Ctrl+C to stop.")
    
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
