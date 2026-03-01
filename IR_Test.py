import RPi.GPIO as GPIO
import time

IR_PIN = 20

GPIO.setmode(GPIO.BCM)
GPIO.setup(IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Also set GPIO21 HIGH to power the IR module (temporary hack)
GPIO.setup(21, GPIO.OUT)
GPIO.output(21, GPIO.HIGH)

def read_nec_code():
    """Read NEC IR protocol signal"""
    timeout = 0.02
    
    # Wait for signal to go LOW (start of transmission)
    while GPIO.input(IR_PIN) == GPIO.HIGH:
        pass
    
    # Measure start pulse
    start = time.time()
    while GPIO.input(IR_PIN) == GPIO.LOW:
        if time.time() - start > timeout:
            return None
    
    while GPIO.input(IR_PIN) == GPIO.HIGH:
        if time.time() - start > timeout:
            return None
    
    # Read 32 bits
    code = 0
    for i in range(32):
        # LOW period
        t_start = time.time()
        while GPIO.input(IR_PIN) == GPIO.LOW:
            if time.time() - t_start > timeout:
                return None
        
        # HIGH period - duration determines 0 or 1
        t_start = time.time()
        while GPIO.input(IR_PIN) == GPIO.HIGH:
            if time.time() - t_start > timeout:
                return None
        
        bit_time = time.time() - t_start
        code = (code << 1) | (1 if bit_time > 0.001 else 0)
    
    return code

print("IR RECEIVER TEST")
print("Point Elegoo remote at receiver and press buttons")
print("Press Ctrl+C to quit")
print()

try:
    while True:
        code = read_nec_code()
        if code is not None:
            print(f"Button code: {code} (hex: {hex(code)})")
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nDone")
    GPIO.cleanup()

