#!/usr/bin/env python3
"""
Forward-only DC motor control (ON/OFF) for Raspberry Pi 4B.
Uses one GPIO pin to switch a MOSFET/transistor motor driver stage.
"""

import time

import RPi.GPIO as GPIO

# Raspberry Pi 4B GPIO (BCM numbering)
# GPIO18 = physical pin 12
MOTOR_PIN = 18


def motor_on() -> None:
    GPIO.output(MOTOR_PIN, GPIO.HIGH)


def motor_off() -> None:
    GPIO.output(MOTOR_PIN, GPIO.LOW)


def main() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(MOTOR_PIN, GPIO.OUT, initial=GPIO.LOW)

    try:
        print("Motor ON (forward). Press Ctrl+C to stop.")
        motor_on()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping motor...")
    finally:
        motor_off()
        GPIO.cleanup()


if __name__ == "__main__":
    main()

# cd "/path/to/your/Current Projects"
# sudo apt update
# sudo apt install -y python3-rpi.gpio
# python3 motor_pwm.py


