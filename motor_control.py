import argparse
import time

import cv2
import numpy as np
import RPi.GPIO as GPIO

# Motor A (left track)
IN1 = 5
IN2 = 22
ENA = 12

# Motor B (right track)
IN3 = 6
IN4 = 27
ENB = 13

SPEED = 60
BLUE_SPEED = 70
GREEN_SPEED = 50
RED_SPEED = 35
MIN_COLOR_RATIO = 0.03  # 3% of frame


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


def detect_dominant_color(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # HSV ranges tuned for typical indoor lighting; may need small adjustments.
    blue_mask = cv2.inRange(hsv, np.array([100, 120, 50]), np.array([130, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 80, 50]), np.array([85, 255, 255]))
    red_mask_1 = cv2.inRange(hsv, np.array([0, 120, 50]), np.array([10, 255, 255]))
    red_mask_2 = cv2.inRange(hsv, np.array([170, 120, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    frame_pixels = frame_bgr.shape[0] * frame_bgr.shape[1]
    blue_ratio = float(np.count_nonzero(blue_mask)) / frame_pixels
    green_ratio = float(np.count_nonzero(green_mask)) / frame_pixels
    red_ratio = float(np.count_nonzero(red_mask)) / frame_pixels

    ratios = {
        "blue": blue_ratio,
        "green": green_ratio,
        "red": red_ratio,
    }

    color = max(ratios, key=ratios.get)
    if ratios[color] < MIN_COLOR_RATIO:
        return None, ratios
    return color, ratios


def run_color_drive(camera_index=0):
    pwm_a, pwm_b = setup()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cleanup(pwm_a, pwm_b)
        raise RuntimeError(f"Could not open camera index {camera_index}")

    try:
        print("Color-drive mode started. Press 'q' to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                stop(pwm_a, pwm_b)
                print("Camera frame read failed. Stopping.")
                break

            color, ratios = detect_dominant_color(frame)

            if color == "blue":
                speed = BLUE_SPEED
                forward(pwm_a, pwm_b, speed=speed)
            elif color == "green":
                speed = GREEN_SPEED
                forward(pwm_a, pwm_b, speed=speed)
            elif color == "red":
                speed = RED_SPEED
                forward(pwm_a, pwm_b, speed=speed)
            else:
                speed = 0
                stop(pwm_a, pwm_b)

            print(
                f"color={color or 'none':>5} speed={speed:>2} "
                f"(blue={ratios['blue']:.3f}, green={ratios['green']:.3f}, red={ratios['red']:.3f})",
                end="\r",
                flush=True,
            )

            cv2.imshow("Color Drive", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.01)

    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()
        cleanup(pwm_a, pwm_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motor control with camera color speed mapping")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    args = parser.parse_args()

    try:
        run_color_drive(camera_index=args.camera)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("DONE")
