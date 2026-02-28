# Mechathon Sand Unevenness Pipeline

## Files
- `sand_unevenness_pipeline.py`: real-time OpenCV pipeline for per-frame unevenness + grid accumulation + revisit targets
- `calibrate_homography.py`: 4-click calibration utility for image-to-ground homography

## Dependencies
- Python 3.9+
- `opencv-python`
- `numpy`

Install:
```powershell
pip install opencv-python numpy
```

## Step-by-step bring-up

0. Find your camera index (recommended first)
```powershell
python sand_unevenness_pipeline.py --list-cameras --max-camera-index 8 --backend dshow
```
Use the reported index for all runs, for example `--camera 1`.
If none found, try:
```powershell
python sand_unevenness_pipeline.py --list-cameras --max-camera-index 8 --backend msmf
```

1. Camera + preprocessing only
```powershell
python sand_unevenness_pipeline.py --camera 0 --width 640 --height 480 --roi 0 80 640 320
```
Verify:
- `preprocessed` view has reduced shadows and stable contrast under mild light changes.
- FPS stays near or above 10.

2. Gradient + roughness signal
Verify:
- `gradient` highlights ridges/tool marks.
- `rough` rises when you create dunes/ridges and drops after flattening.

3. Homography calibration
```powershell
python calibrate_homography.py --camera 0 --width 640 --height 480 --sandbox-w 1.2 --sandbox-h 0.8 --outfile img_to_robot_h.npy
```
Click corners in this order:
1) `(0,0)` 2) `(W,0)` 3) `(W,H)` 4) `(0,H)`

4. Grid mapping + revisit policy
```powershell
python sand_unevenness_pipeline.py --hfile img_to_robot_h.npy --sandbox-w 1.2 --sandbox-h 0.8 --grid-w 20 --grid-h 20
```
Verify:
- `grid_score` hot regions align with visible uneven patches.
- revisit target count is stable with hysteresis (no rapid flicker).

## Integrating real robot state

In `sand_unevenness_pipeline.py`, replace the placeholder values inside `main()`:
- `pose_xy_yaw = (x_m, y_m, yaw_rad)` from wheel odom + IMU yaw
- `imu_roll_pitch_vib = (roll_deg, pitch_deg, vibration_energy)`

Expected vibration input:
- scalar RMS/high-pass acceleration energy per frame window

## Quick tuning
- Faster on Pi: smaller ROI, larger `sample_stride_px`, use Sobel (default), reduce camera resolution.
- More shadow robustness: increase `shadow_blur_ksize` (odd values like 41, 61).
- More stable revisit map: lower `map_alpha`, increase `min_hits_for_revisit`.
- More/less aggressive revisit behavior: adjust `revisit_hi` / `revisit_lo`.
