from utils import load_camera_params, load_nav_config

# 1. Load the Calibration Parameters
K, dist, res = load_camera_params("camera_K.json")
print(f"Loaded calibration for {res[0]}x{res[1]} resolution.")

# 2. Load the Navigation Settings
config = load_nav_config("settings.ini")
MAP_SCALE = config.getfloat('Mapper', 'scale') # 0.0483398

# 3. Setup Video Source
VIDEO_PATH = "Middle20260226133416.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ALWAYS undistort before doing any mapping or localization
    # This uses the parameters we just loaded
    undistorted = cv2.undistort(frame, K, dist)

    cv2.imshow("Corrected Video", undistorted)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()