import cv2
import numpy as np

def stitch_videos_cropped():
    video_left_path = r"D:\Thursday-26-02-2026_tests\recording_2026_02_26_13_35\Left20260226133416.mp4"
    video_front_path = r"D:\Thursday-26-02-2026_tests\recording_2026_02_26_13_35\Middle20260226133416.mp4"
    video_right_path = r"D:\Thursday-26-02-2026_tests\recording_2026_02_26_13_35\Right20260226133416.mp4"
    output_video_path = r"D:\SDC\Stitched_Output_Cropped_2.mp4"

    cap_left = cv2.VideoCapture(video_left_path)
    cap_front = cv2.VideoCapture(video_front_path)
    cap_right = cv2.VideoCapture(video_right_path)

    if not (cap_left.isOpened() and cap_front.isOpened() and cap_right.isOpened()):
        print("Error: Could not open one or more video files.")
        return

    CANVAS_WIDTH = 4000
    CANVAS_HEIGHT = 2000
    OFFSET_X = 800
    OFFSET_Y = 400

    pts_left = np.float32([[1642, 463], [1781, 540], [1175, 626], [1561, 920]])
    pts_front_for_left = np.float32([[760, 565], [887, 573], [380, 944], [840, 989]])
    pts_front_for_left[:, 0] += OFFSET_X
    pts_front_for_left[:, 1] += OFFSET_Y

    pts_right = np.float32([[178, 493], [329, 429], [660, 738], [149, 1015]])
    pts_front_for_right = np.float32([[1020, 580], [1155, 588], [1352, 1027], [839, 988]])
    pts_front_for_right[:, 0] += OFFSET_X
    pts_front_for_right[:, 1] += OFFSET_Y

    H_left, _ = cv2.findHomography(pts_left, pts_front_for_left)
    H_right, _ = cv2.findHomography(pts_right, pts_front_for_right)
    M_front_shift = np.float32([[1, 0, OFFSET_X], [0, 1, OFFSET_Y], [0, 0, 1]])

    ret_l, frame_l = cap_left.read()
    ret_f, frame_f = cap_front.read()
    ret_r, frame_r = cap_right.read()

    warp_left = cv2.warpPerspective(frame_l, H_left, (CANVAS_WIDTH, CANVAS_HEIGHT))
    warp_right = cv2.warpPerspective(frame_r, H_right, (CANVAS_WIDTH, CANVAS_HEIGHT))
    warp_front = cv2.warpPerspective(frame_f, M_front_shift, (CANVAS_WIDTH, CANVAS_HEIGHT))
    
    first_stitched = np.maximum.reduce([warp_left, warp_front, warp_right])

    gray = cv2.cvtColor(first_stitched, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(thresh)
    
    x, y, w, h = cv2.boundingRect(coords)
    
    print(f"Optimal Crop: X:{x}, Y:{y}, Width:{w}, Height:{h}")


    fps = cap_front.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    cropped_first_frame = first_stitched[y:y+h, x:x+w]
    out.write(cropped_first_frame)

    frame_count = 1

    while True:
        ret_l, frame_l = cap_left.read()
        ret_f, frame_f = cap_front.read()
        ret_r, frame_r = cap_right.read()

        if not (ret_l and ret_f and ret_r):
            break

        warp_left = cv2.warpPerspective(frame_l, H_left, (CANVAS_WIDTH, CANVAS_HEIGHT))
        warp_right = cv2.warpPerspective(frame_r, H_right, (CANVAS_WIDTH, CANVAS_HEIGHT))
        warp_front = cv2.warpPerspective(frame_f, M_front_shift, (CANVAS_WIDTH, CANVAS_HEIGHT))

        stitched_frame = np.maximum.reduce([warp_left, warp_front, warp_right])

        cropped_frame = stitched_frame[y:y+h, x:x+w]

        out.write(cropped_frame)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

        # Display the cropped version
        display_img = cv2.resize(cropped_frame, (1200, int(1200 * (h/w))))
        cv2.imshow("Cropped Processing Video (Press 'q' to quit)", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_front.release()
    cap_right.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nCropped video saved to: {output_video_path}")

if __name__ == "__main__":
    stitch_videos_cropped()