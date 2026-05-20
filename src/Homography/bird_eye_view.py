import cv2
import numpy as np

def generate_bev_map():
   
    input_video_path = r"D:\SDC\Stitched_Output_Cropped.mp4"
    output_video_path = r"D:\SDC\BEV_Map_Video_2.mp4"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open the stitched video.")
        return

    BEV_WIDTH = 800
    BEV_HEIGHT = 1200

    # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
    src_pts = np.float32([
    [924, 542],   # TOP-LEFT
    [1021, 547],  # TOP-RIGHT
    [1014, 681],  # BOTTOM-RIGHT
    [792, 670],   # BOTTOM-LEFT
])

    #src_pts = np.float32([
    #    [1559, 565], 
    #    [1693, 573], 
    #    [1181, 948], 
    #    [739, 907]
    #])

    # Based on assumption that zebra is 60cm wide and 300cm long.
    STRIPE_WIDTH = 60
    STRIPE_LENGTH = 300

    X_CENTER = BEV_WIDTH // 2  # 400
    Y_BOTTOM = 1000            # Leaving 200px (2 meters) of space behind the zebra stripe
    
   
    dst_pts = np.float32([
        [X_CENTER - (STRIPE_WIDTH // 2), Y_BOTTOM - STRIPE_LENGTH], # Top-Left
        [X_CENTER + (STRIPE_WIDTH // 2), Y_BOTTOM - STRIPE_LENGTH], # Top-Right
        [X_CENTER + (STRIPE_WIDTH // 2), Y_BOTTOM],                 # Bottom-Right
        [X_CENTER - (STRIPE_WIDTH // 2), Y_BOTTOM]                  # Bottom-Left
    ])

    # Calculate the BEV Matrix
    M_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Set up the Video Writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (BEV_WIDTH, BEV_HEIGHT))


    frame_count = 0

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bev_frame = cv2.warpPerspective(frame, M_bev, (BEV_WIDTH, BEV_HEIGHT))

        out.write(bev_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

        display_img = cv2.resize(bev_frame, (400, 600))
        cv2.imshow("Bird's-Eye View (Press 'q' to quit)", display_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n BEV Video saved to: {output_video_path}")

if __name__ == "__main__":
    generate_bev_map()