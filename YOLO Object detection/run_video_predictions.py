import cv2
from yolo_predictions import YOLO_Pred
import time
import numpy as np

def main():
    # Initialize the YOLO predictor
    model_path = "Model/weights/best.onnx"
    yaml_path = "data.yaml"
    predictor = YOLO_Pred(model_path, yaml_path)
    
    # Open the video file
    video_path = "Road Traffic.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writer object
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize FPS calculation variables
    frame_count = 0
    start_time = time.time()
    
    # Set OpenCV optimization flags
    cv2.setUseOptimized(True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run predictions on the frame
        result = predictor.predictions(frame)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:  # Update FPS every 30 frames
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            cv2.putText(result, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow("YOLO Video Predictions", result)
        
        # Write the frame to output video
        out.write(result)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 