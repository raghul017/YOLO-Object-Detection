import cv2
from yolo_predictions import YOLO_Pred

def main():
    # Initialize the YOLO predictor
    model_path = "Model/weights/best.onnx"  # Updated path to the ONNX model
    yaml_path = "data.yaml"  # Path to your data.yaml file
    
    predictor = YOLO_Pred(model_path, yaml_path)
    
    # Read the sample image
    image_path = "street_image.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Run predictions
    result = predictor.predictions(image)
    
    # Display the result
    cv2.imshow("YOLO Predictions", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 