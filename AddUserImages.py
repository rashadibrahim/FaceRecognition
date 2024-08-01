import cv2
import os

cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    # Create a directory to save the captured images
    parentDir = './newDataset' # this directory can be adjusted to be the directory of the Dataset
    name = input("Enter Your Name: ") # User Enters His Name as it will be used As The Label for the image
    save_dir = os.path.join(parentDir, name) # Directory Name is the User Name
    os.makedirs(save_dir, exist_ok=True) 

    # Capture 10 images
    for i in range(10):
        # Read a frame from the camera
        ret, frame = cap.read()
        # Check if the frame is read successfully
        if not ret:
            print(f"Error: Could not read frame {i + 1}")
            break
       
        # Convert the Image Into Gray Scale, as Our Dataset Consists of Only Gray Scale Images
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the frame
        cv2.imshow('Captured Image', gray_frame)

        # Save the frame as an image in the specified location
        image_filename = os.path.join(save_dir, f'{i + 1}.png') 
        # We Can Adjust the Extension to Be pgm, so that out model can detect it
        cv2.imwrite(image_filename, gray_frame)
        print(f"Image {i + 1} captured and saved as {image_filename}")

        # Wait for a key press to capture the next image
        cv2.waitKey(0)

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()





