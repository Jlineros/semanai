import cv2
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
import cvzone
from polym import PolylineManager  # Ensure this imports the class correctly

#Set Videopath
video_call = "videos_test\\grupo1.mp4"

# Initialize the video stream
stream = CamGear(source=video_call).start()
#stream = CamGear(source='https://www.youtube.com/watch?v=_TusTf0iZQU', stream_mode=True, logging=True).start()

# Load COCO class names
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load an official or custom model
model = YOLO("yolo11n.pt") #(n, s, m , l, x)
#model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
#model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
#model = YOLO("path/to/best.pt")  # Load a custom trained model

# Create a PolylineManager instance
# PolylineManager: This class allows you to define and manage polygons on
# the video frame. These polygons are used as regions of interest (ROIs), 
# and the system checks if detected objects are crossing between these regions.


polyline_manager = PolylineManager()

# Set up the OpenCV window
cv2.namedWindow('RGB')


# Mouse Callback: The RGB() function is triggered when you click the left 
# mouse button on the frame. It records the clicked point as part of a 
# polygon. These polygons are used as "areas" to track objects entering 
# and exiting.

# Mouse callback to get mouse movements
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        polyline_manager.add_point((x, y))

# Set the mouse callback function
cv2.setMouseCallback('RGB', RGB)
count = 0
going_up = {}
going_down = {}
gnu=[]
gnd=[]
while True:
    # Read a frame from the video stream
    frame = stream.read()
    count += 1
    if count % 3 != 0: #Adjust the amount of frames processed
        continue
    elif frame is None:
        print("Empty frame received, stopping...")
        break  # Stop the loop if no frame is received

    frame = cv2.resize(frame, (1020, 500))
# Object Detection and Tracking:

# Every third frame is processed (count % 3 != 0: continue) to optimize the detection loop.
# YOLO performs detection and returns results in terms of bounding boxes, class IDs, track IDs, and confidence scores.
# The detected objects are filtered for a specific class (class 2 here, which is "car" in the COCO dataset).
# The center of each bounding box (cx, cy) is calculated, and the object's position is checked relative to two predefined areas ("area1" and "area2").
    
    results = model.track(source=frame, conf=0.7, iou=0.5, show=True, persist=True, classes=[2])

# If object confidence score will be low, i.e lower than track_high_thresh, 
# then there will be no tracks successfully returned and updated.

# Argument        Type        Default                  Description
# source          str         'ultralytics/assets'     Specifies the data source for inference. Can be an image path, video file, directory, URL, or device ID for live feeds. Supports a wide range of formats and sources, enabling flexible application across different types of input.
# conf            float       0.25                     Sets the minimum confidence threshold for detections. Objects detected with confidence below this threshold will be disregarded. Adjusting this value can help reduce false positives.
# iou             float       0.7                      Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
# imgsz           int/tuple   640                      Defines the image size for inference. Can be a single integer 640 for square resizing or a (height, width) tuple. Proper sizing can improve detection accuracy and processing speed.
# half            bool        False                    Enables half-precision (FP16) inference, which can speed up model inference on supported GPUs with minimal impact on accuracy.
# device          str         None                     Specifies the device for inference (e.g., cpu, cuda:0 or 0). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.
# max_det         int         300                      Maximum number of detections allowed per image. Limits the total number of objects the model can detect in a single inference, preventing excessive outputs in dense scenes.
# vid_stride      int         1                        Frame stride for video inputs. Allows skipping frames in videos to speed up processing at the cost of temporal resolution. A value of 1 processes every frame, higher values skip frames.
# stream_buffer   bool        False                    Determines whether to queue incoming frames for video streams. If False, old frames get dropped to accommodate new frames (optimized for real-time applications). If True, queues new frames in a buffer, ensuring no frames get skipped, but will cause latency if inference FPS is lower than stream FPS.
# visualize       bool        False                    Activates visualization of model features during inference, providing insights into what the model is "seeing". Useful for debugging and model interpretation.
# augment         bool        False                    Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.
# agnostic_nms    bool        False                    Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.
# classes         list[int]   None                     Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.
# retina_masks    bool        False                    Uses high-resolution segmentation masks if available in the model. This can enhance mask quality for segmentation tasks, providing finer detail.
# embed           list[int]   None                     Specifies the layers from which to extract feature vectors or embeddings. Useful for downstream tasks like clustering or similarity search.

# Visualization arguments:

# Argument        Type        Default                  Description
# show            bool        False                    If True, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing.
# save            bool        False/True               Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results. Defaults to True when using CLI & False when used in Python.
# save_frames     bool        False                    When processing videos, saves individual frames as images. Useful for extracting specific frames or for detailed frame-by-frame analysis.
# save_txt        bool        False                    Saves detection results in a text file, following the format [class] [x_center] [y_center] [width] [height] [confidence]. Useful for integration with other analysis tools.
# save_conf       bool        False                    Includes confidence scores in the saved text files. Enhances the detail available for post-processing and analysis.
# save_crop       bool        False                    Saves cropped images of detections. Useful for dataset augmentation, analysis, or creating focused datasets for specific objects.
# show_labels     bool        True                     Displays labels for each detection in the visual output. Provides immediate understanding of detected objects.
# show_conf       bool        True                     Displays the confidence score for each detection alongside the label. Gives insight into the model's certainty for each detection.
# show_boxes      bool        True                     Draws bounding boxes around detected objects. Essential for visual identification and location of objects in images or video frames.
# line_width      None/int    None                     Specifies the line width of bounding boxes. If None, the line width is automatically adjusted based on the image size. Provides visual customization for clarity.

# Load COCO class names
    with open("coco.txt", "r") as f:
        class_names = f.read().splitlines()

    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

        # Draw boxes and labels on the frame
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            
            # Calculate the center of the bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
# Crossing Events: The code tracks objects moving between two areas:

# going_up: Tracks objects moving from "area1" to "area2."
         
            if polyline_manager.point_polygon_test((cx, cy), 'area1'):
                going_up[track_id] = (cx, cy)
            if track_id in going_up:
               if polyline_manager.point_polygon_test((cx, cy), 'area2'): 
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                  cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                  cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                  if gnu.count(track_id)==0:
                     gnu.append(track_id)

# going_down: Tracks objects moving from "area2" to "area1."
# When an object crosses from one area to another, it is counted, and a rectangle is drawn around the detected object.

            if polyline_manager.point_polygon_test((cx, cy), 'area2'):
                going_down[track_id] = (cx, cy)
            if track_id in going_down:
               if polyline_manager.point_polygon_test((cx, cy), 'area1'): 
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                  cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                  cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                  if gnd.count(track_id)==0:
                     gnd.append(track_id)

# Counting the Objects: The system maintains two lists:

# gnu: Keeps track of unique IDs of objects that moved up (from area1 to area2).
# gnd: Keeps track of unique IDs of objects that moved down (from area2 to area1).
# The length of these lists represents the total counts of objects moving in each direction, and the counts are displayed on the screen.
              
    godown=len(gnd)       
    goup=len(gnu)
    cvzone.putTextRect(frame, f'GoDown:-{godown}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'GoUp:-{goup}', (50, 160), 2, 2)

    # Draw polylines and points on the frame
    frame = polyline_manager.draw_polylines(frame)

# Displaying Results: The processed frame, with bounding boxes, labels, and polylines drawn, is displayed using OpenCV's imshow() function. You can see the objects being tracked and the counts for each direction on the video.

    # Display the frame
    cv2.imshow("RGB", frame)

    # Handle key events for polyline management
    if not polyline_manager.handle_key_events():
        break

# Release the video capture object and close the display window
stream.stop()
cv2.destroyAllWindows()