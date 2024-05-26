import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('./best.pt')

# Open the video file
video_path = "./traffic_test.mp4"
cap = cv2.VideoCapture(video_path)

#  {0: 'green', 1: 'yellow', 2: 'red', 3: 'all_green', 4: 'left'}
count_dict = {0: 0, 1: 3, 2: 1, 3: 0, 4: 2}
results_dict = {0: 'go', 1: 'stop', 2: 'left', 3: 'yellow'}   # 결과 dictionary
flutter_result = 'None'  # 플러터로 보낼 result

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        for result in results:
            traffic_weight = [0, 0, 0, 0]   # 신호등 가중치

            clist = result.boxes.cls

            for cno in clist:
                if int(cno) >= len(model.names):
                    continue
                # 가중치 증가
                try:
                    traffic_weight[count_dict[int(cno)]] += 1
                except:
                    print("Error Detection")

        # 결과가 없는 경우
        if max(traffic_weight) == 0:
            flutter_result = 'None'
        else:
            # 결과가 있는 경우
            flutter_result = results_dict[traffic_weight.index(
                max(traffic_weight))]

        print("flutter: ", flutter_result)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
