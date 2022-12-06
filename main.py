import cv2
import numpy as np

def detection(video):
    height, width, _ = video.shape
    net.setInput(cv2.dnn.blobFromImage(video, 1 / 255, (608, 608),(0, 0, 0), swapRB=True, crop=False))
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0
    for out in outs:
        for i in out:
            scores = i[5:]
            if scores[np.argmax(scores)] > 0:
                boxes.append([int(i[0] * width) - int(i[2] * width) // 2,
                            int(i[1] * height) - int(i[3] * height) // 2,
                            int(i[2] * width), int(i[3] * height)])
                class_indexes.append(np.argmax(scores))
                class_scores.append(float(scores[np.argmax(scores)]))

    for box_index in cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4):
        if classes[class_indexes[box_index]] in classes_to_look_for:
            objects_count += 1
            x, y, w, h = boxes[box_index]
            video = cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 0), 1)
    video = cv2.putText(video, "number of cars = " + str(objects_count), (10, 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    finalVideo = cv2.putText(video, "number of cars = " + str(objects_count), (10, 30), cv2.FONT_HERSHEY_TRIPLEX,
                             0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return finalVideo


if __name__ == '__main__':
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")
    video = "1.mp4"
    classes_to_look_for = ["car"]
    while True:
        try:
            video_camera_capture = cv2.VideoCapture(video)
            while video_camera_capture.isOpened():
                ret, frame = video_camera_capture.read()
                if not ret:
                    break
                frame = detection(frame)
                frame = cv2.resize(frame, (1920 // 3, 1080 // 3))
                cv2.imshow("Video Capture", frame)
                cv2.waitKey(1)
            video_camera_capture.release()
            cv2.destroyAllWindows()
        except KeyboardInterrupt:
            pass