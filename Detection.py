import torch
import numpy as np
import cv2
from time import time
import sys

sys.path.insert(0, "yolov5")


class FaceDetection:
    """
    Class implements Yolo5 model to make inferences on real time video using Opencv2.
    """

    def __init__(self, model_name, image_size=1024, conf=0.1, capture_index=0):
        """
        Initializes the class with camera to capture video.
        :param model_name: Name of model.
        :param capture_index: Number of camera. Defect: 0.
        """
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.image_size = image_size
        self.conf = conf
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.color = (255, 255, 255)
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """

        return cv2.VideoCapture(self.capture_index, cv2.CAP_DSHOW)

    def load_model(self, model_name):
        """
        Loads pre-trained Yolo5 model.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = cv2.resize(frame, (self.image_size, self.image_size))
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= self.conf:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
 
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)
                cv2.putText(frame, f"{self.class_to_label(labels[i])}: {row[4]*100:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color, 2)

        return frame

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()

        print(f'Image size: {int(self.image_size)}x{int(self.image_size)}')
        print(f'Clases: {self.classes}')

        while True:

            ret, frame = cap.read()
            assert ret

            frame = cv2.flip(frame, 1)

            start_time = time()
            results = self.score_frame(frame)
            labels = [self.class_to_label(results[0][i]) for i in range(len(results[0]))]
            frame = self.plot_boxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            print(f"Frames Per Second : {int(fps)} - Detected: {labels}")



            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.color, 2)

            output = cv2.resize(frame, (512, 512))

            cv2.imshow('YOLOv5 Detection', output)

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()


# Create a new object and execute.
if __name__ == '__main__':
    model_name = 'yolov5/best.pt'
    image_size = 512
    conf = .5
    capture_index = 0

    detector = FaceDetection(model_name, image_size, conf, capture_index)
    detector()
