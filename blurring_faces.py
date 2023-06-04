import copy
import cv2
from ultralytics import YOLO
from tracker import Tracker
import numpy as np


class DetectedObject:
    """
    Reproduce detected object
    """
    def __init__(self, object_id, object_bbox):
        self.id = object_id
        self.bbox = object_bbox


class FaceTracksFinder:
    """
    This class helps to find tracks of faces with identities
    """
    def __init__(self, path_to_video="", path_to_detector="best.pt", detection_threshold=0.5):
        self.path_to_video = path_to_video
        self.path_to_detector_model = path_to_detector
        self.detection_threshold = detection_threshold
        self.video_capture = cv2.VideoCapture(self.path_to_video)
        self.detector = YOLO(self.path_to_detector_model)
        self.tracker = Tracker()
        self.current_frame_number = 0
        self.ret, self.frame = self.video_capture.read()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if not self.ret:
                raise StopIteration
            results = self.detector(self.frame)
            for result in results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, _ = r
                    if score > self.detection_threshold:
                        detections.append([x1, y1, x2, y2, score])

                detected_objects_on_frame = []
                if detections:
                    self.tracker.update(self.frame, detections)
                    if self.tracker.tracks:
                        for track in self.tracker.tracks:
                            bounding_box = track.bounding_box
                            track_id = track.track_id
                            detected_objects_on_frame.append(DetectedObject(track_id, bounding_box))
                self.ret, self.frame = self.video_capture.read()
                self.current_frame_number += 1
                return detected_objects_on_frame

    def get_face_tracks(self):
        """
        :return:
        Returns the full set of tracks not by iterating
        """
        video_capture = cv2.VideoCapture(self.path_to_video)
        ret, frame = video_capture.read()
        model = YOLO(self.path_to_detector_model)
        tracker = Tracker()
        n = 0
        detected_objects = []
        while ret:
            results = model.predict(frame)
            for result in results:
                detections = []
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2, score, _ = r
                    if score > self.detection_threshold:
                        detections.append([x1, y1, x2, y2, score])

                detected_objects_on_frame = []
                if detections:
                    tracker.update(frame, detections)
                    if tracker.tracks:
                        for track in tracker.tracks:
                            bounding_box = track.bounding_box
                            track_id = track.track_id
                            detected_objects_on_frame.append(DetectedObject(track_id, bounding_box))
                detected_objects.append(detected_objects_on_frame)
            n += 1
            ret, frame = video_capture.read()

        video_capture.release()

        return detected_objects


def remove_noise_from_tracks(tracks):
    """
    :param tracks: list
    :return: tracks: list
    Helps to remove some noise
    """
    for frame_index in range(2, len(tracks) - 2):
        if tracks[frame_index] and not tracks[frame_index - 1] and not tracks[frame_index + 1]:
            tracks[frame_index] = []
        elif not tracks[frame_index - 1] and not tracks[frame_index - 2] and tracks[frame_index]:
            tracks[frame_index - 1] = tracks[frame_index - 2] = tracks[frame_index]
    return tracks


def blur_face(image, bbox, core=(25, 25), sigma=10):
    """
    :param image:
    :param bbox:
    :param core:
    :param sigma:
    :return: image: np.ndarray
    This image blurs faces
    """
    if image is None or bbox is None:
        raise Exception("Image and bounding box must not be None")
    x1, y1, x2, y2 = np.abs(bbox.astype(int))
    image_copy = copy.deepcopy(image)
    target_area = image_copy[y1:y2, x1:x2, ::]
    blurred_area = cv2.GaussianBlur(target_area, core, sigma)
    image[y1:y2, x1:x2, ::] = blurred_area
    final_image = image
    return final_image



