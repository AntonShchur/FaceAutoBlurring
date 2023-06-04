from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Track:
    """
    Reproduce track
    """
    def __init__(self, track_id, bounding_box):
        self.track_id = track_id
        self.bounding_box = bounding_box


class Tracker:
    """
    object tracker
    """
    def __init__(self):

        self.encoder_model_filename = 'model_data\\mars-small128.pb'
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
        self.tracker = DeepSortTracker(self.metric)
        self.encoder = generate_detections.create_box_encoder(self.encoder_model_filename, batch_size=1)
        self.tracks = None

    def update(self, frame, detections):
        if detections:
            bounding_boxes = np.asarray([detection[:-1] for detection in detections])
            bounding_boxes[:, 2:] = bounding_boxes[:, 2:] - bounding_boxes[:, 0:2]
            scores = [detection[-1] for detection in detections]

            features = self.encoder(frame, bounding_boxes)

            face_detections = []
            for bbox_id, bbox in enumerate(bounding_boxes):
                face_detections.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

            self.tracker.predict()
            self.tracker.update(face_detections)
            self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bounding_box = track.to_tlbr()
            track_id = track.track_id
            tracks.append(Track(track_id, bounding_box))

        self.tracks = tracks


