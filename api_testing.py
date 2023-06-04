import unittest

from blurring_faces import FaceTracksFinder, remove_noise_from_tracks


class TestDetector(unittest.TestCase):

    def test_without_faces(self):
        face_detector = FaceTracksFinder("no_faces.mp4")
        tracks = []
        for track in face_detector:
            tracks.append(track)
        all_tracks = face_detector.get_face_tracks()
        none_tracks = [[] for d in range(len(tracks))]
        self.assertEqual(all_tracks, none_tracks)
        self.assertEqual(tracks, none_tracks)

    def test_with_full_faces(self):
        face_detector = FaceTracksFinder("full_faces.mp4")
        tracks = []
        for track in face_detector:
            tracks.append(track)
        all_tracks = face_detector.get_face_tracks()
        for i in range(2, len(tracks)):
            assert tracks[i] is not None
        for i in range(2, len(all_tracks)):
            assert tracks[i] is not None

    def test_remove_noise(self):
        face_detector = FaceTracksFinder("full_faces.mp4")
        tracks = []
        for track in face_detector:
            tracks.append(track)
        all_tracks = face_detector.get_face_tracks()
        tracks = remove_noise_from_tracks(tracks)
        all_tracks = remove_noise_from_tracks(all_tracks)
        for i in range(len(tracks)):
            assert tracks[i] is not None
        for i in range(len(all_tracks)):
            assert tracks[i] is not None

