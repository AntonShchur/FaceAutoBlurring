import time
from pathlib import Path
import numpy as np
from PyQt6.QtCore import (QThread, pyqtSignal, Qt)
from PyQt6.QtGui import (QImage, QPixmap, QAction)

from PyQt6.QtWidgets import (QWidget, QMainWindow, QGridLayout, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QProgressBar, QComboBox, QFileDialog, QApplication)
import sys

import cv2
from blurring_faces import FaceTracksFinder, blur_face


class MainWindow(QMainWindow):
    """
    Main program class
    """
    WINDOW_SIZE = (800, 600)

    def __init__(self):
        super().__init__()
        self.is_video_loaded = False
        self.is_video_changed = False
        self.video_path = None
        self.face_tracks = []
        self.face_tracker = FaceTracksDetector()
        self.current_frame_number = 0
        self.current_frame = None
        self.frame_width = 0
        self.frame_height = 0
        self.video_player_dims = (800, 800)
        self.selected_ids = []
        self.blur_mode = 0
        self.last_selected = 0
        self.id_to_blur_intensity = dict()
        self.setWindowTitle("FacialAutoBlur")

        # Grid layout
        self.general_layout = QGridLayout(self)
        # Video player layout
        self.video_player_layout = QGridLayout(self)
        # Right buttons layout
        self.right_buttons_layout = QVBoxLayout(self)
        # Slider layout background
        self.slider_background_layout = QHBoxLayout(self)
        # Bottom buttons layout
        self.bottom_buttons_layout = QHBoxLayout(self)


        # Buttons
        self.open_button = QPushButton("OPEN VIDEO")
        self.open_button.setCheckable(True)
        self.open_button.clicked.connect(self.get_path_to_source_video)

        self.start_find_face_tracks = QPushButton("FIND FACES")
        self.start_find_face_tracks.setCheckable(True)
        self.start_find_face_tracks.clicked.connect(self.start_find_faces)

        self.start_pause_video = QPushButton("PLAY/PAUSE")
        self.start_pause_video.setCheckable(True)
        self.start_pause_video.clicked.connect(self.play_pause_video)

        # Next frame button
        self.next_frame = QPushButton(">")
        self.next_frame.setCheckable(True)
        self.next_frame.clicked.connect(self.get_next_frame)
        # Previous frame button
        self.previous_frame = QPushButton("<")
        self.previous_frame.setCheckable(True)
        self.previous_frame.clicked.connect(self.get_previous_frame)
        # Save video button
        self.save_video = QPushButton("SAVE VIDEO")
        self.save_video.setCheckable(True)
        self.save_video.clicked.connect(self.write_video)

        # Video player
        self.video_layout = QLabel()
        self.video_layout.mousePressEvent = self.get_mouse_position
        self.label = QLabel()
        self.label.setText("a")

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)

        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.get_specific_frame)
        # Finding faces progressbar
        self.face_tracks_progressbar = QProgressBar()
        self.face_tracks_progressbar.setMinimum(0)

        # Blurring slider
        self.blurring_slider = QSlider(Qt.Orientation.Horizontal)
        self.blurring_slider.setMinimum(1)
        self.blurring_slider.setMaximum(10)
        self.blurring_slider.valueChanged.connect(self.change_blurring_intensity)

        # Blurring mode combobox
        self.blurring_mode_combobox = QComboBox(self)
        self.blurring_mode_combobox.addItems(["BLUR ALL", "BLUR SELECTED"])
        self.blurring_mode_combobox.currentIndexChanged.connect(self.change_blurring_mode)

        self.general_layout.addLayout(self.video_player_layout, 0, 0)
        self.general_layout.addLayout(self.right_buttons_layout, 0, 1)
        self.general_layout.addLayout(self.slider_background_layout, 1, 0)
        self.general_layout.addLayout(self.bottom_buttons_layout, 2, 0)

        self.video_player_layout.addWidget(self.video_layout)

        self.right_buttons_layout.addWidget(self.open_button)
        self.right_buttons_layout.addWidget(self.start_find_face_tracks)
        self.right_buttons_layout.addWidget(self.face_tracks_progressbar)
        self.right_buttons_layout.addWidget(QLabel("MODE:"))
        self.right_buttons_layout.addWidget(self.blurring_mode_combobox)
        self.right_buttons_layout.addWidget(QLabel("BLUR INTENSITY:"))
        self.right_buttons_layout.addWidget(self.blurring_slider)
        self.right_buttons_layout.addWidget(self.save_video)
        self.right_buttons_layout.addStretch()

        self.slider_background_layout.addWidget(self.slider)

        self.bottom_buttons_layout.addWidget(self.previous_frame)
        self.bottom_buttons_layout.addWidget(self.start_pause_video)
        self.bottom_buttons_layout.addWidget(self.next_frame)

        self.action_open_video = QAction()
        self.action_open_video.setShortcut('Ctrl+O')
        self.action_open_video.setStatusTip("Open video")

        self.layout_widget = QWidget(self)
        self.layout_widget.setLayout(self.general_layout)
        self.setCentralWidget(self.layout_widget)

        self.video_player = VideoPlayerThread()

        self.video_player.ImageUpdate.connect(self.update_pixmap_slot)
        self.video_player.FrameNumberUpdate.connect(self.update_frame_number)
        self.video_player.FrameDimensionsUpdate.connect(self.set_frame_dims)
        self.video_player.MaxFrameUpdate.connect(self.set_max_frame)

        self.video_writer = BlurringFacesThread()
        self.start_image = QImage(np.zeros(self.video_player_dims), *self.video_player_dims, QImage.Format.Format_RGB888)
        self.video_layout.setPixmap(QPixmap.fromImage(self.start_image))

    def change_blurring_mode(self, blur_mode):
        self.blur_mode = blur_mode
        print(self.blur_mode)

    def change_blurring_intensity(self):
        self.id_to_blur_intensity[self.last_selected] = self.sender().value()
        print(self.last_selected)
        print(self.sender().value())

    def get_mouse_position(self, event):

        self.label.setText(f"{self.selected_ids} ")
        if self.face_tracks:
            aspect_ratio_x = self.frame_width / self.video_player_dims[0]
            aspect_ratio_y = self.frame_height / self.video_player_dims[1]

            mouse_x_pos = int(event.pos().x() * aspect_ratio_x)
            mouse_y_pos = int(event.pos().y() * aspect_ratio_y)


            for detection in self.face_tracks[self.current_frame_number]:
                print(detection.id)
                print(detection.bbox)
                x1, y1, x2, y2 = detection.bbox
                if x1 < mouse_x_pos < x2 and y1 < mouse_y_pos < y2:
                    if detection.id not in self.selected_ids:
                        self.selected_ids.append(detection.id)
                        self.last_selected = detection.id
                    else:
                        self.selected_ids.remove(detection.id)
                        self.last_selected = 0

    def write_video(self):
        destination, extension = self.get_path_to_save_video()
        if destination != '':
            self.video_writer.face_tracks = self.face_tracks
            self.video_writer.source_video_path = self.video_path
            self.video_writer.face_to_blur = self.selected_ids
            self.video_writer.save_path = destination
            self.video_writer.extension = extension
            self.video_writer.blur_mode = self.blur_mode
            self.video_writer.video_dimensions = (self.frame_width, self.frame_height)
            self.video_writer.start()

    def set_frame_dims(self, width, height):
        self.frame_width = width
        self.frame_height = height

    def set_max_frame(self, max_frame):
        self.slider.setMaximum(max_frame)
        self.face_tracks_progressbar.setMaximum(max_frame - 1)

    def update_frame_number(self, current_frame_number):

        self.current_frame_number = current_frame_number
        self.update_slider()

    def get_next_frame(self):
        self.current_frame_number += 1
        self.video_player.current_frame += 1
        self.video_player.is_frame_changed = True

    def get_previous_frame(self):
        self.current_frame_number -= 1
        self.video_player.current_frame -= 1
        self.video_player.is_frame_changed = True

    def get_specific_frame(self):
        self.current_frame_number = self.sender().value()
        print(self.sender())
        self.video_player.current_frame = self.current_frame_number
        self.video_player.is_frame_changed = True

    def update_slider(self):
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_frame_number)
        self.slider.blockSignals(False)

    def play_pause_video(self):
        if self.is_video_loaded and self.is_video_changed:
            print("Thread")
            self.video_player.video_path = self.video_path
            self.video_player.start()
            self.is_video_changed = False
        self.video_player.is_video_playing = not self.video_player.is_video_playing

    def set_face_tracks(self, track):
        self.face_tracks.append(track)
        # print(tracks)

    def update_face_finding_progressbar(self, processed_frames):
        self.slider.blockSignals(True)
        self.face_tracks_progressbar.setValue(processed_frames)
        self.slider.blockSignals(False)

    def update_pixmap_slot(self, image, n):

        if self.face_tracks:
            if len(self.face_tracks) > n:
                if self.face_tracks[n]:

                    detections = self.face_tracks[n]
                    for detection in detections:
                        x1, y1, x2, y2 = detection.bbox
                        color = (255, 0, 0)
                        if detection.id in self.id_to_blur_intensity.keys():
                            color = (0, 255, 0)
                        if detection.id in self.id_to_blur_intensity.keys():
                            image = blur_face(image, detection.bbox, sigma=self.id_to_blur_intensity[detection.id])
                        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        image = cv2.putText(image, f"{detection.id}", ((int(x1) + int(x2)) // 2, int(y1) - 10),
                                                                        cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        resized = cv2.resize(image[:, :, ::-1], (800, 800))
        final_image = QImage(resized, 800, 800, QImage.Format.Format_RGB888)
        self.video_layout.setPixmap(QPixmap.fromImage(final_image))

    def get_path_to_source_video(self):
        file_name = QFileDialog.getOpenFileName(self, "Open File",
                                                "C://",
                                                "Videos (*.mp4);;")
        video_path = file_name[0]
        if video_path != '':
            if video_path != self.video_path:
                self.is_video_changed = True
            self.video_path = video_path
            self.is_video_loaded = True

    def get_path_to_save_video(self):
        file_name = QFileDialog.getSaveFileName(self, "Save File", "C://", "Videos (*.mp4);;Videos (*.mkv);;")
        file_extension = file_name[0][-3:]

        return file_name[0], file_extension

    def start_find_faces(self):
        self.face_tracker.video_path = self.video_path

        self.face_tracker.start()
        self.face_tracker.detected_face_tracks.connect(self.set_face_tracks)
        self.face_tracker.CurrentProcessedFrameUpdate.connect(self.update_face_finding_progressbar)


class BlurringFacesThread(QThread):

    def __init__(self):
        super().__init__()
        self.face_tracks = None
        self.face_to_blur = None
        self.source_video_path = None
        self.save_path = None
        self.extension = "mp4v"
        self.blur_mode = 0
        self.video_dimensions = (0, 0)

    def run(self):
        print(self.face_to_blur)
        print(self.save_path)
        self.extension += "v"
        capture = cv2.VideoCapture(self.source_video_path)
        out = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*self.extension), 25, self.video_dimensions)

        ret, frame = capture.read()
        current_frame = 0
        while ret:
            detections = self.face_tracks[current_frame]
            if detections:
                for detection in detections:
                    if self.blur_mode == 0:
                        frame = blur_face(frame, detection.bbox)
                    elif self.blur_mode == 1:
                        if detection.id in self.face_to_blur:
                            frame = blur_face(frame, detection.bbox)
            out.write(frame)
            current_frame += 1
            ret, frame = capture.read()


class VideoPlayerThread(QThread):
    ImageUpdate = pyqtSignal(np.ndarray, int)
    FrameNumberUpdate = pyqtSignal(int)
    FrameDimensionsUpdate = pyqtSignal(int, int)
    MaxFrameUpdate = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.is_running = True
        self.is_video_playing = True
        self.current_frame = 0
        self.max_frames_in_video = 0
        self.is_frame_changed = False
        self.is_frame_updated = False
        self.frame_width = 0
        self.frame_height = 0
        self.frame_time = 0

    def run(self):
        capture = cv2.VideoCapture(self.video_path)
        _, frame = capture.read()

        self.max_frames_in_video = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_time = 1 / capture.get(cv2.CAP_PROP_FPS)
        self.MaxFrameUpdate.emit(self.max_frames_in_video)
        self.FrameDimensionsUpdate.emit(self.frame_width, self.frame_height)

        while self.is_running and 0 <= self.current_frame < self.max_frames_in_video-1:
            start_time = time.time()

            if self.is_frame_changed:
                capture.set(1, self.current_frame)
                self.is_frame_changed = False
                self.is_frame_updated = False
                self.FrameNumberUpdate.emit(self.current_frame)
                _, frame = capture.read()

            if self.is_video_playing:
                _, frame = capture.read()
                img1 = frame

                self.FrameNumberUpdate.emit(self.current_frame)
                self.ImageUpdate.emit(img1, self.current_frame)
                self.current_frame += 1
            else:
                if not self.is_frame_updated:
                    img2 = frame
                    self.ImageUpdate.emit(img2, self.current_frame)
                    self.FrameNumberUpdate.emit(self.current_frame)
                    self.is_frame_updated = True
            current_frame_time = time.time() - start_time
            if current_frame_time < self.frame_time:
                time.sleep(np.abs(self.frame_time - current_frame_time))
            if self.current_frame >= self.max_frames_in_video - 1:
                self.current_frame = self.max_frames_in_video - 1

    def stop(self):
        self.is_running = False
        self.quit()


class FaceTracksDetector(QThread):
    """

    """
    detected_face_tracks = pyqtSignal(list)
    CurrentProcessedFrameUpdate = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.is_active = False

    def run(self):
        self.is_active = True
        self.face_tracks_finder = FaceTracksFinder(self.video_path, "best.pt")
        for detection in self.face_tracks_finder:
            self.detected_face_tracks.emit(detection)
            self.CurrentProcessedFrameUpdate.emit(self.face_tracks_finder.current_frame_number)

    def stop(self):
        self.is_active = False
        self.quit()


if __name__ == "__main__":
    application = QApplication(sys.argv)
    application.setStyleSheet(Path('style.qss').read_text())
    application_main_window = MainWindow()
    application_main_window.show()
    application.exec()