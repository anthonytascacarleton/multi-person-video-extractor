import cv2
import numpy as np
from random import sample
from extractors import FaceExtractor

class VideoFeatureExtractor:
    def __init__(self, use_gpu=False):
        self.face_extractor = FaceExtractor(use_gpu=use_gpu)

    def get_frames(self, video_file, number_of_frames=-1):
        video_capture = cv2.VideoCapture(video_file)
        if number_of_frames == -1:
            number_of_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        frames = []
        for _ in range(number_of_frames):
            _, image = video_capture.read()
            frames.append(image)
        return frames

    def is_multi_person_video(self, video_file, number_of_frames=12, display_results=False):
        frames = self.get_frames(video_file, number_of_frames)
        if len(frames) < number_of_frames:
            print('VideoFeatureExtractor.is_multi_person_video - not enough frames in {0}'.format(video_file))
            return False
        multi_count = 0
        for frame in frames:
            faces = self.face_extractor.extract_faces(frame, display_results=display_results)
            if len(faces) > 1:
                multi_count += 1
        is_multi_person = multi_count > (number_of_frames / 2)
        return is_multi_person
