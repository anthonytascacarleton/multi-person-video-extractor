import cv2
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch.models.mtcnn import MTCNN

FACE_SCORE_THRESHOLD = 0.99

class FaceExtractor:
    def __init__(self, use_gpu=False):
        if use_gpu:
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.detector = MTCNN(keep_all=True, device=device)

    def extract_faces(self, frame, display_results=False):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, scores = self.detector.detect(img)

        if boxes is None:
            return []

        if display_results:
            frame_draw = img.copy()
            draw = ImageDraw.Draw(frame_draw)

        faces = []
        for box, score in zip(boxes, scores):
            if box is None or score < FACE_SCORE_THRESHOLD:
                continue
            if display_results:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                draw.text((box[0], box[1]), str(score))
            faces.append(box)

        if display_results:
            cv2.imshow("frame",  cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return faces