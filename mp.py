from mediapipe.python.solutions import pose as mp_pose

mp_pose.PoseLandmark.NOSE
_lmk = mp_pose.PoseLandmark

landmark_mapping = [
    _lmk.NOSE,
    _lmk.NOSE,  # Replace with neck center
    _lmk.RIGHT_SHOULDER,
    _lmk.RIGHT_ELBOW,
    _lmk.RIGHT_WRIST,
    _lmk.LEFT_SHOULDER,
    _lmk.LEFT_ELBOW,
    _lmk.LEFT_WRIST,
    _lmk.RIGHT_HIP,
    _lmk.RIGHT_KNEE,
    _lmk.RIGHT_ANKLE,
    _lmk.LEFT_HIP,  # Replace with hip center
    _lmk.LEFT_HIP,
    _lmk.LEFT_KNEE,
    _lmk.LEFT_ANKLE,
    _lmk.RIGHT_EYE,
    _lmk.LEFT_EYE,
    _lmk.RIGHT_EAR,
    _lmk.LEFT_EAR,
    _lmk.LEFT_FOOT_INDEX,
    _lmk.LEFT_FOOT_INDEX,
    _lmk.LEFT_HEEL,
    _lmk.RIGHT_FOOT_INDEX,
    _lmk.RIGHT_FOOT_INDEX,
    _lmk.RIGHT_HEEL,
]


def convert_to_openpose(results, h, w):
    marks = []
    landmarks = results.pose_landmarks.landmark
    print('the lanmark',results)
    for lmk_index in landmark_mapping:
        lmk = landmarks[lmk_index]
        marks.append(lmk.x * w)
        marks.append(lmk.y * h)
        marks.append(1)

    return marks


class MediaPipeDetector():

    def __init__(self):
        self._pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    def get_pose(self, image):
        h, w, _ = image.shape
        result = self._pose.process(image)
        return result, convert_to_openpose(result, h, w)
