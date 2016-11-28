import detection
import numpy as np

# Currently just contains the trivial parametrization
# where each coordinate of each corner is a separate parameter.

# Could also try parametrizing the center + size of the car.

class CornerParametrization:
    def fromDetection(self, det):
        return np.reshape(det.bbox, (-1,))
    def toDetection(self, params):
        return detection.Detection(np.reshape(params, (4,2)))
