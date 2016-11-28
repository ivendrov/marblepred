import numpy as np
import detection

def computeIOU(d1, d2):
    """ Compute the intersection over union (IOU) between two detections """
    b1 = d1.bbox
    b2 = d2.bbox
    [xmin1, xmax1, ymin1, ymax1] = d1.limits()
    [xmin2, xmax2, ymin2, ymax2] = d2.limits()
    # compute size of intersection using the min-max trick
    dx = min(xmax1, xmax2) - max(xmin1, xmin2)
    dy = min(ymax1, ymax2) - max(ymin1, ymin2)
    dx = max(dx, 0)
    dy = max(dy, 0)
    intersect_area = dx * dy
    
    union_area = d1.area() + d2.area() - intersect_area
    return float(intersect_area) / union_area


def readDetectionsFromFile(filepath, **kwargs):
    """ Read a list of detections from a file 
    
    Parameters:
    -----------
    filepath: path to text file containing comma-delimited bounding box coordinates, one box per line
    **kwargs: additional arguments to pass to the detection constructor
    
    Returns:
    ------------
    detections: list of Detection objects with the bounding boxes specified in the file
    """
    coords = np.loadtxt(filepath, delimiter=',')
    detections = []
    for bbox_flat in coords:
        # convert flat list of coordinates into the 4x2 bbox representation
        xs = bbox_flat[0::2]
        ys = bbox_flat[1::2]
        bbox = np.transpose(np.array([xs,ys]))
        detections.append(detection.Detection(bbox, **kwargs))
    return detections
