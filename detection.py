class Detection:
    """ Simple container for storing an object detection.
    
    Fields:
    -----------
    bbox: 4x2 numpy array storing the (x,y) values of the 4 corners, counterclockwise
          from top corner
    object_class: str representing the class of object, e.g. 'car'
    score: degree of confidence in this detection.
    """
    def __init__(self, bbox, object_class=None, score=None):
        self.bbox = bbox
        self.object_class = object_class
        self.score = score
        
    def limits(self):
        """ Convenience method, returns [xmin, xmax, ymin, ymax] of the detection"""
        # top left corner gives us xmin and ymax
        xmin = self.bbox[0,0]
        ymax = self.bbox[0,1]
        # bottom right corner gives us xmax and ymin
        xmax = self.bbox[2,0]
        ymin = self.bbox[2,1]
        return [xmin, xmax, ymin, ymax]
    
    def area(self):
        """ Convenience method, returns area of detection"""
        [xmin, xmax, ymin, ymax] = self.limits()
        return (xmax - xmin) * (ymax - ymin)
