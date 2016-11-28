import numpy as np

class SimpleMotionPredictor:
    """ A class that predicts future motion based on a history of past motion 
    
    Parameters:
    -----------
    parametrization: a Parametrization object that converts between a Detection and an array of scalar parameters
    num_derivatives: number of derivatives to use for prediction (0 means constant, 1 means use velocity, etc)
    smooth_steps: number of steps to use to smooth derivatives for prediction
    """
    def __init__(self, parametrization, num_derivatives=0, smooth_steps=1):
        self.parametrization = parametrization
        self.num_derivatives = num_derivatives
        self.smooth_steps = smooth_steps
    
    def predict(self, past_detections, predict_steps=10):
        """ Predict future detections based on the given history of detections """
        # convert detections to parameters of motion
        past_motions = []
        for det in past_detections:
            past_motions.append(self.parametrization.fromDetection(det))
        past_motions = np.array(past_motions).astype(np.float)
        # estimate all required derivatives via smoothed finite differences
        derivatives = [past_motions[-1]] # 0th derivative is position
        for n in range(1, self.num_derivatives+1):
            # compute finite differences of nth order 
            diffs = np.diff(past_motions,n=n,axis=0) # 0 is time axis
            # smooth by averaging over last smooth_steps timesteps
            derivatives.append(np.mean(diffs[-self.smooth_steps:], axis=0))
            
        # run simulation forward using the derivatives and current state
        predictions = []
        for _ in range(predict_steps):
            # update all derivatives based on the higher ones, in reverse order
            for n in range(self.num_derivatives-1, -1, -1):
                derivatives[n] += derivatives[n+1]
            # 0th derivative is position, so add it to the predictions
            position = derivatives[0]
            predictions.append(self.parametrization.toDetection(np.copy(position)))
        return predictions
