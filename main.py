import parametrizations
import detection
import predictors
import utils

# Main driver script


def evaluatePredictorOnDetections(predictor, detections, history_steps, predict_steps):
    """ Evaluate a given motion predictor on a trajectory of object detections"""
    
    history = detections[:history_steps]
    predictions = predictor.predict(history, predict_steps=predict_steps)
    truth = detections[history_steps:history_steps+predict_steps]
    IOU_results = []
    for i in range(predict_steps):
        IOU_results.append(utils.computeIOU(predictions[i], truth[i]))
    return IOU_results

def main():
    corner_param = parametrizations.CornerParametrization()
    detections = utils.readDetectionsFromFile('car/groundtruth.txt')
    print "--------------------------------------------------"
    print "Evaluating motion predictor with 1 derivative and smoothing "
    print "IOUs for 20 steps: "
    predictor = predictors.SimpleMotionPredictor(corner_param, num_derivatives=1, smooth_steps=5)
    print evaluatePredictorOnDetections(predictor, detections, 100, 20)	

if __name__ == "__main__":
    main()
