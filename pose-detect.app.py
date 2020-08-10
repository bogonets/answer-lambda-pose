import numpy as np

import sys
import os

import pose


def on_init():
    return True


def on_run(keypoints):

    result = pose.xposeOverShoulderByPrediction(keypoints)

    if result['result']:
        return {'result': keypoints}
    else:
        return {'result': None}

