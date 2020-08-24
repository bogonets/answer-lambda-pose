import numpy as np

import sys
import os

import pose


def on_init():
    return True


def on_run(keypoints):

    if not keypoints.shape:
        return {'result': None}

    result = pose.xposeOverShoulderByPrediction(keypoints)

    is_poses = [ x['result'] for x in result if x['result'] ]

    if is_poses:
        return {'result': keypoints}
    else:
        return {'result': None}

