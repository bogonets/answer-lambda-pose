from shapely.geometry import LineString, Point

# fmt: off
COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)
# fmt: on

_KEYPOINT_THRESHOLD = 0.05


def convertDictForCocoKeyPoint(prediction, threshold=_KEYPOINT_THRESHOLD):
    persons = []
    for person in prediction:
        keypoints = {}
        for idx, keypoint in enumerate(person):
            # draw keypoint
            x, y, prob = keypoint
            if prob > threshold:
                keypoint_name = COCO_PERSON_KEYPOINT_NAMES[idx]
                keypoints[keypoint_name] = (x, y)
        persons.append(keypoints)
    return persons


def xposeOverShoulderByPrediction(prediction, threshold=0.5):
    persons = convertDictForCocoKeyPoint(prediction)
    # print(keypoints)
    result = []
    for keypoints in persons:
        if not all(key in keypoints for key in ("left_wrist", "right_wrist", "left_elbow", "right_elbow", "left_shoulder", "right_shoulder")):
            # print("not enough keypoints")
            result.append({"result": False})
        # print("enough keypoints")
        else:
            result.append(xposeOverShoulder(keypoints["left_wrist"], keypoints["right_wrist"],
                                            keypoints["left_elbow"], keypoints["right_elbow"],
                                            keypoints["left_shoulder"], keypoints["right_shoulder"], threshold))
    return result


def xposeOverShoulder(left_wrist, right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder, threshold):
    result = {"result": False}
    if not left_wrist or not right_wrist or not left_elbow or not right_elbow or not left_shoulder or not right_shoulder:
        print("NO!!")
        return result

    import sys
    # sys.stderr.write(f"[pose.xposeOverShoulder] l_s < l_w : {left_shoulder} < {left_wrist}\n")
    # sys.stderr.write(f"[pose.xposeOverShoulder] r_s < r_w : {right_shoulder} < {right_wrist}\n")
    # sys.stderr.write(f"[pose.xposeOverShoulder] l_s < l_e : {left_shoulder} < {left_elbow}\n")
    # sys.stderr.write(f"[pose.xposeOverShoulder] r_s < r_e : {right_shoulder} < {right_elbow}\n")
    # sys.stderr.flush()

    # The wrist's y should be over the shoulder's y.
    if not isUpper(left_wrist[1], left_shoulder[1], right_wrist[1], right_shoulder[1]):
        print("The wrist should be over the shoulder")
        return result

    # The elbow's y should be over the shoulder's y.
    if not isUpper(left_elbow[1], left_shoulder[1], right_elbow[1], right_shoulder[1]):
        print("The elbow should be over the shoulder")
        return result

    # left = LineString([left_wrist, left_elbow])
    # right = LineString([right_wrist, right_elbow])
    # intersection = intersectionOfTwoLineByLineString(left, right)

    # result["left"] = (left.coords[0], left.coords[1])
    # result["right"] = (right.coords[0], right.coords[1])
    # if not intersection:
    #     print("Two lines is not intersect")

    #     extend_left_line = extendLine(left_elbow, left_wrist, 1 + threshold)
    #     extend_right_line = extendLine(right_elbow, right_wrist, 1 + threshold)

    #     extend_left_intersection = intersectionOfTwoLineByLineString(
    #         extend_left_line, right)
    #     extend_right_intersection = intersectionOfTwoLineByLineString(
    #         extend_right_line, left)

    #     extend_both_intersection = intersectionOfTwoLineByLineString(
    #         extend_left_line, extend_right_line)

    #     result["extend_left"] = (
    #         extend_left_line.coords[0], extend_left_line.coords[1])
    #     result["extend_right"] = (
    #         extend_right_line.coords[0], extend_right_line.coords[1])
    #     if extend_both_intersection:
    #         result['result'] = True
    #         return result
    #     elif not extend_left_intersection and not extend_right_intersection:
    #         return result
    #     else:
    #         result['result'] = True
    #         return result

    # # Arm length.
    # temp_left_arm = LineString([intersection, left_elbow])
    # temp_right_arm = LineString([intersection, right_elbow])

    # if left.length < temp_left_arm.length and right.length < temp_right_arm.length:
    #     print("Arms is not cross")
    #     return False

    # # Convert threshold to ratio of position.
    # pos_th_min = threshold / 2
    # pos_th_max = threshold / 2 + threshold

    # # 팔이 교차하는 위치로 제한.
    # pos1 = ratioOfPositionOfLine(left_wrist, left_elbow, intersection)
    # if pos_th_min < pos1 > pos_th_max:
    #     return False

    # pos2 = ratioOfPositionOfLine(right_wrist, right_elbow, intersection)
    # if pos_th_min < pos2 > pos_th_max:
    #     return False
    result['result'] = True
    return result


def isUpper(value1, compare1, value2, compare2):
    return value1 < compare1 and value2 < compare2


def intersectionOfTwoLine(l1pt1, l1pt2, l2pt1, l2pt2):
    line1 = LineString([l1pt1, l1pt2])
    line2 = LineString([l2pt1, l2pt2])
    return intersectionOfTwoLineByLineString(line1, line2)


def intersectionOfTwoLineByLineString(l1, l2):
    result = l1.intersection(l2)
    if not result or isinstance(result, LineString):
        return ()
    return (result.x, result.y)


def ratioOfPositionOfLine(lpt1, lpt2, pt):
    line1 = LineString([lpt1, lpt2])
    line2 = LineString([lpt1, pt])
    return line2.length / line1.length


def extendLine(pt1, pt2, ratio_of_distance):
    extend_pt3 = extrapolateLinear(pt1, pt2, ratio_of_distance)
    return LineString([pt1, extend_pt3])


def extrapolateLinear(pt1, pt2, ratio_of_distance):
    if pt1 == pt2:
        return ()
    x1, y1 = pt1
    x2, y2 = pt2

    dx = (x2 - x1) * ratio_of_distance
    rx = dx + x1

    dy = (y2 - y1) * ratio_of_distance
    ry = dy + y1
    return (rx, ry)

    # rx = (ratio_of_distance * x2) - x1
    # if x2 == 0:
    #     rx =  ratio_of_distance * (-x1) + x1

    # ry = (ratio_of_distance * y2) - y1
    # if y2 == 0:
    #     ry = ratio_of_distance * (-y1) + y1

    # if x1 == x2:
    #     return (x1, ry)
    # elif y1 == y2:
    #     return (rx, y1)
    # else:
    #     print(pt1, pt2, rx, ry, LineString([pt1, pt2]).length, LineString([pt1, (rx, ry)]).length)
    #     return (rx, ry)


if __name__ == "__main__":
    # print(intersectionOfTwoLine((0, 0), (3, 3), (0, 3), (1, 2)))

    line1 = LineString([(0, 0), (3, 3)])
    line2 = LineString([(0, 3), (1, 2)])

    print(extrapolateLinear((-2, 2), (0, 0), 4))
    print(extrapolateLinear((0, 2), (0, 0), 4))
    print(extrapolateLinear((0, 1.5), (0, 0), 4))
    print(extrapolateLinear((0, 1.33), (1, 4), 2))
    print(line1.intersects(line2))
    print(line1.interpolate(1))
