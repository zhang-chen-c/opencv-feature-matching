# -*- coding:utf-8 -*-

import cv2
import argparse
import os
import numpy as np
from utils.descriptors import CreateDescriptors


def feature_matching(template_image,
                     src_image,
                     distance_threshold=0.7,
                     match_threshold=10,
                     homography_threshold=3.0):
    match_mask = None
    rect = None
    center_point = None

    template_key_points, template_descriptors = CreateDescriptors.create_sift_descriptors(template_image)
    src_key_points, src_descriptors = CreateDescriptors.create_sift_descriptors(src_image)

    # index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)  # Locality Sensitive Hashing
    index_params = dict(algorithm=0, trees=5)  # KD Tree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    if (template_descriptors is None) or (src_descriptors is None):
        return match_mask, rect, center_point

    # https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    matches = list(flann.knnMatch(template_descriptors, src_descriptors, k=2))

    good_match = []
    for match in matches:
        if len(match) == 1:
            good_match.append(match)
        else:
            m, n = match
            if m.distance < distance_threshold * n.distance:
                good_match.append(m)

    if len(good_match) > match_threshold:
        src_pts = np.float32([template_key_points[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
        dst_pts = np.float32([src_key_points[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, homography_threshold)
        match_mask = mask.ravel().tolist()

        if matrix is not None:
            h, w, _ = template_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            rect = cv2.perspectiveTransform(pts, matrix)
            center_point = np.float32([w / 2, h / 2]).reshape(-1, 1, 2)
            center_point = cv2.perspectiveTransform(center_point, matrix)

            # draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=match_mask, flags=2)
            # match_image = cv2.drawMatches(cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY),
            #                               template_key_points,
            #                               cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY),
            #                               src_key_points,
            #                               good_match, None, **draw_params)
            # cv2.imshow("match", match_image)
            # while True:
            #     if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            #         break
            # cv2.destroyAllWindows()

    return match_mask, rect, center_point


def visualization(image, rect, point):
    image = cv2.circle(image, (int(point[0][0][0]), int(point[0][0][1])), 5, (255, 255, 0), -1, 0)
    image = cv2.polylines(image, [np.int32(rect)], True, (0, 255, 0), 3, cv2.LINE_AA)
    return image


def start_match(path):
    template_path = os.path.join(path, "template")
    dataset_path = os.path.join(path, "dataset")

    # load template images(RGB)
    template_list = []
    for each_template in os.listdir(template_path):
        each_template_image = cv2.imread(os.path.join(template_path, each_template))
        template_list.append(each_template_image)

    # video_capture = None
    video_capture = cv2.VideoCapture(0)

    if video_capture is None:
        for each in os.listdir(dataset_path):
            image = cv2.imread(os.path.join(dataset_path, each))
            image_vis = cv2.imread(os.path.join(dataset_path, each))
            for each_template in template_list:
                _, rect, center_point = feature_matching(each_template, image)
                if rect is not None:
                    image_vis = visualization(image_vis, rect, center_point)

            cv2.imshow("image", image_vis)
            while True:
                if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
                    break
            cv2.destroyAllWindows()
    else:
        while True:
            ret, frame = video_capture.read()
            if ret is not True:
                continue

            image_vis = frame.copy()
            for each_template in template_list:
                _, rect, center_point = feature_matching(each_template, frame)
                if rect is not None:
                    image_vis = visualization(image_vis, rect, center_point)

            cv2.imshow("video", image_vis)
            if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
                break


def main():
    parse = argparse.ArgumentParser(description="=========opencv-feature-matching=========")
    parse.add_argument("--path", default="./image", type=str, help="The path of image.")
    args = parse.parse_args()

    start_match(args.path)


if __name__ == "__main__":
    main()
