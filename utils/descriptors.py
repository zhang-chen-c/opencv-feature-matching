# -*- coding:utf-8 -*-

import cv2


class CreateDescriptors:
    @staticmethod
    def create_sift_descriptors(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()  # cv2.xfeatures2d.SIFT_create()
        key_points, descriptors = sift.detectAndCompute(image, None)

        return key_points, descriptors

    @staticmethod
    def create_orb_descriptors(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(1000000)
        key_points, descriptors = orb.detectAndCompute(image, None)

        return key_points, descriptors

    @staticmethod
    def create_brisk_descriptors(image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brisk = cv2.BRISK_create()
        key_points, descriptors = brisk.detectAndCompute(image, None)

        return key_points, descriptors
