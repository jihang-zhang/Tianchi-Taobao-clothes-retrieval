import imgaug.augmenters as iaa
from scipy.ndimage import affine_transform
from tqdm import tqdm_notebook as tqdm
import numpy as np

import albumentations as aug
from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import os
import numpy as np
import random
import skimage
import math
#===================================================paug===============================================================
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    original = np.array([[0, 0],
                         [image.shape[1] - 1, 0],
                         [image.shape[1] - 1, image.shape[0] - 1],
                         [0, image.shape[0] - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(original, rect)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    return warped

def Perspective_aug(img,   threshold1 = 0.25, threshold2 = 0.75):
    # img = cv2.imread(img_name)
    rows, cols, ch = img.shape

    x0,y0 = random.randint(0, int(cols * threshold1)), random.randint(0, int(rows * threshold1))
    x1,y1 = random.randint(int(cols * threshold2), cols - 1), random.randint(0, int(rows * threshold1))
    x2,y2 = random.randint(int(cols * threshold2), cols - 1), random.randint(int(rows * threshold2), rows - 1)
    x3,y3 = random.randint(0, int(cols * threshold1)), random.randint(int(rows * threshold2), rows - 1)
    pts = np.float32([(x0,y0),
                      (x1,y1),
                      (x2,y2),
                      (x3,y3)])

    warped = four_point_transform(img, pts)

    x_ = np.asarray([x0, x1, x2, x3])
    y_ = np.asarray([y0, y1, y2, y3])

    min_x = np.min(x_)
    max_x = np.max(x_)
    min_y = np.min(y_)
    max_y = np.max(y_)

    warped = warped[min_y:max_y,min_x:max_x,:]
    return warped

#===================================================origin=============================================================
def aug_image(image):

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),

        iaa.Affine(rotate= (-8, 8),
                   shear = (-8, 8),
                   mode='edge'),

        iaa.SomeOf((0, 2),
                   [
                       iaa.GaussianBlur((0, 0.3)),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                       iaa.AddToHueAndSaturation((-5, 5)),  # change hue and saturation
                       iaa.PiecewiseAffine(scale=(0.01, 0.03)),
                       iaa.PerspectiveTransform(scale=(0.01, 0.1)),
                       iaa.JpegCompression(20, 40)
                   ],
                   random_order=True
                   ),

        iaa.Cutout(nb_iterations=1, size=(0.02, 0.2), squared=False)
    ])

    image = seq.augment_image(image)
    return image

#===================================================crop===============================================================
# def get_cropped_img(image, bbox, is_mask=False):
#     crop_margin = 0.1 # 0.683 ratio

#     size_x = image.shape[1]
#     size_y = image.shape[0]

#     x0, y0, x1, y1 = bbox

#     dx = x1 - x0
#     dy = y1 - y0

#     x0 -= dx * crop_margin
#     x1 += dx * crop_margin + 1
#     y0 -= dy * crop_margin
#     y1 += dy * crop_margin + 1

#     if x0 < 0:
#         x0 = 0
#     if x1 > size_x:
#         x1 = size_x
#     if y0 < 0:
#         y0 = 0
#     if y1 > size_y:
#         y1 = size_y

#     if is_mask:
#         crop = image[int(y0):int(y1), int(x0):int(x1)]
#     else:
#         crop = image[int(y0):int(y1), int(x0):int(x1), :]

#     return crop

def get_cropped_img_fast(image, bbox):
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]

def get_center_aligned_img(crop_img, pt):
    x0, y0 = pt[2]
    x1, y1 = pt[7]

    x = int((x0 + x1) / 2.0)
    w = crop_img.shape[1]
    radius = max(x, w - x)
    s = radius - x
    fill = np.ones([crop_img.shape[0], radius * 2 + 1, 3]).astype(np.uint8) * 128

    if x < radius:
        fill[:, s:s + crop_img.shape[1], :] = crop_img
    else:
        fill[:, 0:crop_img.shape[1], :] = crop_img

    return fill


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def cutout(img, max_height, max_width, min_height=0, min_width=0, fill_value=114):
    img = img.copy()

    height, width = img.shape[:2]

    hole_height = random.randint(min_height, max_height)
    hole_width = random.randint(min_width, max_width)

    y1 = random.randint(0, height - hole_height)
    x1 = random.randint(0, width - hole_width)
    y2 = y1 + hole_height
    x2 = x1 + hole_width

    img[y1:y2, x1:x2] = fill_value
    return img


def random_affine(img, degrees=8, translate=.0625, scale=.1, shear=8, border=0):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Combined rotation matrix
    M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
    if (border != 0) or (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    return img


def aug_medium(prob=1):
    return aug.Compose([
        aug.HorizontalFlip(p=.5),
        aug.OneOf([
            aug.CLAHE(clip_limit=2, p=.5),
            aug.IAASharpen(p=.25),
            ], p=0.35),
        aug.RandomBrightnessContrast(p=.7),
        aug.OneOf([
            aug.GaussNoise(p=.35),
            aug.ISONoise(p=.7),
            aug.ImageCompression(quality_lower=70, quality_upper=100, p=.7)
            ], p=.6),
        aug.RGBShift(p=.5),
        aug.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=.5),
        aug.ToGray(p=.3)
    ], p=prob)