from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_plate(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

    # perform a blackhat morphological operation
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)

    light = cv2.morphologyEx(image, cv2.MORPH_CLOSE, squareKernel)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
        dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")

    gradX = cv2.GaussianBlur(gradX, (81, 81), 200)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, squareKernel)
    thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    # return the list of contours
    return cnts


def locate_license_plate(image, candidates, minAR=4, maxAR=5):
    # initialize the license plate contour and ROI
    lpCnt = None
    roi = None
    # loop over the license plate candidate contours
    print("Candidates: ", len(candidates))
    for c in candidates:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        print("Aspect Ratio: ", ar)
    
        # check to see if the aspect ratio is rectangular
        if ar >= minAR and ar <= maxAR:
            lpCnt = c
            licensePlate = image[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            break
    return (roi, lpCnt)