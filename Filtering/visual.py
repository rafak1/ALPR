from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_plate(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))

    # perform a blackhat morphological operation
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    cv2.imshow("Blackhat", blackhat)
    cv2.waitKey()

    light = cv2.morphologyEx(image, cv2.MORPH_CLOSE, squareKernel)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    cv2.imshow("light", light)
    cv2.waitKey()

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
        dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")
    
    cv2.imshow("gradX", gradX)
    cv2.waitKey()


    gradX = cv2.GaussianBlur(gradX, (81, 81), 200)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, squareKernel)
    thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    cv2.imshow("thresh", thresh)
    cv2.waitKey()


    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    cv2.imshow("thresh", thresh)
    cv2.waitKey()


    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)

    cv2.imshow("thresh", thresh)
    cv2.waitKey()

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
    for c in candidates:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        print(ar)
    
        # check to see if the aspect ratio is rectangular
        if ar >= minAR and ar <= maxAR:
            lpCnt = c
            licensePlate = image[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            cv2.imshow("License Plate", licensePlate)
            cv2.waitKey()
            cv2.imshow("ROI", roi)
            cv2.waitKey()
            break
    return (roi, lpCnt)


img = cv2.imread("car.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cnts = find_plate(gray)

roi, lpCnt = locate_license_plate(gray, cnts, minAR=2, maxAR=3)