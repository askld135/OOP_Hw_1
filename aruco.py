import cv2
import cv2.aruco as aruco
import os
import numpy as np

def findArucoMarker(img, markerSize=4,totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected =  aruco.detectMarkers(imgGray,
                                                arucoDict,
                                                parameters = arucoParam)
    print(ids)
    

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        sccuess, img = cap.read()
        findArucoMarker(img)
        cv2.imshow("image",img)
        cv2.waitKey(1)
        

if __name__ == "__main__":
    main()
