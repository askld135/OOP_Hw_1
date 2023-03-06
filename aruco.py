# python == 3.7
# numpy == 1.20.1
# opencv-contrib-python == 4.5.1.48

import cv2
import cv2.aruco as aruco
import os
import numpy as np

def loadAugImage(path):
    """_summary_

    Args: floder in which all the markers images with ids are stored

    Returns: dictionary with key as the id and values as augment image
    """
    myList = os.listdir(path)
    noOFMarkers = len(myList)
    print("Totla Number of Markers Detected: ", noOFMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
        return augDics

def findArucoMarker(img, markerSize=4,totalMarkers=250, draw=True):             #aruco 마커를 검출
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected =  aruco.detectMarkers(imgGray,
                                                arucoDict,
                                                parameters = arucoParam)
   # print(ids) : 정상적으로 인식하는지 테스트하는데 사용
    if draw:
        aruco.drawDetectedMarkers(img, bboxs)
        
    return [bboxs, ids]
    
def augmentAruco(bbox, id,img, imgAug, drawId = True):

    tl = bbox[0][0][0], bbox[0][0][1]
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]
    
    h, w, c = imgAug.shape
    
    pts1 = np.array([tl, tr, br, bl])
    pts2 = np.float32([[0, 0],[w, 0],[w, h],[0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1],img.shape[0]))
    cv2.fillConvexPoly(img, pts1.astype(int),(0, 0, 0))
    imgOut = img + imgOut
    
    if drawId:
        cv2.putText(imgOut, str(id), tl, cv2.FONT_HERSHEY_PLAIN, 2,
                    (230, 230, 250), 2)
        
    return imgOut
    
def main():
    cap = cv2.VideoCapture(0)
    augDics = loadAugImage("Markers")
    
    while True:
        sccuess, img = cap.read()
        arucoFound = findArucoMarker(img)
        
        # Loop through all the markers and augment each one
        if len(arucoFound[0])!=0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in augDics.keys():
                    img = augmentAruco(bbox, id, img, augDics[int(id)])
        
        cv2.imshow("image",img)
        cv2.waitKey(1)
        

if __name__ == "__main__":
    main()
# https://www.youtube.com/watch?v=v5a7pKSOJd8 36:00부터 이어서 