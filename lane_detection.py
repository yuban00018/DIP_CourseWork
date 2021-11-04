import cv2
import numpy as np
import time

path = './video/'

video1 = cv2.VideoCapture(path+'1.mp4')
video2 = cv2.VideoCapture(path+'2.mp4')


def show_video(title,video):
    success, img = video.read()
    mask = np.zeros((img.shape[0],img.shape[1])).astype(np.uint8)
    triangle = np.array([[img.shape[1]/100*45,img.shape[0]/1.65],
                         [img.shape[1]/100*55,img.shape[0]/1.65],
                         [img.shape[1]/12*11,img.shape[0]],
                         [img.shape[1]/12,img.shape[0]]],dtype=int)
    cv2.fillConvexPoly(mask,triangle, 1)
    while True:
        success, frame = video.read()
        if not success:
            break
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        after_blur = cv2.medianBlur(gray, 5)
        gray = cv2.Canny(after_blur,50,150)
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        lines = cv2.HoughLinesP(gray,rho=1, theta=np.pi/180,threshold=10,minLineLength=90,maxLineGap=150)
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(frame,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
        cv2.imshow(title,frame)
        cv2.imshow(title+" gray",gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


show_video("video1",video1)
show_video("video2",video2)
