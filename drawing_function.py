import cv2

img = cv2.imread('./img/jp.jpg', 1)
img = cv2.line(img, (0, 0), (255, 255), (0, 0, 255), 5)
img = cv2.arrowedLine(img, (0, 255), (255, 255), (255, 0, 0), 5)
img = cv2.rectangle(img,(320, 0), (420, 100), (0, 255, 0), 5)
img = cv2.circle(img, (370, 50),50,(255, 0, 0),-1)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img,"OpenCv",(10, 200), font, 4, (0, 255, 255), 10, cv2.LINE_AA)
cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
