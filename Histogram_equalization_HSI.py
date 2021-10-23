import cv2
import tools
path = "./img/"

stream_img = cv2.imread(path+"stream.jpg")
stream_hsi = tools.cvt_hsi(stream_img)

cv2.imshow("original",stream_img)
cv2.waitKey(0)
cv2.destroyAllWindows()