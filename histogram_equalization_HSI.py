import cv2
import tools
import numpy as np
path = "./img/"

stream_img = cv2.imread(path+"test.jpg")
stream_hsi = tools.cvt_bgr2hsi(stream_img.copy())
tools.show_hist(stream_hsi[:,:,2],"light before")
stream_hsi[:,:,2] = tools.equalize_hist(stream_hsi[:,:,2].astype(np.uint8))
tools.show_hist(stream_hsi[:,:,2],"hsi after equalize")
stream_dst1 = tools.cvt_hsi2bgr(stream_hsi)

stream_hsv = cv2.cvtColor(stream_img,cv2.COLOR_BGR2HSV)
tools.show_hist(stream_hsv[:,:,2],"value before")
stream_hsv[:,:,2] = tools.equalize_hist(stream_hsv[:,:,2])
tools.show_hist(stream_hsv[:,:,2],"hsv after equalize")
stream_dst2 = cv2.cvtColor(stream_hsv,cv2.COLOR_HSV2BGR)

cv2.imshow("original",stream_img)
cv2.imshow("dst hsi",stream_dst1)
cv2.imshow("dst hsv",stream_dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()