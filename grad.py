import cv2
import time
import tools

path = "./img/"

magic_img = cv2.imread(path+"magic.png", cv2.IMREAD_GRAYSCALE)
plate_img = cv2.imread(path+"plate.png", cv2.IMREAD_GRAYSCALE)
lane_img = cv2.imread(path+"lane.png", cv2.IMREAD_GRAYSCALE)

start = time.time()
magic_sobel_cv2 = cv2.Sobel(magic_img,-1,1,0) + cv2.Sobel(magic_img,-1,0,1)
plate_sobel_cv2 = cv2.Sobel(plate_img,-1,1,0) + cv2.Sobel(plate_img,-1,0,1)
lane_sobel_cv2 = cv2.Sobel(lane_img,-1,1,0) + cv2.Sobel(lane_img,-1,0,1)
end = time.time()
print(end-start)

start = time.time()
magic_sobel_my = tools.sobel(magic_img,1,0) + tools.sobel(magic_img,0,1)
plate_sobel_my = tools.sobel(plate_img,1,0) + tools.sobel(plate_img,0,1)
lane_sobel_my = tools.sobel(lane_img,1,0) + tools.sobel(lane_img,0,1)
end = time.time()
print(end-start)

cv2.imshow("magic original", magic_img)
cv2.imshow("magic cv2 sobel",magic_sobel_cv2)
cv2.imshow("magic my sobel",magic_sobel_my)

cv2.imshow("plate original", plate_img)
cv2.imshow("plate cv2 sobel", plate_sobel_cv2)
cv2.imshow("plate my sobel", plate_sobel_my)

cv2.imshow("lane original", lane_img)
cv2.imshow("lane cv2 sobel", lane_sobel_cv2)
cv2.imshow("lane my sobel", lane_sobel_my)
cv2.waitKey(0)
cv2.destroyAllWindows()