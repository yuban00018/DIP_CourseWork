import cv2 as cv

im = cv.imread('./img/jp.jpg')
print(im)
cv.imshow("Image1", im)
key = cv.waitKey(0) & 0xFF

if key == 27:
    cv.destroyAllWindows()
elif key == ord('s'):
    cv.imwrite("./img/test.jpg",im)
    cv.destroyAllWindows()