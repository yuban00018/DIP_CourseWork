import cv2
path = './img/'

araras_img = cv2.imread(path+"araras.jpg")

b,g,r = cv2.split(araras_img)

cv2.imshow("original",araras_img)
cv2.imshow("blue",b)
cv2.imshow("green",g)
cv2.imshow("red",r)
cv2.waitKey(0)
cv2.destroyAllWindows()