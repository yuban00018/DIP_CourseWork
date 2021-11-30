import cv2
import matplotlib.pyplot as plt
import numpy as np

path = "./img/"

man_img = cv2.imread(path+"cameraman.tif",cv2.IMREAD_GRAYSCALE)
# 离散傅里叶变换
dft = cv2.dft(np.float32(man_img),flags=cv2.DFT_COMPLEX_OUTPUT)
# 离散傅里叶变换后如果想让直流分量在输出图像的中心，需要将结果沿两个方向平移
dft_shift = np.fft.fftshift(dft)
# 构建振幅的公式
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

plt.imshow(magnitude_spectrum,cmap="gray")
plt.title("Magnitude Spectrum"),plt.xticks([]),plt.yticks([])
plt.show()
cv2.imshow("origin",man_img)
cv2.waitKey(0)
cv2.destroyAllWindows()