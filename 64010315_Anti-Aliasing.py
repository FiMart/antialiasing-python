import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from skimage import io, transform

# อ่านภาพ
image_path = 'd:/Leetcode/barbara.jpg'
a = io.imread(image_path, as_gray=True)

# ลดขนาดภาพลง 0.25 เท่า (Downsampling) และขยายกลับเป็นขนาดเดิม (Upsampling)
b = transform.rescale(a, 0.25, anti_aliasing=True)
c = transform.rescale(b, 4, anti_aliasing=True)

# สร้างกรองความถี่ต่ำ (Low-pass filter)
H = np.zeros(a.shape)
center = (np.array(a.shape) // 2).astype(int)
H[center[0]-64:center[0]+64, center[1]-64:center[1]+64] = 1

# ใช้ Fourier Transform และกรองความถี่ต่ำ
Da = fftshift(fft2(a))
Dd = Da * H
Dd = fftshift(Dd)
d = np.real(ifft2(Dd))

# แสดงผล
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(a, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(c, cmap='gray')
plt.title("Resized & Restored")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(d, cmap='gray')
plt.title("Anti-Aliasing Image (Fourier)")
plt.axis('off')

plt.show()