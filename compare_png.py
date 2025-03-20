import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取两个png图像
img1 = mpimg.imread('./output/curve/acc_curve_Corr_ep80_dp0.03_20250320033659.png')
img2 = mpimg.imread('./output/curve/acc_curve_no_Corr_ep80_dp0.03_20250320181153.png')

fig, ax = plt.subplots(figsize=(10, 5))

# 在同一子图中显示第一个图像
ax.imshow(img1, alpha=0.5)

# 在同一子图中显示第二个图像，与第一个图像叠加
ax.imshow(img2, alpha=0.5)

# 显示图形
plt.show()