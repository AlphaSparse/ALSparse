# coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import math

import sys

filename = sys.argv[1]

# 创建自定义字体管理器

df = pd.read_csv(filename)
# df["speed_up"] = df["unroll_hygon"]/df["shuffle_hygon"]
# df["speed_up"] = df["mkl_6226"]/df["mkl_hygon"]
# df["speed_up2"] = df["unroll"]/df["shuffle"]
# df["speed_up3"] = df["unroll"]/df["vgather"]

# df["speed_up"][df["speed_up"] > 5] = 5
step = 0.3
data_min = step * (df["speed_up"].min() // step)
data_max = step * math.ceil(df["speed_up"].max() / step)
# data_max = 20

# 创建一个包含1的边界列表
bins_list = list(np.arange(data_min, 1, step)) + [1] + list(np.arange(1 + step, data_max + step, step))

plt.hist(df["speed_up"], bins=bins_list, edgecolor="black", color="gray")
plt.title("SpGEMM-CSR-f32")

# 在横坐标为1.0的位置绘制一条红色虚线
plt.axvline(x=1.0, color='green', linestyle='--', linewidth=1)
plt.axvline(x=0.8, color='yellow', linestyle='--', linewidth=1)
# plt.axvline(x=1.25, color='violet', linestyle='--', linewidth=1)

# 设置x轴范围
plt.xlim(None, 18)
plt.xlabel("SpeedUp")
plt.ylabel("#. of Matix")
over1 = df["speed_up"][df["speed_up"]>1].count()/df["speed_up"].count() 
over8 = df["speed_up"][df["speed_up"]>=0.8].count()/df["speed_up"].count() 
over25 = df["speed_up"][df["speed_up"]>1.25].count()/df["speed_up"].count() * 100
# plt.text(12.5, 18, "SpeedUp > 1.25 : " + str(round(over25,1)) + "%")
# plt.text(12.5, 17, "SpeedUp > 1.0  : " + str(round(over1,1)) + "%")
# plt.text(12.5, 16, "SpeedUp >= 0.8 : " + str(round(over8,1)) + "%")

text = f"SpeedUp>1.0 {over1:.2%}\nSpeedUp>0.8 {over8:.2%}"
plt.plot([], label=text)
plt.legend(fontsize='x-large', frameon=False, handlelength=0)

plt.savefig("spgemm32.jpg", dpi=300, bbox_inches="tight")

print()
print(df["speed_up"].count())
print("SpeedUp >  1.0 : # ", df["speed_up"][df["speed_up"]>1].count(), " pecentage : ", df["speed_up"][df["speed_up"]>1].count()/df["speed_up"].count())
print("SpeedUp >= 0.8 : # ", df["speed_up"][df["speed_up"]>=0.8].count(), " pecentage : ", df["speed_up"][df["speed_up"]>=0.8].count()/df["speed_up"].count())
print("SpeedUp > 1.25 : # ", df["speed_up"][df["speed_up"]>1.25].count(), " pecentage : ", df["speed_up"][df["speed_up"]>1.25].count()/df["speed_up"].count())
print("SpeedUp < 0.8 : # ", df["speed_up"][df["speed_up"]<0.8].count(), " pecentage : ", df["speed_up"][df["speed_up"]<0.8].count()/df["speed_up"].count())
print(df["speed_up"].mean())

# plt.hist(df["speed_up2"], bins=int(
#     (df["speed_up2"].max()-df["speed_up2"].min())/0.01),
#     edgecolor="black",
#     color="black")
# plt.xlabel("speed up")
# plt.ylabel("matrix num")
# plt.savefig("intel6226_2_unroll.jpg", dpi=300)

# print(df["speed_up2"][df["speed_up2"]>1].count())
# print(df["speed_up2"].count())
# print(df["speed_up2"][df["speed_up2"]>1].count()/df["speed_up2"].count())


# plt.hist(df["speed_up3"], bins=int(
#     (df["speed_up3"].max()-df["speed_up3"].min())/0.01),
#     edgecolor="black",
#     color="black")
# plt.xlabel("speed up", fontsize=14)
# plt.ylabel("matrix num", fontsize=14)
# plt.savefig("intel6226_2_unroll_vgather.jpg", dpi=300)
# plt.tick_params(axis='both', labelsize=12)

# print(df["speed_up3"][df["speed_up3"]>1].count())
# print(df["speed_up3"].count())
# print(df["speed_up3"][df["speed_up3"]>1].count()/df["speed_up3"].count())
