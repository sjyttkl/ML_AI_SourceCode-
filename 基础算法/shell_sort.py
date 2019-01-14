# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     shell_sort
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/1/14
   Description :  希尔排序  https://mp.weixin.qq.com/s/vn3KiV-ez79FmbZ36SX9lg
==================================================
"""
__author__ = 'sjyttkl'
def shell_sort(L):
    step = int(len(L) / 2)
    while step > 0:
        for i in range(step, len(L)):  # 在索引为step到len（L）上，比较L[i]和L[i-step]的大小
            while (i >= step and L[i] < L[i - step]):  # 这里可以调整step从小到大或者从大到小排列
                L[i], L[i - step] = L[i - step], L[i]
                i -= step
        step = int(step /2)
    print(L)

L = [1, 3, 2, 32, 5, 4]
shell_sort(L)


