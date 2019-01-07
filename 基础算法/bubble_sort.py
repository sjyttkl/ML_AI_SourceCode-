# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     bubble_sort.py
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/1/7
   Description :   冒泡排序
==================================================
"""
__author__ = 'songdongdong'

def bubble_sort(arr):
    def swap(i, j):
        arr[i], arr[j] = arr[j], arr[i]

    n = len(arr)
    swapped = True

    x = -1
    while swapped:
        swapped = False
        x = x + 1
        for i in range(1, n - x):
            if arr[i - 1] > arr[i]:
                swap(i - 1, i)
                swapped = True

    return arr
