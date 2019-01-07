# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     selection_sort.py
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/1/7
   Description :  选择排序
==================================================
"""
__author__ = 'songdongdong'

def selection_sort(arr):
    for i in range(len(arr)):
        minimum = i

        for j in range(i + 1, len(arr)):
            # Select the smallest value
            if arr[j] < arr[minimum]:
                minimum = j

        # Place it at the front of the
        # sorted end of the array
        arr[minimum], arr[i] = arr[i], arr[minimum]

    return arr
