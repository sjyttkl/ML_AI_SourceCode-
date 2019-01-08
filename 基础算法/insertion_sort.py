# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     insertion_sort.py
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/1/7
   Description :  插入排序
==================================================
"""
__author__ = 'sjyttkl'

def insertion_sort(arr,simulation = False):
    for i in range(len(arr)):
        cursor = arr[i]
        pos = i
        while pos > 0 and arr[pos-1] > cursor:
            arr[pos] = arr[pos-1]
            pos = pos-1
        # Break and do the final swap
        arr[pos] = cursor
    return arr

a= insertion_sort([3,4,2,1,5,1,3,5,10])
print(a)