# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     merge_sort.py
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/1/7
   Description :  
==================================================
"""
__author__ = 'sjyttkl'

'''
归并排序是分而治之算法的完美例子。它简单地使用了这种算法的两个主要步骤：
（1）连续划分未排序列表，直到有N个子列表，其中每个子列表有1个“未排序”元素，N是原始数组中的元素数。
（2）重复合并，即一次将两个子列表合并在一起，生成新的排序子列表，直到所有元素完全合并到一个排序数组中。
'''


def merge_sort(arr):
    # The last array split
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    # Perform merge_sort recursively on both halves
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])

    # Merge each side together
    return merge(left, right, arr.copy())


def merge(left,right,merged):
	left_cursor , right_cursor = 0,0
	while left_cursor < len(left) and right_cursor <len(right):
		if left[left_cursor] < right[right_cursor] :
			merged[left_cursor + right_cursor] = left[left_cursor]
			left_cursor += 1
		else:
			merged[left_cursor + right_cursor] = right[right_cursor]
			right_cursor +=1

	for left_cursor in range(left_cursor ,len(left)):
		merged[left_cursor + right_cursor] = left[left_cursor]

	for right_cursor in range(right_cursor,len(right)):
		merged[left_cursor + right_cursor] = right[right_cursor]

	return merged

a = [1,2,1,4,4,2,0,3,2,8,2,4,5]
print(merge_sort(a))