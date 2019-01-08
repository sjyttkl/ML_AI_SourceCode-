# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     quick_sort.py
   email:         695492835@qq.com
   Author :       sjyttkl
   date：          2019/1/8
   Description :  快速排序
==================================================
"""
__author__ = 'sjyttkl'
"""
快速排序也是一种分而治之的算法，如归并排序。虽然它有点复杂，但在大多数标准实现中，它的执行速度明显快于归并排序，
并且很少达到最坏情况下的复杂度O(n) 。它有三个主要步骤：
（1）我们首先选择一个元素，称为数组的基准元素(pivot)。
（2）将所有小于基准元素的元素移动到基准元素的左侧；将所有大于基准元素的元素移动到基准元素的右侧。这称为分区操作。
（3）递归地将上述两个步骤分别应用于比上一个基准元素值更小和更大的元素的每个子数组。
"""

def partition(array, begin, end):
    pivot_idx = begin
    for i in range(begin + 1, end + 1):
        if array[i] <= array[begin]:
            pivot_idx += 1
            array[i], array[pivot_idx] = array[pivot_idx], array[i]
    array[pivot_idx], array[begin] = array[begin], array[pivot_idx]
    return pivot_idx


def quick_sort_recursion(array, begin, end):
    if begin >= end:
        return
    pivot_idx = partition(array, begin, end)
    quick_sort_recursion(array, begin, pivot_idx - 1)
    quick_sort_recursion(array, pivot_idx + 1, end)


def quick_sort(array, begin=0, end=None):
    if end is None:
        end = len(array) - 1

    return quick_sort_recursion(array, begin, end)

a = [1,2,1,4,4,2,0,3,2,8,2,4,5]
quick_sort(a)
print(a)