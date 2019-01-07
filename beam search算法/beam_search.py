# -*- coding: utf-8 -*-

"""
==================================================
   File Name：     beam_search
   email:         songdongdong@jd.com
   Author :       songdongdong
   date：          2019/1/4
   Description :
beamsearch 算是一种单模型的集成算法，在decoder端的每一步，不再是单纯的只生成一个token，而是beam_size大小的token，这样会生成beam_size个备选序列
而由beam_size个备选序列，继续向后扩展，会生成beam_size*beam_size个备选序列，对其进行截断，保留概率最大的beam_size备选序列
重复上述过程，直到达到最优结果 或者 达到序列最大长度

==================================================
"""

"""
传统的广度优先策略能够找到最优的路径，但是在搜索空间非常大的情况下，内存占用是指数级增长，很容易造成内存溢出，
因此提出了beam search的算法。 
beam search尝试在广度优先基础上进行进行搜索空间的优化（类似于剪枝）达到减少内存消耗的目的。
"""

#http://jhave.org/algorithms/graphs/beamsearch/beamsearch.shtml
__author__ = 'songdongdong'

from math import log
from numpy import array
from numpy import argmax

# beam search
def beam_search_decoder(data, k):
    sequences = [[list(), 1.0]]
    # walk over each step in sequence
    for row in data:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup :tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences

def greedy_decoder(data):
    # index for largest probability each row
    return [argmax(s) for s in data]

# define a sequence of 10 words over a vocab of 5 words
data = [[0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1]]
data = array(data)
# decode sequence
result = beam_search_decoder(data, 3)
# print result
for seq in result:
    print(seq)
