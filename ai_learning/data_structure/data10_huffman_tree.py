#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 5/8/20 10:56 PM
@File    : data10_huffman_tree.py
@Desc    : 霍夫曼树

"""



"""
algorithm(Huffman 树构造算法)
给定n个权值(w1,w2...,wn)作为二叉树n个叶子节点
1）将(w1,w2...,wn)看成有n棵树的森林（每棵树仅有一个结点）
2）在森林中选取两个根节点的权值最小的树合并，作为一颗新的树的左、右子树，且新树的根节点全职为其左右子树根节点权值之和
3）在森林中删除选取的两颗树，并将新树加入森林
4）重复2/3步，知道森林中只剩一棵树为止，该树即为所求的Huffman树
"""




class Node(object):
    """
    节点类
    """
    def __init__(self, name=None, value=None):
        self._name = name
        self._value = value
        self._left = None
        self._right = None


class HuffmanTree(object):

    # 根据huffman思想，以叶子节点为基础，反向建立huffman
    def __init__(self, char_weights):
        self.a = [Node(part[0], part[1]) for part in char_weights]  # 根据输入的字符及其频数生成叶子节点
        while len(self.a) != 1:
            self.a.sort(key=lambda node:node._value, reverse=True)
            c = Node(value=(self.a[-1]._value + self.a[-2]._value))
            c._left = self.a.pop(-1)
            c._right = self.a.pop(-1)
            self.a.append(c)
        self.root = self.a[0]
        self.b = list(range(10))  # 保存每个叶子节点的huffman编码，range的值只需要不小于树的深度就行


    # 递归的思想生成编码
    def pre(self, tree, length):
        node = tree
        if not node:
            return
        elif node._name:
            print(node._name, "的编码为：")
            for i in range(length):
                print(self.b[i])
            print("\n")
            return
        self.b[length] = 0
        self.pre(node._left, length+1)
        self.b[length] = 1
        self.pre(node._right, length+1)

    def get_huffman_code(self):
        self.pre(self.root, 0)


if __name__ == "__main__":
    #输入字符串机频数
    char_weights = [("a", 5), ("b", 7), ("c", 4), ("d", 8), ("f", 15), ("g", 2)]
    tree = HuffmanTree(char_weights)
    tree.get_huffman_code()