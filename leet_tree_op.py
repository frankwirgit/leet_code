import collections
from Queue import Queue
import itertools
from threading import Semaphore #1114
from threading import Event #1114
import bisect #706


class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

#####################################################
# 100. Same binary trees

class solution:
#recursive
    def SameTree(self, t1, t2):
        if t1 and t2:
            return t1.val == t2. val and self.SameTree(t1.left, t2.left) and self.SameTree(t1.rigt, t2.right)
#interative
    def SameTree2(self, t1, t2):
        q = collections.deque([t1, t2])
        while q:
            nd1, nd2 = q.popleft()
            if (not nd1 and nd2) or (nd1 and not nd2):
                return False
            elif (nd1 and nd2):
                if nd1.val != nd2.val:
                    return False
                else:
                    q.append(nd1.left)
                    q.append(nd2.left)
                    q.append(nd1.right)
                    q.append(nd1.right)
        return True
    


#####################################################
# 101. Symmetric binary tree 
#  a binary tree is a mirror of itself

    def SymmetricTree(self, root):
        if not root:
            return True
        return self.dfs(root.left, root.right)
    
    def dfs(self, l, r):
        if l and r:
            return l.val == r.val and self.dfs(l.left, r.right) and self.dfs(l.right, r.left)
        return l == r


#####################################################
# 104. Maximum depth of binary tree
#

#recursive:
    def MaximumDepthBinaryTree(self, root):
        if not root:
            return 0
        else:
            return 1 + max(self.MaximumDepthBinaryTree(root.left), self.MaximumDepthBinaryTree(root.right))

#iterative:
    def MaximumDepthBinaryTree2(self, root):
        q, depth = collections.deque(), 0
        if root:
           q.append(root)
        while len(q):
            depth += 1
            for _ in range(len(q)):
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return depth

#iterative:
    def MaximumDepthBinaryTree3(self, root):
        if not root:
            return 0
        q = collections.deque([root, 1])
        while q:
            node, depth = q.popleft()
            if node:
                if node.left:
                    q.append((node.left, depth+1))
                if node.right:
                    q.append((node.right, depth+1))
        return depth


#####################################################
# 107. Binary tree level order traversal II
# Given a binary tree, return the bottom-up level order traversal of its nodes' values. 
# (ie, from left to right, level by level from leaf to root).

#recursive:
    def BinaryTreeLevelOrderBottomUp(self, root):
        res = []
        self.dfss(root, 0, res)
        return res
    
    def dfss(self, root, level, res):
        if root:
            if len(res) < level + 1: #while stay in the current level
                res.insert(0, [])
            res[-(level + 1)].append(root.val) #note - bottom up, so the index is "-"
            self.dfss(root.left, level+1, res)  #process the next level
            self.dfss(root.right, level+1, res)

#iterative:
    def BinaryTreeLevelOrderBottomUp2(self, root):
        res, stack = [], [(root, 0)]
        while stack:
            node, level = stack.pop()
            if node:
                if len(res) < level + 1:
                    res.insert(0, [])
                res[-(level + 1)].append(node.val)
                stack.append((node.right, level + 1)) #push right in first
                stack.append((node.left, level + 1)) #push left in second
        return res
    


#####################################################
# 108. Convert sorted array to binary search tree
#
# recursive:
    def SortedArraytoBinarySearchTree(self, sa):
        l, r = 0, len(sa)-1
        if l <= r:
            mid = (l+r)//2
            root = TreeNode(sa[mid])
            root.left = self.SortedArraytoBinarySearchTree(sa[:mid])
            root.right = self.SortedArraytoBinarySearchTree(sa[mid+1:])
            return root


#####################################################
# 110. Balanced binary tree
# Given a binary tree, determine if it is height-balanced.
# a height-balanced binary tree is defined as:
# a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

#recursive: 
    def BalancedBinaryTree(self, root):
        if not root:
            return True
        return abs(self.getHeight(root.left)-self.getHeight(root.right)) < 2 and self.BalancedBinaryTree(root.left) and self.BalancedBinaryTree(root.right)

    def getHeight(self, root):
        if not root:
            return 0
        return 1+max(self.getHeight(root.left), self.getHeight(root.right))

#####################################################
# 111. Minimum depth of binary tree
# The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
#

#recursive:
    def MinimumDepthBinaryTree(self, root):
        if not root:
            return 0
        if None in [root.left, root.right]:
            return max(self.MinimumDepthBinaryTree(root.left), self.MinimumDepthBinaryTree(root.right))+1
        else:
            return min(self.MinimumDepthBinaryTree(root.left), self.MinimumDepthBinaryTree(root.right))+1

#iterative:
    def MinimumDepthBinaryTree2(self, root):
        if not root:
            return 0
        q = collections.deque([root, 1])
        while q:
            node, level = q.popleft()
            if node:
                if not node.left and not node.right:
                    return level
                else:
                    q.append((node.left, level+1))
                    q.append((node.right, level+1))


#####################################################
# 112. Path sum
#Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.

    def PathSum(self, root, target):
        if not root:
            return False
        stack = [(root, target)]
        while stack:
            node, ss = stack.pop()
            if not node.left and not node.right and node.val == ss:
                return True
            if node.left:
                stack.append((node.left, ss-node.val))
            if node.right:
                stack.append((node.right, ss-node.val))


#####################################################
# 155. Min stack
# Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
#push(x) -- Push element x onto stack.
#pop() -- Removes the element on top of the stack.
#top() -- Get the top element.
#getMin() -- Retrieve the minimum element in the stack.

class MinStack:
    def __init__(self):
        self.stk = []

    def getMin(self):
        return self.stk[-1][1]

    def top(self):
        return self.stk[-1][0]

    def pop(self):
        self.stk.pop()
        return
    
    def push(self, x):
        if len(self.stk) == 0:
            self.stk.append(x, x)
        else:
            minx = min(x, self.stk[-1][1])
            self.stk.append(x, minx)


        
#####################################################
# 225. Implement stack using queues
#push(x) -- Push element x onto stack.
#pop() -- Removes the element on top of the stack.
#top() -- Get the top element.
#empty() -- Return whether the stack is empty.

class stack():
    def __init__(self):
        self.q1, self.q2 = Queue(), Queue()
        return

    def push(self, x):
        self.q1.put(x)
        return
    
    def pop(self):
        while self.q1.qsize() > 1:
            self.q2.put(self.q1.get())
        if self.q1.qsize == 1:
            self.q1, self.q2 = self.q2, self.q1
            res = self.q2.get()
            return res
    
    def top(self):
        while self.q1.qsize() > 1:
            self.q2.put(self.q1.get())
        if self.q2.qsize == 1:
            res = self.q1.get()
            self.q2.put(res)
            self.q1, self.q2 = self.q2, self.q1
            return res

    def empty(self):
        return not len(self.q1.qsize())

#use the deque

class stack2():
    def __init__(self):
        self.qu = collections.deque()

    def push(self, x):
        q = self.qu
        q.append(x)
        for _ in range(len(q) - 1):
            q.append(q.popleft())

    def pop(self):
        return self.qu.popleft()

    def empty(self):
        return not len(self.qu)

#####################################################
# 226. Invert binary tree
# use recursive

    def InvertBinaryTree(self, root):
        if root:
            root.left, root.right = self.InvertBinaryTree(root.right), self.InvertBinaryTree(root.left)


#####################################################
# 235. Lowest common ancestor of a binary search tree
# Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
# LCA on Wikipedia: 
# "The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants 
# (where we allow a node to be a descendant of itself).”

    def LowestCommonAncestor(self, root, p, q):
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root


#####################################################
# 257. Binary tree paths
# return all root-to-leaf paths
# use dfs

    def BinaryTreePaths(self, root):
        if not root:
            return []
        res = []
        self.dffs(root, "", res)
        return res
    
    def dffs(self, root, path, res):
        if not root.left and not root.right:
            res.append(path + str(root.val))
        if root.left:
            self.dffs(root.left, path + str(root.val) + "->", res)
        if root.right:
            self.dffs(root.right, path + str(root.val) + "->", res)

#####################################################
# 270. Closest binary search tree value
# Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

    def ClosestBinarySearchTreeValue(self, root, target):
        res = root.val
        while root:
            if abs(root.val - target) < abs(res - target):
                res = root.val
            root = root.left if target < root.val else root.right
        return res




#####################################################
# 346. Moving average from data stream
# Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

class MovingAverage():
    def __init__(self, size):
        self.size = size  #note the window size is fixed
        self.queue = []

    def next(self, val):
        if not self.queue or len(self.queue) < self.size:
            self.queue.append(val)
        else:
            self.queue.pop(0)
            self.queue.append(val)
        return float(sum(self.queue)) / len(self.queue)


class MovingAverage2():
    def __init__z(self, size):
        self.size = size
        self.queue = collections.deque()

    def next(self, val):
        self.queue.append(val)
        if len(self.queue) > self.size:
                self.queue.popleft()
        return sum(self.queue)*1. / len(self.queue)


#####################################################
# 359. Logger rate limiter
#Design a logger system that receive stream of messages along with its timestamps, each message should be printed if and only if it is not printed in the last 10 seconds.
#Given a message and a timestamp (in seconds granularity), return true if the message should be printed in the given timestamp, otherwise returns false.
#It is possible that several messages arrive roughly at the same time.

class Logger():
    timeStoreLen = 10
    def __init__(self):
        self.timeToMessages = collections.defaultdict(set) # maps a time to set of messages printed at that time

    def shouldPrintMessage(self, timestamp, message):
        oldTimes = list( self.timeToMessages.keys() )

        # remove timestamps which are expired
        for oldTime in oldTimes:
            if timestamp - oldTime >= self.timeStoreLen:
                del self.timeToMessages[ oldTime ]

        # printed the same message recently?
        for oldTime in self.timeToMessages: #for oldTime in oldTimes ?
            if message in self.timeToMessages[oldTime]:
                return False

        self.timeToMessages[ timestamp ].add( message )
        return True


class Logger2():

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = dict()
        

    def shouldPrintMessage(self, timestamp: 'int', message: 'str') -> 'bool':
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity.
        """
        if message not in self.data or timestamp - self.data[message] >= 10:
            self.data[message] = timestamp
            return True
        else:
            return False


#####################################################
# 404. Sum of left leaves
# Find the sum of all left leaves in a given binary tree.

#recursive
    def SumOfLeftLeaves(self, root):
        def DFs(root, left):
            if not root:
                return
            if left and not root.left and not root.right:
                self.res += root.val
            DFs(root.left, True)
            DFs(root.right, False)
        
        self.res = 0
        DFs(root, False)
        return self.res



#####################################################
# 429. N-ary tree level ordertraversal
    def NaryTreeLevelOrder(self, root):
        q, res = [root], []
        while any(q):
            res.append([node.val for node in q])
            #q = [child for node in q for child in node.children if child]
            q = [child for child in node.children for node in q if child]
        return res

    def NaryTreeLevelOrder2(self, root):
        if not root:
            return []
        queue, res = [root], []
        while queue:
            tmp, tmpq = [], []
            for node in queue:
                tmp.append(node.val)
                if node.children:
                    for child in node.children:
                        tmpq.append(child)
            res.append(tmp)
            queue = tmpq
        return res


#####################################################
# 530. Minimum absolute difference in BST
# Given a binary search tree with non-negative values, find the minimum absolute difference between values of any two nodes.

#inorder traversal (the resulted list is sorted in asce order)

    def GetMinimumDifferenceBST(self, root):
        res = []
        def Dfs(node):
            if node.left:
                Dfs(node.left)
            res.append(node.val)
            if node.right:
                Dfs(node.right)
        Dfs(root)
        return min(abs(a-b) for a, b in zip(res, res[1:]))



#####################################################
# 538. Convert BST to greater tree
# Given a Binary Search Tree (BST), convert it to a Greater Tree such that every key of the original BST is changed to the original key plus sum of all keys greater than the original key in BST.

#recursive 
    def ConvertBSTGreaterTree(self, root):
        self.cur_sum = 0
        self.convertHelper(root)
        return root

    def convertHelper(self, root):
        if not root:
            return 
        self.convertHelper(root.right)
        root.val += self.cur_sum
        self.cur_sum = root.val
        self.convertHelper(root.left)

    def ConvertBSTGreaterTree2(self, root):
        def dFs(node, res):
            if node:
                dFs(node.right, res)
                node.val = node.val + res
                dFs(node.left, node.val)
            res = 0
            return dFs(root, res)

#recursive

    def ConvertBSTGreaterTree3(self, root):
        if not root:
            return root
        prev, stack = 0, [(False, root)]
        while stack:
            visited, node = stack.pop()
            if visited:
                node.val += prev
                prev = node.val
            else:
                if node.left:
                    stack.append((False, node.left))
                stack.append((True, node))
                if node.right:
                    stack.append((False, node.right))
        return root


#####################################################
# 543. Diameter of binary tree
# Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
    def DiameterOfBinaryTree(self, root):
        self.res = 0
        def DFfs(node):
            if not node:
                return 0
            left, right = DFfs(node.left), DFfs(node.right)
            self.res = max(self.res, left + right)
            return 1 + max(left, right)

        DFfs(root)
        return self.res
    
    


#####################################################
# 559. Maximum depth of N-ary tree

#recursive:

    def MaxDepthNaryTree(self, root):
        def DFSS(root):
            if not root:
                return 0
            res = 0
            for child in root.children:
                res = max(DFSS(child), res)
            return res + 1
        return DFSS(root)


    #define for a node

class NaryTreeNode:
    def __intit__(self, val, children):
        self.val = val
        self.children = children
    
    def MaxDepthNaryTree1(self, root):
        return 1 + max([self.MaxDepthNaryTree1(child) for child in root.children], depfault=0) if root else 0

    def MaxDepthNaryTree2(self, root):
        q, level = root and [root], 0
        while q:
            q, level = [ child for node in q for child in node.children if child ], level + 1
        return level

    def MaxDepthNaryTree3(self, root, level=1):
        return max(root and [self.MaxDepthNaryTree3(child, level+1) for child in root.children] + [level] or [0])


#####################################################
# 563. Binary tree tilt
#Given a binary tree, return the tilt of the whole tree.
#The tilt of a tree node is defined as the absolute difference between the sum of all left subtree node values and the sum of all right subtree node values. Null node has tilt 0.
#The tilt of the whole tree is defined as the sum of all nodes' tilt.

#recursive:

    def FindTilt(self, root):
        def tilt(root):
            if not root:
                return (0, 0)  # return (sum, tilt) of tree
            left = tilt(root.left)
            right = tilt(root.right)
            return (left[0] + right[0] + root.val, abs(left[0] - right[0]) + left[1] + right[1])
        return tilt(root)[1]

    def FindTilt2(self, root):
        def Ddfs(root):
            if not root:
                return 0
            left = Ddfs(root.left)
            right = Ddfs(root.right)
            self.tilt += abs(left-right)
            return root.val + left + right
        
        self.tilt = 0
        Ddfs(root)
        return self.tilt

    def FindTilt3(self, root):
        self.tilt = 0
        def _sum(node):
            if not node:
                return 0
            left, right = _sum(node.left), _sum(node.right)
            self.tilt += abs(left - right)
            return node.val + left + right
        _sum(root)
        return self.tilt




#####################################################
# 589. N-ary tree preorder traversal

#recursive:
    def NaryTreePreorder(self, root):
        if not root:
            return []
        res = []
        res.append(root.val)
        for s in root.children:
            res += self.NaryTreePreorder(s)
        return res

#iterative - dfs:
    def NaryTreePreorder2(self, root):
        if not root:
            return []
        res = []
        stack = [root]
        while stack:
            u = stack.pop()
            res.append(u.val)
            if u.children:
                for c in u.children[::-1]: #reverse - put right -> left, then pop left first
                    stack.append(c)
        return res

#####################################################
# 590. N-ary tree postorder traversal

#recursive
    def NaryTreePostorder(self, root):
        def ddffs(root):
            if not root:
                return
            for c in root.children:
                ddffs(c)
            res.append(root.val)
        res = []
        ddffs(root)
        return res


#iterative
    def NaryTreePostorder2(self, root):
        res, stack = [], root and [root]
        while stack:
            node = stack.pop()
            res.append(node.val) #root first
            stack += [ child for child in node.children if child ]
        return res[::-1] #stack has root pushed in first, right children above, then left children above, so reverse the result

    def NaryTreePostorder3(self, root):
        if not root:
            return []
        res = collections.deque()  #use deque appendleft()
        stack = [root]
        while stack:
            u = stack.pop()
            res.appendleft(u.val)  #append left root first, then right children, then left children
            for c in u.children:
                stack.append(c)
        return list(res)  #list order - left, right children, then root

#####################################################
# 606. Construct string from binary tree
# You need to construct a string consists of parenthesis and integers from a binary tree with the preorder traversing way.
# The null node needs to be represented by empty parenthesis pair "()". And you need to omit all the empty parenthesis pairs that don't affect the one-to-one mapping relationship between the string and the original binary tree.

# recursive
#If the tree is empty, we return an empty string.
#We record each child as '(' + (string of child) + ')'
#If there is a right child but no left child, we still need to record '()' instead of empty string.

    def TreeToString(self, root):
        if not root:
            return ""
        left = "({})".format(self.TreeToString(root.left)) if (root.left or root.right) else ""
        right = "({})".format(self.TreeToString(root.right)) if root.right else ""
        return "{}{}{}".format(root.val, left, right)

    def TreeToString2(self, root):
        if not root:
            return ""
        res = str(root.val)  #preorder tranvesal
        if root.left:
            res += "(" + self.TreeToString2(root.left) + ")"
            if root.right:
                res += "(" + self.TreeToString2(root.right) + ")"
        elif root.right:
            res += "()" + "(" + self.TreeToString2(root.right) + ")"
        return res
    
    def TreeToString3(self, root):
        if not root:
            return ""
        res = ""
        left = self.TreeToString3(root.left)
        right = self.TreeToString3(root.right)
        if left or right:
            res += "(%s)" % left
        if right:
            res += "(%s)" % right
        return str(root.val) + res

    def TreeToString4(self, root):
        def preorder(root):
            if not root:
                return ""
            s, l, r = str(root.val), preorder(root.left), preorder(root.right)
            if r == "" and l == "":
                return s
            elif l == "":
                s += "()" + "(" + r + ")"
            elif r == "":
                s += "(" + l + ")"
            else:
                s += "(" + l + ")" + "(" + r + ")"
            return s
        return preorder(root)

    

    















#####################################################
# 617. Merge two binary trees
#Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.
#You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

#recursive

    def MergeTrees(self, r1, r2):
        if not r1 and not r2:
            return None
        ans = TreeNode((r1.val if r1 else 0) + (r2.val if r2 else 0))
        ans.left = self.MergeTrees(r1 and r1.left, r2 and r2.left)
        ans.right = self.MergeTrees(r1 and r1.right, r2 and r2.right)
        return ans
    
    def MergeTrees2(self, r1, r2):
        if r1 and r2:
            r1.val += r2.val
            r1.left = self.MergeTrees2(r1.left, r2.left)
            r1.right = self.MergeTrees2(r1.right, r2.right)
            return r1
        else:
            return r1 or r2
    
#iterative

    def MergeTrees3(self, r1, r2):
        if r1 is None and r2 is None:
            return r1 or r2
        else:
            root = r1
            stack = [(r1, r2)]
            while stack:
                r1, r2 = stack.pop()
                r1.val += r2.val
                if r1.left is None or r2.left is None:
                    r1.left = r1.left or r2.left
                else:
                    stack.append((r1.left, r2.left))
                if r1.right is None or r2.right is None:
                    r1.right = r1.right or r2.right
                else:
                    stack.append((r1.right, r2.right))
            return root


#####################################################
# 637. Average of levels in binary tree
# Given a non-empty binary tree, return the average value of the nodes on each level in the form of an array.

#recursive 
    def AverageOfLevels(self, root):
        res = []
        def ddffss(node, depth=0):
            if node:
                if len(res) <= depth:
                    res.append([0, 0])  # append another level, with value and number of nodes on the same level
                res[depth][0] += node.val
                res[depth][1] += 1
                ddffss(node.left, depth+1)
                ddffss(node.right, depth+1)
            ddffss(root)
            return [ val*1./numbs for val, numbs in res]

#iterative
    def AverageOfLevels2(self, root):
        stack = [(root, 1)] if root else []
        total = collections.defaultdict(int)
        count = collections.defaultdict(int)
        while stack:
            node, level = stack.pop()
            total[level] += node.val
            count[level] += 1
            if node.left:
                stack.append((node.left, level+1))
            if node.right:
                stack.append((node.right, level+1))
        return [1.0*total[level]/count[level] for level in sorted(total.keys())]

    def AverageOfLevels3(self, root):
        res = []
        if not root:
            return res
        q = [root]
        while q:
            q1, total, cnt = [], 0, 0
            while q:
                    node = q.pop()
                    if node.left: q1.append(node.left)
                    if node.right: q1.append(node.right)
                    total += node.val
                    cnt += 1
            res.append(total*1.0/cnt)
            q = list(q1)
        return res

#####################################################
# 653. Two sum IV - input is a BST
# Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that their sum is equal to the given target.

#recursive - inorder travesal
#then use binary search
    def FindTargetBST2(self, root, target):
        def inorderTrave(root): #then the output is a sorted array with node.val
            if not root:
                return []
            elif not root.left and not root.right:
                return [root.val]
            else:
                return inorderTrave(root.left) + [root.val] + inorderTrave(root.right)
        res = inorderTrave(root)
        l, r = 0, len(res)-1
        while l < r:
            if res[l] + res[r] == target:
                return True
            elif res[l] + res[r] < target:
                l += 1
            else:
                r -= 1
        return False

#iterative

    def FindTargetBST(self, root, target):
        if not root:
            return False
        queue, res = [root], []
        for node in queue:
            if target - node.val in res:
                return True
            res.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            return False



#####################################################
# 669. Trim a binary search tree
# Given a binary search tree and the lowest and highest boundaries as L and R, trim the tree so that all its elements lies in [L, R] (R >= L). You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.

#recursive:

    def TrimBST(self, root, L, R):
        def trim(node):
            if node:
                if node.val > R:
                    return trim(node.left)
                elif node.val < L:
                    return trim(node.right)
                else:
                    node.left = trim(node.left)
                    node.right = trim(node.right)
                    return node
        return trim(root)

    def TrimBST2(self, root, L, R):
        def trim(node):
            if not node:
                return None
            node.left, node.right = trim(node.left), trim(node.right)
            if not (L <= node.val <= R):
                node = node.left if node.left else node.right
            return node
        return trim(root)

    def TrimBST3(self, root, L, R):
        if root:
            if L <= root.val <= R:
                root.left = self.TrimBST3(root.left, L, R)
                root.right = self.TrimBST3(root.right, L, R)
                return root
            if root.val < L:
                return self.TrimBST3(root.right, L, R)
            elif root.val > R:
                return self.TrimBST3(root.left, L, R)
        return None

#iterative:
    def TrimBST4(self, root, L, R):
        if not root:
            return None
        stack = [root]
        while stack:
            node = stack.pop()
            if node.val > R:
                if node.left:
                    node = node.left
                    stack.append(node)
            elif node.val < L:
                if node.right:
                    node = node.right
                    stack.append(node)
            else: 
                stack.append(node.left)
                stack.append(node.right)
        return root

            

#####################################################
# 700. Search in a binary search tree
    def searchBST(self, root, val):
        if root is None:
            return []
        tmp = root
        while tmp:
            if tmp.val == val:
                return tmp
            elif tmp.val < val:
                tmp = tmp.right
            else:
                tmp = tmp.left
        
    
    def searchBST2(self, root, val):
        if root is None:
            return []
        if root.val < val:
            return self.searchBST2(root.right, val)
        elif root.val > val:
            return self.searchBST2(root.left, val)
        return root




#####################################################
# 705. Design hashset 

#Design a HashSet without using any built-in hash table libraries.

#To be specific, your design should include these functions:

#add(value): Insert a value into the HashSet. 
#contains(value) : Return whether the value exists in the HashSet or not.
#remove(value): Remove a value in the HashSet. If the value does not exist in the HashSet, do nothing.

class ListNodee():
    def __init__(self,key,next):
        self.key, self.next =key, next
class MyHashSet():
    def __init__(self):
        self.size, self.bucket=10000,[None]*10000
    
    def add(self,key):
        idx = key%self.size
        cur = self.bucket[idx]
        if not cur:
            self.bucket[idx]=ListNodee(key,None)
            return
        while cur:
            if cur.key==key: return
            if not cur.next: 
                cur.next=ListNodee(key,None)
                return
            if cur.next:
                cur=cur.next
        return
    

    def remove(self,key):
        idx = key%self.size
        cur = prev = self.bucket[idx]
        if not cur: return
        if cur.key == key:
            self.bucket[idx]=cur.next
        else:
            cur = cur.next
        while cur:
            if cur.key==key:
                prev.next = cur.next
                return
            else:
                cur, prev = cur.next, prev.next
        return
    
    def contain(self,key):
        idx = key%self.size
        cur = self.bucket[idx]
        while cur:
            if cur.key==key:
                return True
            cur=cur.next
        return False

#####################################################
# 706. Design hashmap
# Design a HashMap without using any built-in hash table libraries.
# To be specific, your design should include these functions:

#put(key, value) : Insert a (key, value) pair into the HashMap. If the value already exists in the HashMap, update the value.
#get(key): Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
#remove(key) : Remove the mapping for the value key if this map contains the mapping for the key.

#Example:

#MyHashMap hashMap = new MyHashMap();
#hashMap.put(1, 1);          
#hashMap.put(2, 2);         
#hashMap.get(1);            // returns 1
#hashMap.get(3);            // returns -1 (not found)
#hashMap.put(2, 1);          // update the existing value
#hashMap.get(2);            // returns 1 
#hashMap.remove(2);          // remove the mapping for 2
#hashMap.get(2);            // returns -1 (not found) 

# using just arrays, direct access table
# using linked list for chaining
class ListNode:
    def __init__(self, key, val):
        self.pair = (key, val)
        self.next = None

class MyHashMap:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.m = 1000
        self.h = [None]*self.m

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: void
        """
        index = key % self.m
        if self.h[index] == None:
            self.h[index] = ListNode(key, value)
        else:
            cur = self.h[index]
            while True:
                if cur.pair[0] == key:
                    cur.pair = (key, value) #update
                    return
                if cur.next == None:
                    break
                cur = cur.next
            cur.next = ListNode(key, value)
    
    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        index = key % self.m
        cur = self.h[index]
        while cur:
            if cur.pair[0] == key:
                return cur.pair[1]
            else:
                cur = cur.next
        return -1

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: void
        """
        index = key % self.m
        cur = prev = self.h[index]
        if not cur:
            return
        if cur.pair[0] == key:
            self.h[index] = cur.next
        else:
            cur = cur.next
            while cur:
                if cur.pair[0] == key:
                    prev.next = cur.next
                    break
                else:
                    cur, prev = cur.next, prev.next
        
#Python straightforward solution using one/two lists
class MyHashMap2(object):

    def __init__(self):
        self.items = [0] * 1000000
        
    def put(self, key, value):
        self.items[key] = value + key + 1
        
    def get(self, key):
        if self.items[key]:
            return self.items[key] - key - 1
        else:
            return -1
        
    def remove(self, key):
        self.items[key] = 0

#It has been pointed out the average search, insert and delete functions shall be O(1) for Hashtable. So the below implementation is not strictly hashtable by definition. Though the question itself does not explicitly state the time complexity requirement and it gets passed for all test cases

class MyHashMap3(object):

    def __init__(self):
        self.keys = []
        self.values = []
        
    def put(self, key, value):
        if key in self.keys:
            self.values[self.keys.index(key)] = value
        else:
            self.keys.append(key)
            self.values.append(value)
        
    def get(self, key):
        if key in self.keys:
            return self.values[self.keys.index(key)]
        else:
            return -1
        
    def remove(self, key):
        if key in self.keys:
            del self.values[self.keys.index(key)]
            del self.keys[self.keys.index(key)] 


class MyHashMap4(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dic = []
        

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: void
        """
        i = bisect.bisect_left(self.dic, (key,))
        if i==len(self.dic) or key != self.dic[i][0]:
            self.dic.insert(i,(key, value))
        else: 
            self.dic[i] = (key, value)            
            

    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        i = bisect.bisect_left(self.dic, (key,))
        if i==len(self.dic) or key != self.dic[i][0]:
            return -1
        else:
            return self.dic[i][1]

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: void
        """
        i = bisect.bisect_left(self.dic, (key,))
        if i==len(self.dic) or key != self.dic[i][0]:
            pass
        else:
            self.dic.pop(i)


### a easy and better one
class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None

        
class MyHashMap5(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.map = [None] * 10000

    def put(self, key: int, value: int) -> None:
        """
        value will always be non-negative.
        """
        hash_index = hash(key) % 10000
        if self.map[hash_index] is None:
            self.map[hash_index] = Node(key, value)
        else:
            previous = None
            start = self.map[hash_index]
            while start is not None:
                if start.key == key:
                    start.value = value
                    return
                previous = start
                start = start.next
                
            previous.next = Node(key, value)
        
    def get(self, key: int) -> int:
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        """
        hash_index = hash(key) % 10000
        start = self.map[hash_index]
        while start is not None:
            if start.key == key:
                return start.value
            start = start.next
        return -1
         

    def remove(self, key: int) -> None:
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        """
        hash_index = hash(key) % 10000
        previous = None
        start = self.map[hash_index]
        while start is not None:
            if start.key == key:
                if previous is None:
                    self.map[hash_index] = start.next
                else:
                    previous.next = start.next
                return
            previous = start
            start = start.next

#####################################################
# 783. Minimum distance between BST nodes
# Given a Binary Search Tree (BST) with the root node root, return the minimum difference between the values of any two different nodes in the tree.
# The same solution for problem #530 Minimum absolute difference in BST
    def MinimumDifferenceBST(self, root):
        self.pre = -float('inf')
        self.res = float('inf')
        if not root:
            return None
        if root.left:
            self.MinimumDifferenceBST(root.left)
        self.res = min(self.res, root.val - self.pre)
        self.pre = root.val
        if root.right:
            self.MinimumDifferenceBST(root.right)
        return self.res

        


#####################################################
# 872. Leaf-smiliar trees
# Consider all the leaves of a binary tree.  From left to right order, the values of those leaves form a leaf value sequence.
# Two binary trees are considered leaf-similar if their leaf value sequence is the same.

    def LeafSimilar(self, root1, root2):
        def DFFS(node):
            if not node: 
                return
            if not node.left and not node.right:
                yield node.val
            for i in DFFS(node.left):
                yield i
            for i in DFFS(node.right):
                yield i
        return all(a == b for a, b in itertools.zip_longest(DFFS(root1), DFFS(root2)))

    
    def LeafSimilar2(self, root1, root2):
        def DDFS(node, res):
            if node:
                if not node.left and not node.right:
                    res += [node.val]
                DDFS(node.left, res)
                DDFS(node.right, res)
                return res
        return DDFS(root1, []) == DDFS(root2, []) 

    def LeafSimilar3(self, root1, root2):
        def getLeaves(leaves, root):
            if root:
                if not root.left and not root.right:
                    leaves.append(root.val)
                getLeaves(leaves, root.left)
                getLeaves(leaves, root.right)
                
        leaves_1, leaves_2 = [],  []
        getLeaves(leaves_1, root1)
        getLeaves(leaves_2, root2)
        return leaves_1 == leaves_2


            



#####################################################
# 897. Increasing order search tree
# Given a binary search tree, rearrange the tree in in-order so that the leftmost node in the tree is now the root of the tree, and every node has no left child and only 1 right child.
#Straigh forward idea:
#res = inorder(root.left) + root + inorder(root.right)

#pass a tail part to the function, so it can link it to the last node.
#This operation takes O(1), instead of O(N). Otherwise the whole time complexity will be O(N^2).
#Also, remember to set root.left = null.
#Otherwise it will be TLE for Leetcode to traverse your tree.


    def IncreasingBST(self, root, tail=None):
        if not root:
            return tail
        res = self.IncreasingBST(root.left, root)
        root.left = None
        root.right = self.IncreasingBST(root.right, tail)
        return res





#####################################################
# 933. Number of recent calls
# write a class to count recent requests
# It has only one method: ping(int t), where t represents some time in milliseconds.
# Return the number of pings that have been made from 3000 milliseconds ago until now.
# Any ping with time in [t - 3000, t] will count, including the current ping.
# It is guaranteed that every call to ping uses a strictly larger value of t than before.

# Example 1:
# Input: inputs = ["RecentCounter","ping","ping","ping","ping"], inputs = [[],[1],[100],[3001],[3002]]
# Output: [null,1,2,3,3]

class RecentCounter():
    def __init__(self):
        self.p = collections.deque()
        
    def ping(self, t):
        self.p.append(t)
        while self.p[0] < t-3000:  #expire those ping older than 3000 ms
            self.p.popleft() 
        return len(self.p)









#####################################################
# 938. Range sum of BST
# Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

#recursive

    def RangeSumBST(self, root, L, R):
        if not root:
            return 0
        l = self.RangeSumBST(root.left, L, R)
        r = self.RangeSumBST(root.right, L, R)
        return l + r + (L <= root.val <= R)*root.val


    def RangeSumBST2(self, root, L, R):
        return self.inorder(root, 0, L, R)

    def inorder(self, root, value, L, R):
        if root:
            value = self.inorder(root.left, value, L, R)
            if root.val >=L and root.val <= R:
                value += root.val
            value = self.inorder(root.right, value, L, R)
        return value


    def RangeSumBST3(self, root, L, R):
        def ddfs(root):
            if not root:
                return 
            if L <= root.val <= R:
                self.res += root.val
            if L <= root.val:
                ddfs(root.left)
            if R >= root.val:
                ddfs(root.right)

        self.res = 0
        ddfs(root)
        return self.res

#iterative
    def RangeSumBST4(self, root, L, R):
        stack = [root]
        res = 0
        while stack:
            u = stack.pop()
            if L <= u.val <= R:
                res += u.val
            if u.left and u.val >= L:
                stack.append(u.left)
            if u.right and u.val <= R:
                stack.append(u.right)
        return res


#####################################################
# 965. Univalued binary tree
# A binary tree is univalued if every node in the tree has the same value.
# Return true if and only if the given tree is univalued

#recursive:

    def IsUnivaluedTree(self, root):
        def DFS(node):
            return not node or node.val == root.val and DFS(node.left) and DFS(node.right)
        return DFS(root)
    
    def IsUnivaluedTree2(self, root):
        if not root:
            return True
        if root.left and root.left.val != root.val:
            return False
        if root.right and root.right.val != root.val:
            return False
        return self.IsUnivaluedTree2(root.left) and self.IsUnivaluedTree2(root.right)

#iterative:

    def IsUnivaluedTree3(self, root):
        stack = []
        stack.append(root)
        while stack:
            node = stack.pop()
            if node.val != root.val:
                return False
            else:
                if node.left:
                    stack.append(node.left)
                if node.right:
                    stack.append(node.right)
        return True

#####################################################
# 993. Cousins in binary tree
# In a binary tree, the root node is at depth 0, and children of each depth k node are at depth k+1.
#Two nodes of a binary tree are cousins if they have the same depth, but have different parents.
#We are given the root of a binary tree with unique values, and the values x and y of two different nodes in the tree.
#Return true if and only if the nodes corresponding to the values x and y are cousins.

#recursive

    def IsCousins(self, root, x, y):
        def dfS(node, parent, depth, val):
            if node:
                if node.val == val:
                    return depth, parent
                return dfS(node.left, node, depth+1, val) or dfS(node.right, node, depth+1, val)
        dx, px, dy, py = dfS(root, None, 0, x) + dfS(root, None, 0, y)
        return dx == dy and px != py

#iterative
    def IsCousins2(self, root, x, y):
        nodes = collections.defaultdict(list)
        queue = [(root, 0, 0)]
        while queue:
            node, depth, parent = queue.pop()
            nodes[node.val] = [depth, parent]

            if node.left:
                queue.append((node.left, depth+1, node.val))
            if node.right:
                queue.append((node.right, depth+1, node.val))
            
        return nodes[x][0] == nodes[y][0] and nodes[x][1] != nodes[y][1]
            
    def IsCousins3(self, root, x, y):
        firstParent, queue = None, [root]
        while queue:
            nextQueue = []
            for node in queue:
                for nextNode in (node.left, node.right):
                    if nextNode.val in (x, y):
                        if not firstParent:
                            firstParent = node
                        else:
                            return firstParent != node
                        nextQueue.append(nextNode)
            if firstParent: #found the parent but not having the comparison, meaning the other parent is not found, return false
                return False
            queue = nextQueue

    def IsCousins4(self, root, x, y):
        q = collections.deque((root, 0))
        if root.val in (x, y):
            return False
        depth, d = 0, []
        while q:
            for _ in range(len(q)):
                node, depth = q.popleft()
                if node.left:
                    if node.left.val in (x, y):
                        d.append((depth+1, node))
                    q.append((node.left, depth+1))
                if node.right:
                    if node.right.val in (x, y):
                        d.append((depth+1, node))
                    q.append((node.right, depth+1))
            depth += 1
        return len(d) == 2 and d[0][0] == d[1][0] and d[0][1] != d[1][1]


#####################################################
# 1022. Sum of root to leaf binary numbers
# Given a binary tree, each node has value 0 or 1.  Each root-to-leaf path represents a binary number starting with the most significant bit.  For example, if the path is 0 -> 1 -> 1 -> 0 -> 1, then this could represent 01101 in binary, which is 13.
# For all leaves in the tree, consider the numbers represented by the path from the root to that leaf.
# Return the sum of these numbers.

    def SumRootToLeaf(self, root, val=0):
        if not root:
            return 0
        val = val * 2 + root.val
        if root.left == root.right: #get down to the bottom of the tree
            return val
        return self.SumRootToLeaf(root.left, val) + self.SumRootToLeaf(root.right, val)

    def SumRootToLeaf2(self, root):
        res = []
        def DDFFSS(root, res, tmp):
            if not root:
                return
            if not root.left and not root.right:
                res.append(tmp*2 + root.val)
                return
            DDFFSS(root.left, res, tmp*2 + root.val)
            DDFFSS(root.right, res, tmp*2 + root.val)

        DDFFSS(root, res, 0)
        return sum(res)

    
#####################################################
# 1114. Print in order
# Suppose we have a class:

#public class Foo {
#  public void first() { print("first"); }
#  public void second() { print("second"); }
#  public void third() { print("third"); }
#}
#The same instance of Foo will be passed to three different threads. Thread A will call first(), thread B will call second(), and thread C will call third(). Design a mechanism and modify the program to ensure that second() is executed after first(), and third() is executed after second().

#Example 2: Input: [1,3,2] Output: "firstsecondthird"
#Explanation: The input [1,3,2] means thread A calls first(), thread B calls third(), and thread C calls second(). "firstsecondthird" is the correct output.

class PrintInOrder:
    def __init__(self):
        self.two = Semaphore(0)
        self.three = Semaphore(0)

    def first(self, printFirst):
        printFirst() #print("first")
        self.two.release()

    def second(self, printSecond):
        with self.two:
            printSecond() #print("second")
            self.three.release()

    def third(self, printThird):
        with self.three:
            printThird() #print("third")


# Logic 1: Naive -100 pass 5% faster
# * Create state variables to maintain the order of execution
# * As the execution can be on any order, we make a blocking call by using while loops for the state variables
# * Only after the blocking call breaks the execution will happen thereby maintaining the order

class PrintInOrder2:
    def __init__(self):
        # those are labels checking whether the functions were executed
        self.first_check = False        
        self.second_check = False       

    def first(self, printFirst: 'Callable[[], None]') -> None:
        # just execute the first function
        printFirst()
        self.first_check = True

    def second(self, printSecond: 'Callable[[], None]') -> None:
        # this while loop runs continously to wait for first function to run
        while not self.first_check:
            continue

        # when it leaves the loop, second function can be run
        printSecond()
        self.second_check = True

    def third(self, printThird: 'Callable[[], None]') -> None:
        # analogously third function
        while not self.second_check:
            continue

        printThird()
        

# Logic 2:  Using Locks --> Events for trigerring ==> 70% faster
# * Create events based on the order of execution and use them


class PrintInOrder3(object):
    def __init__(self):
        self.first_done = Event()
        self.second_done = Event()


    def first(self, printFirst):
        """
        :type printFirst: method
        :rtype: void
        """
        
        printFirst()
        self.first_done.set()


    def second(self, printSecond):
        """
        :type printSecond: method
        :rtype: void
        """
        self.first_done.wait()
        printSecond()
        self.second_done.set()
        self.first_done.clear()
            
            
    def third(self, printThird):
        """
        :type printThird: method
        :rtype: void
        """
        self.second_done.wait()
        printThird()
        self.second_done.clear()