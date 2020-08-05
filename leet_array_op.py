#import itertools
#from itertools import chain
#from itertools import islice
import itertools

import random
import collections
import numpy #for 867
import math
import bisect
import operator
import heapq # for 1005

#####################################################
# 4. Median of two sorted array
# use binary search to compare the value
class solution:
    def MedianSortedArray(self, n1, n2):
        a, b = sorted((n1, n2), key=len) #a is the array in shorter length 
        m, n = len(a), len(b)
        limit = int((m + n + 1)/2)  #(m + n)//2
        l, r = 0, m
        while(l < r):
            i = int((l + r)/2)
            if(i < m and b[limit-i-1] > a[i]):
                l = i + 1
            elif(i > 0 and a[i-1]>b[limit-i]):
                r = i - 1
            else:
                if i == 0: maxl = b[limit-i-1]
                elif (limit-i) == 0: maxl = a[i-1]
                else: maxl = max(b[limit-i-1], a[i-1])

                if (m+n)%2 == 1:
                    return maxl
                
                if i == m: maxr = b[limit-i]
                elif (limit - i) == n: maxr = a[i]
                else: maxr = max(b[limit-i, a[i]])
                return (maxl + maxr)/2

#nums1 = [1, 2]
#nums2 = [3, 4, 5, 6]
#a = solution()
#a.MedianSortedArray(nums1,nums2)


#####################################################
# 10. Container With Most Water - left and right borders to define the rectangular area
#  which holds the most water
# use left -> center <- right
    def MostWater(self, h):  
        # h is a list with various heights
        w, l, r = 0, 0, len(h)-1
        while l < r:
            w = max(w, min(h[l], h[r])*(r-l))
            if (h[l] < h[r]):
                l += 1
            else:
                r -= 1
        return w
    
#h = [1,8,6,2,5,4,8,3,7]
#a.MostWater(h)

#####################################################
# Remove Nest Lists
#use recursive

#n = ['2', '7', ['11', '15']]
#print(list(itertools.chain(*n)))
#['2', '7', '11', '15']

#n1 = [[1,2,3],[4,5,6], [7], [8,9]]
#print(list(itertools.chain.from_iterable(n1)))
#[1, 2, 3, 4, 5, 6, 7, 8, 9]

    def reemovNestings(self, lst): 
        output=[]
        for i in lst: 
            if type(i) == list: 
                self.reemovNestings(i) 
            else: 
                output.append(i) 
        return output

#n = [2,7,[11,15]]       
#a.reemovNestings(n)
#[2, 7, 11, 15]


#####################################################
# 26. Remove duplicates from sorted array
# use the same storage space by moving two indexes

    def Dedup(self, sa):
        if not sa:
            print("empty array")
            return
        t = sa[0]
        k = 1
        for i in range(1, len(sa)):
            if(sa[i] != t):
                sa[k] = sa[i]
                t = sa[i]
                k += 1
        return sa[:k]

#n = [0,0,1,1,1,2,2,3,3,4]
#a=solution()
#a.Dedup(n)

#####################################################
# 27. Remove duplicates from sorted array
# use the same storage space by moving two indexes

    def RemoveVal(self, n, val):
        if not n:
            return
        k = 0
        for i in range(len(n)):
            if n[i] != val:
                n[k] = n[i]
                k += 1
        return n[:k]

#n = [0,1,2,2,3,0,4,2]
#val = 2
#a.RemoveVal(n,val)

#####################################################
# 35. Search insert position in sorted array
# use divide in half
    def SearchInsertPosition(self, sa, target):
        for i in range(len(sa)):
            if sa[i] >= target:
                return i
        return len(sa)

    def SearchInsertPosition2(self, sa, target):
        l, r = 0, len(sa)-1
        while l <= r:
            mid = (l + r)//2
            if sa[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return l

#nums, target = [1,3,5,7], 6
#a=Solution()
#a.SearchInsertPosition2(nums,target)

#####################################################
# 55. Maximum subarray 
# Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

    def MaximumSubarray(self, nums):
        if not nums:
            return 0
        curS = maxS = nums[0]
        for n in nums[1:]:
            curS = max(n, curS + n)
            maxS = max(maxS, curS)
        return maxS
    
#nums = [-2,1,-3,4,-1,2,1,-5,4]
#a=Solution()
#a.MaximumSubarray(nums)


#####################################################
# 70. Climing stairs
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# use dynamic programming

    def ClimbingStairs(self, n):
        a, b = 1, 1
        for  _ in range(n):
            a, b = b, a+b
        return a

    def ClimbingStairs2(self, n):
        res = [0 for i in range(n)]
        res[0], res[1] = 1, 2
        for i in range(2, n):
            res[i] = res[i-1] + res[i-2]
        return res[-1]

#n=10
#a = solution()
#a.ClimbingStairs(n)

#####################################################
# 88. Merge sorted array
# use the same storage space, starting backwards from the highest value

    def MergeSortedArray(self, nums1, m, nums2, n):
        i, j, k = m-1, n-1, m+n-1
        while i >= 0 and j <= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        if j >= 0:
            nums1[:k+1] = nums2[: j+1]
        return nums1
    
#nums1, m, nums2, n=[1,2,3,0,0,0], 3, [2,5,6], 3
#a = solution()
#a.MergeSortedArray(nums1,m,nums2,n)

#####################################################
# 121. Best time to buy and sell stock - one transaction only
# return maximum profit

    def BestTimeStock(self, nums):
        if not nums: 
            return 0
        mins, maxs = nums[0], 0 #mins - min cost, maxs - max profit
        for i in range(1, len(nums)):
            mins = min(mins, nums[i])
            maxs = max(maxs, nums[i]-mins)
        return maxs

#nums=[7,1,5,3,6,4]
#a=solution()
#a.BestTimeStock(nums)
#nums=[7,6,4,3,1]
#a.BestTimeStock(nums)

#####################################################
# 122. Best time to buy and sell stock II - multiple transactions in total
# but no multiple transaction at the same time - you must sell the stock before you buy again

    def BestTimeStockII(self, nums):
        return sum([max(nums[i]-nums[i-1], 0) for i in range(1, len(nums))])

#####################################################
# 136. Single numbrer
# Given a non-empty array of integers, every element appears twice except for one. Find that single one.

    def SingleNumber(self, nums):
        res = 0
        for n in nums:
            res ^= n
        return res

#nums=[4,1,2,1,2]
#a.SingleNumber(nums)

#####################################################
# 167. Two Sum II - sorted array
# use binary search

    def TwoSumII(self, nums, target):
        for i in range(len(nums)):
            l, r, tmp = i+1, len(nums)-1, target - nums[i]
            while l <= r:
                mid = (l+r)//2
                if nums[mid] == tmp:
                    return [i+1, mid+1]
                elif nums[mid] < tmp:
                    l = mid+1
                else:
                    r = mid-1

#nums,target=[2,7,11,15],9
#a.TwoSumII(nums,target)

#####################################################
# 169. Majority Element
# The majority element is the element that appears more than ⌊ n/2 ⌋ times.

    def MajorityElement(self, nums):
        dic = {}  #set
        for n in nums:
            dic[n] = dic.get(n, 0) + 1
            if dic[n] > len(nums)//2:
                return n

#nums=[2,2,1,1,1,2,2]
#a.MajorityElement(nums)

# return sorted(nums)[n//2]

#####################################################
# 170. Two Sum III - Data structure design.
#Design and implement a TwoSum class. It should support the following
#operations: add and find.
#add - Add the number to an internal data structure.
#find - Find if there exists any pair of numbers which sum is equal to the value.
#For example,
#    add(1); add(3); add(5);
#    find(4) -> true
#    find(7) -> false

class TwoSumIII():
    def __init__(self):
        self.dic={}

    def add(self, n):
        self.dic[n] = self.dic.get(n, 0) + 1

    def find(self, target):
        for m in self.dic:
            if (target - m) in self.dic and ((target - m) != m or self.dic[m] >1):
                return True
            return False


#####################################################
# 189. Rotate Array
# Given an array, rotate the array to the right by k steps, where k is non-negative.

    def RotateArray(self, nums, k):
        if not nums or not k:
            return None
        k %= len(nums)  # k = k%len(nums)
        if k:
            nums[:k], nums[k:] = nums[-k:], nums[:-k]
        return nums


#nums,k=[1,2,3,4,5,6,7],3
#a=solution()
#a.RotateArray(nums,k)
#[5, 6, 7, 1, 2, 3, 4]

#####################################################
# 198. House Robber - maximum the visits without adjacent visit
# dynamic programming

    def HouseRobber(self, nums):
        prev, now = 0, 0
        for i in nums:
            prev, now = now, max(prev + i, now)
        return now

#nums=[2,7,9,3,1]
#a=solution()
#a.HouseRobber(nums)
#nums=[1,2,3,1]
#a.HouseRobber(nums)


#####################################################
# 217. Contains duplicates

    def ContainesDuplicate(self, nums):
        return len(nums) == len(set(nums))

#####################################################
# 219. Contains duplicates II
# Given an array of integers and an integer k, find out whether there are two distinct indices i and j in the array such that 
# nums[i] = nums[j] and the absolute difference between i and j is at most k.

    def ContainesDuplicateII(self, nums, k):
        dic = {}
        for i in range(len(nums)):
            if nums[i] in dic:
                if i - dic[nums[i]] <= k:
                    return True
            dic[nums[i]] = i
        return False

#nums, k = [1,2,3,1], 3
#a.ContainsDuplicateII(nums,k)

#####################################################
# 252. Meeting rooms
# Given an array of meeting time intervals consisting of start and end times
#[[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.

    def MeetingRoom(self, intervals):
        intervals.sort(key=lambda i: i[0])
        return all([intervals[i][0] >= intervals[i-1][1] for i in range(1, len(intervals))])


#intervals=[[0,30],[5,10],[15,20]]
#a.MeetingRooms(intervals)
#intervals=[[7,10],[2,4]]
#a.MeetingRooms(intervals)

#####################################################
# 256. Paint house
# use three colors to paint n houses such that no two adjuacent houses have the same color
# The cost of painting each house with a certain color is represented by a n x 3 cost matrix. 
# Find the minimum cost to paint all houses.

# use dynamic programming
    def PaintHouse(self, costs): #costs is the n x 3 matrix
        h = [0]*3
        for rowc in costs: 
            h = [ min(h[:j] + h[j+1:]) + rowc[j] for j in range(3) ]
            print (h)
        return min(h)

#costs = [
#    [1, 2, 4],
#    [3, 0, 1],
#    [2, 3, 4]
#]
#a.PaintHouse(costs)

#[1, 2, 4]
#[5, 1, 2] # 5 = min(2,4)+3; 1 = min(1,4)+0; 2 = min(1,2)+1
#[3, 5, 5] # 3 = min(1,2)+2; 5 = min(5,2)+3; 5 = min(5,1)+4
# output: 3


#####################################################
# 276. Paint fence
# n posts paited with k colors, no more than two adjacent fence post have the same color
# return total number of ways you can paint
# use dynamic programming

    def PaintFence(self, n, k):
        if n <= 0 or k <= 0: 
            return 0
        dp = [0]*n
        dp[0] = k
        if n > 1:
            dp[1] = k*k
        for i in range(2, n):
            dp[i] = dp[i-1]*(k-1) + dp[i-2]*(k-1)
        return dp[-1]

#n,k=3, 3
#a.PaintFence(3,3)

#####################################################
# 278. First bad version
# Given n versions and find the first bad version
# call isBadVersion(i) -> (False/True)
# use binary search

    def isBadVersion(self, vnumb):
        return bool(random.getrandbits(1))

    def FirstBadVersion(self, n):
        low, high = 1, n
        version = -1
        while low <= high:
            mid = (low + high)//2
            if self.isBadVersion(mid):
                version = mid
                high = mid - 1
            else:
                low = mid + 1
        return version

#####################################################
# 283. Move zeroes
# Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.
# use the in-place operation

    def MoveZeroes(self, nums):
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i] #nums[j] is zero
                j += 1
        return nums

    def MoveZeroes2(self, nums):
        for j in range(len(nums)):
            i = j
            if nums[j] == 0:
                while i < len(nums)-1 and nums[i] == 0:
                    i += 1
                if i > j:
                    nums[i], nums[j] = nums[j], nums[i]
            print(nums, i, j)
        return nums

#####################################################
# 292. Nim game
# move stones in turns
# use dynamic programming

    def NimGame(self, n):
        return n%4 != 0

    def NimGame2(self, n):
        if n <=3:
            return True
        res = [False]*n
        res[0], res[1], res[2], i = True, True, True, 3
        while i < n:
            if res[i-3] and res[i-2] and res[i-1]:
                res[i] = False
            else:
                res[i] = True
            i += 1
        return res[-1]

#n=24
#a.NimGame2(n)


#####################################################
# 293. Flip game
# each move is to flip two consecutive "++" into "--"
# return all the moves
# use dynamic programming

    def Flipgame(self, s):
        i, res = 0, []
        while i < len(s) - 1:
            if s[i] == s[i+1] == '+':
                res.append(s[:i] + "--" + s[i+2:])
            i += 1
        return res
    
#strs="++++"
#a.FlipGame(strs)
#['--++', '+--+', '++--']


#####################################################
# 349. Intersection of two arrays

    def IntersectionTwoArrays(self, nums1, nums2):
        return list(set(nums1) & set(nums2))
    
    def IntersectionTwoArrays2(self, nums1, nums2):
        nums1 = nums1 if len(nums1)<=len(nums2) else nums2
        nums2 = nums2 if len(nums1)<=len(nums2) else nums1
        res = []
        for n in nums1:
            if n in nums2:
                res.append(n)
        return list(set(res))

#What if the given array is already sorted? How would you optimize your algorithm?
#What if nums1's size is small compared to nums2's size? Which algorithm is better?
#What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

    def IntersectionTwoArrays3(self, nums1, nums2):
        res = []
        nums1.sort()
        nums2.sort()
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] > nums2[j]:
                j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                if nums1[i] not in res:
                    res.append(nums1[i])
                i += 1
                j += 1
        return res

    def IntersectionTwoArrays4(self, nums1, nums2):
        nums1 = nums1 if len(nums1)<=len(nums2) else nums2
        nums2 = nums2 if len(nums1)<=len(nums2) else nums1
        return [key for key, v in collections.Counter(nums1).items() if key in nums2]

#####################################################
# 350. Intersection of two arrays II
# Given two arrays, write a function to compute their intersection.
    def IntersectionTwoArraysII(self, nums1, nums2):
        c1 = collections.Counter(nums1)
        c2 = collections.Counter(nums2)
        return list((c1 & c2).elements())

    def IntersectionTwoArraysII2(self, nums1, nums2):
        dic, res = dict(), []
        for n1 in nums1:
            dic[n1] = dic.get(n1, 0)
        for n2 in nums2:
            if n2 in dic and dic.get(n2, 0) > 0:
                res.append(n2)
                dic[n2] -= 1
        return res


            
        


#####################################################
# 447. Number of boomerangs
#Given n points in the plane that are all pairwise distinct, a "boomerang" is a tuple of points (i, j, k) such that the distance between i and j equals the distance between i and k (the order of the tuple matters).
#Find the number of boomerangs. You may assume that n will be at most 500 and coordinates of points are all in the range [-10000, 10000] (inclusive).
#Example: Input: [[0,0],[1,0],[2,0]]  Output: 2
#Explanation:
#The two boomerangs are [[1,0],[0,0],[2,0]] and [[1,0],[2,0],[0,0]]

    def distance(self, a, b):
        return (a[0]-b[0])**2 + (a[1]-b[1])**2
    def NumberOfBoomerangs(self, points):
        distmap = collections.Counter()
        res = 0
        for i in range(0, len(points)):
            distmap.clear()
            for j in range(0, len(points)):
                if i == j:
                    continue
                dist = self.distance(points[i], points[j])
                distmap[dist] += 1
            for val in distmap.values():
                res += val * (val - 1)    #pair permutation
        return res
    
#HashMap Solution: O(N^2)
#For a point p, find its distance from all other points and store it in dictionary indexed by the distance. The entry in the dictionary would have key as a distance "d" and value as the number of points which are a distance d from p.
#Imagine there are 3 points (p2,p3,p4) at a distance 10 from point p1. How many boomerangs do these contribute? For p2, p3, we have (p1,p2,p3) and (p1,p3,p2). Similarly for p3,p4 and p1,p4 - so a total of 6 boomerangs. Let us generalize it.
#Imagine we have k points which are at a distance d1 from point p. How many ways can we choose two items from k items? Answer: (k * (k-1)/2 ). Now each choice yield 2 boomerangs (i.e. (p2, p3) and (p3,p2)). So we have k * (k-1).

    def NumberOfBoomerangs2(self, points):
        res = 0
        for i in range(len(points)):
            distance = {}
            for j in range(len(points)):
                if i != j:
                    f, g = points[i][0] - points[j][0], points[i][1] - points[j][1]
                    d = f*f + g*g
                    distance.setdefault(d, 0)
                    distance[d] += 1
            for d in distance:
                res += distance[d]*(distance[d]-1)
        return res

    





#####################################################
# 448. Find all numbers disappeared in an array
#Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.
#Find all the elements of [1, n] inclusive that do not appear in this array.

    def FindDisappearedNumbers(self, nums):
        t = set(range(1, len(nums)+1))
        return list(t - set(nums) & t)
    
    def FindDisappearedNumbers2(self, nums):
        marked = set(nums)
        return [i for i in range(1, len(nums)+1) if i not in marked]

#Approach 1: Iterate the array and mark the position implied by every element as negative. Then in the second iteration, we simply need to report the positive numbers.

    def FindDisappearedNumbers3(self, nums):
        for i in range(len(nums)):
            x = abs(nums[i])
            nums[x-1] = -1*abs(nums[x-1])
        return [i+1 for i in range(len(nums)) if nums[i] >0]

#Approach 2: Iterate the array and add N to the existing number at the position implied by every element. This means that positions implied by the numbers present in the array will be strictly more than N (smallest number is 1 and 1+N > N). Therefore. in the second iteration, we simply need to report the numbers less than equal to N to return the missing numbers..
    def FindDisappearedNumbers4(self, nums):
        N = len(nums)
        for i in range(len(nums)):
            x = nums[i]%N
            nums[x-1] += N
        return [i+1 for i in range(len(nums)) if nums[i] <= N]


#####################################################
# 453. Minimum moves to equal array elements
# Given a non-empty integer array of size n, find the minimum number of moves required to make all array elements equal, where a move is incrementing n - 1 elements by 1.
    def MinimumMovesToEqual(self, A):
        n, total, res = len(A), sum(A), 0
        while total % n != 0:
            res += 1
            total += res * (n-1)
        return res

#Solution using sorting the array
#Visualize the nums array as a bar graph where the value at each index is a bar of height nums[i]. Sort the array such that the bar at index 0 is minimum height and the bar at index N-1 is highest.
#Now in the first iteration, make a sequence of moves such that the height at index 0 becomes equal to height at index N-1. Clearly this takes nums[N-1]-nums[0] moves. After these moves, index N-2 will be the highest and index 0 will still be the minimum. Moreover, nums[0] will be same as nums[N-1].
#In the next iteration, lets do nums[N-2]-nums[0] moves on all elements except N-2. After this iteration, nums[0] and nums[N-2] will be same. In addition, nums[N-1] will also be the same as these two bars. Clearly, when we the repeat the process for nums[N-3] i.e. move all elements (except N-3) by nums[N-3]-nums[0], we make elements 0, N-3, N-2, and N-1 same.
    def MinimumMovesToEqual2(self, A):
        A.sort()
        res = 0
        for i in range(len(A)-1, -1, -1):
            if A[i] == A[0]:
                break
            res += A[i] - A[0]
        return res


#Optimal Solution by Transforming the problem
#A move can be interpreted as: "Add 1 to every element and subtract one from any one element". sum(nums_new) = sum(nums) + (n-1): we increment only (n-1) elements by 1.
#Visualize the nums array as a bar graph where the value at each index is a bar of height nums[i]. We are looking for minimum moves such that all bars reach the final same height.
#Now adding 1 to all the bars in the initial state does not change the initial state - it simply shifts the initial state uniformly by 1.This gives us the insight that a single move is equivalent to subtracting 1 from any one element with respect to the goal of reaching a final state with equal heights.
#So our new problem is to find the minimum number of moves to reach a final state where all nums are equal and in each move we subtract 1 from any element.
#The final state must be a state where every element is equal to the minimum element. Say we make K moves to reach the final state. Then we have the equation, N * min(nums) = sum(nums) - K.

    def MinimumMovesToEqual3(self, A):
        return sum(A) - len(A)*min(A)


#####################################################
# 455. Assign Cookies
# Refer to the online question description

#Example 1: Input: [1,2,3], [1,1] Output: 1
#Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
#And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
#You need to output 1.

#Example 2: Input: [1,2], [1,2,3] Output: 2
#Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2. 
#You have 3 cookies and their sizes are big enough to gratify all of the children, 
#You need to output 2.
    def FindContentChildren(self, greed, cookies):
        greed.sort()
        cookies.sort()
        i, j = 0, 0
        while i < len(greed) and j < len(cookies):
            if cookies[j] >= greed[i]:
                i += 1
            j += 1
        return i







#####################################################
# 463. Island perimeter
#You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water.
#Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).
#The island doesn't have "lakes" (water inside that isn't connected to the water around the island). One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.
# use dynamic programming
#Example:
#Input:
#[[0,1,0,0],
# [1,1,1,0],
# [0,1,0,0],
# [1,1,0,0]]
#Output: 16

#There's an edge if you can find two adjacent cells of different color.
    def IslandPerimeter(self, A):
        R, C = len(A, len(A[0]))
        def get(r, c):
            return A[r][c] if 0 <= r < R and 0 <= c < C else 0
        return sum( (get(r, c) ^ get(r-1, c)) + (get(r, c) ^ get(r, c-1)) \
            for r in range(R+1) for c in range(C+1) )
    # R+1, C+1 to include the bounds of perimeter

    def IslandPerimeter2(self, A):
        res = 0
        if not A: 
            return 0
        h, w = len(A), len(A[0])
        for i in range(h):
            for j in range(w):
                if A[i][j] == 1:
                    dir = [ [0, 1], [1, 0], [0, -1], [-1, 0]]
                    for t in range(4):
                        x, y = i + dir[t][0], j + dir[t][1]
                        if 0 <= x < h and 0 <= y < w and A[x][y] == 0:
                            res += 1
                        elif x == -1 or x == h or y == -1 or y == w:
                            res += 1
        return res 

#Since there are no lakes, every pair of neighbour cells with different values is part of the perimeter 
#(more precisely, the edge between them is). 
#So just count the differing pairs, both horizontally and vertically (for the latter I simply transpose the grid).
    def IslandPerimeter3(self, A):
        return sum( sum(map(operator.ne, [0]+row, row+[0])) for row in A + map(list, zip(*A)) )

    def IslandPerimeter4(self, A):
        grid_ext = ['0' + ''.join(str(x) for x in row) + '0' for row in A]
        grid_trans = list(map(list, zip(*A)))
        grid_ext += ['0' + ''.join(str(x) for x in row) + '0' for row in grid_trans ]
        return sum(row.count('01') + row.count('10') for row in grid_ext)

    def IslandPerimeter5(self, A):
        area = 0
        for row in A + list(map(list, zip(*A))):
            for x, y in zip([0] + row, row + [0]):
                area += int(x != y)
        return area

    def IslandPerimeter6(self, grid):
        m, n = len(grid), len(grid[0]) if grid else 0
        return sum([(r - 1 < 0  or grid[r-1][c] == 0) +\
                    (c - 1 < 0  or grid[r][c-1] == 0) +\
                    (r + 1 >= m or grid[r+1][c] == 0) +\
                    (c + 1 >= n or grid[r][c+1] == 0)
                    for r in range(m) 
                    for c in range(n) 
                    if grid[r][c] == 1]
                    )

    def IslandPerimeter7(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        num = 0

        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1:
                    if r == 0 or grid[r-1][c] == 0:
                        num += 1
                    if r == m-1 or grid[r+1][c] == 0:
                        num += 1
                    if c == 0 or grid[r][c-1] == 0:
                        num += 1
                    if c == n-1 or grid[r][c+1] == 0:
                        num += 1
        return num

    def IslandPerimeter8(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        M, N, perimeter = len(grid), len(grid[0]), 0
        for i in range(M):
            for j in range(N):
                if grid[i][j] == 1:
                    for x, y in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                        if 0 <= x < M and 0 <= y < N:
                            perimeter += int(grid[x][y] == 0)
                        else:
                            perimeter += 1  #add the borders
        return perimeter


#####################################################
# 485. Max consecutive ones
# Given a binary array, find the maximum number of consecutive 1s in this array.

    def FindMaxConsecutiveOnes(self, nums):
        return max([ len(one) for one in (''.join(str(n) for n in nums)).split('0') ])
    
    def FindMaxConsecutiveOnes4(self, nums):
        return len( max((''.join([str(n) for n in nums])).split('0'), key=len))

    def FindMaxConsecutiveOnes2(self, nums):
        cnt, res = 0, 0
        for num in nums:
            if num == 1:
                cnt += 1
                res = max(res, cnt)
            else:
                cnt = 0
        return res
    
    def FindMaxConsecutiveOnes3(self, nums):
        res = 0
        for key, group in itertools.groupby(nums):
            if key == 1:
                res = max(res, len(list(group)))
            return res
    
#nums=[1,1,0,1,1,1]
#for key, group in itertools.groupby(nums):
#    print(key,list(group))
#1 [1, 1]
#0 [0]
#1 [1, 1, 1]

#####################################################
# 506. Relative ranks
# Given scores of N athletes, find their relative ranks and the people with the top three highest scores, who will be awarded medals: "Gold Medal", "Silver Medal" and "Bronze Medal".
#Example 1: Input: [5, 4, 3, 2, 1]
#Output: ["Gold Medal", "Silver Medal", "Bronze Medal", "4", "5"]
    def FindRelativeRanks(self, nums):
        t = sorted(range(len(nums)), key = lambda x: nums[x], reverse=True)
        rank = ["Gold Medal", "Silver Medal", "Bronze Medal"] + list(map(str, range(4, len(nums)+1)))
        return ([rank(i) for i in t])

    def FindRelativeRanks2(self, nums):
        nums_sorted = sorted(nums)[::-1] #nums = sorted(nums, revers=True)
        rank = ["Gold Medal", "Silver Medal", "Bronze Medal"] + list(map(str, range(4, len(nums) + 1)))
        return map(dict(zip(nums_sorted, rank)).get, nums)

    def FindRelativeRanks3(self, nums):
        numsindx = sorted(range(len(nums)), key = lambda x: nums[x], reverse=True)
        rank = [str(i+1) for i in numsindx if i>2]
        return ["Gold Medal", "Silver Medal", "Bronze Medal"] + rank

    


    
#####################################################
# 521. Longest uncommon subsequence I
# Given a group of two strings, you need to find the longest uncommon subsequence of this group of two strings. The longest uncommon subsequence is defined as the longest subsequence of one of these strings and this subsequence should not be any subsequence of the other strings.
# The input will be two strings, and the output needs to be the length of the longest uncommon subsequence. If the longest uncommon subsequence doesn't exist, return -1.


#For strings A, B, when len(A) > len(B), the longest possible subsequence of either A or B is A, and no subsequence of B can be equal to A. Answer: len(A).
#When len(A) == len(B), the only subsequence of B equal to A is B; so as long as A != B, the answer remains len(A).
#When A == B, any subsequence of A can be found in B and vice versa, so the answer is -1.

    def FindLUSLength(self, a, b):
        return -1 if a == b else max(len(a), len(b))



#####################################################
# 561. Array partition I
# Given an array of 2n integers, your task is to group these integers into n pairs of integer, 
# say (a1, b1), (a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i from 1 to n as large as possible.

    def ArrayPairSum(self, nums):
        return sum( sorted(nums)[::2]) #sum up every other elements

    def ArrayPairSum2(self, nums):
        nums.sort()
        return sum(nums[i*2] for i in range(len(nums)//2))




#####################################################
# 566. Reshape the matrix
# 'reshape', which can reshape a matrix into a new one with different size but keep its original data.
# If the 'reshape' operation with given parameters is possible and legal, output the new reshaped matrix; Otherwise, output the original matrix.

    def MatrixReshape(self, A, R, C):
        if len(A) * len(A[0]) != R * C:
            return A
        vals = (val for row in A for val in row)
        return [[vals.next() for c in range(C)] for r in range(R)]

#Alternative solution without generators:

    def MatrixReshape2(self, A, R, C):
        if len(A) * len(A[0]) != R * C:
            return A
        vals = [val for row in A for val in row]
        res = [[None]*C for _ in range(R)]
        i = 0
        for r in range(R):
            for c in range(C):
                res[r][c] = vals[i]
                i += 1
        return res

    def MatrixReshape3(self, A, R, C):
        if not A:
            return []
        org, res = [], []
        if len(A)*len(A[0]) != R*C:
            return A
        for val in A:
            org += val  #org.append(val)
        for y in range(R):
            res += org[y*C:y*C+C],  #res is a list, "," at the end of the operand is necessary
        return res

    def MatrixReshape4(self, A, R, C):
        if len(A) * len(A[0]) != R * C:
            return A
        it = itertools.chain(*A)
        return [list(itertools.islice(it, C)) for _ in range(R)]

    def MatrixReshape5(self, A, R, C):
        flat = sum(A, [])
        if len(flat) != R*C:
            return A
        tuples = zip(*([iter(flat)] * C))
        return map(list, tuples)

    def MatrixReshape6(self, A, R, C):
        try:
            return numpy.reshape(A, (R, C)).tolist()
        except:
            return A






#####################################################
# 575. Distribute candies
# refer to the question description online

    def DistributeCandies(self, candies):
        return min(len(candies) / 2, len(set(candies)))


#####################################################
# 598. Range addition II
# Given an m * n matrix M initialized with all 0's and several update operations.
# Operations are represented by a 2D array, and each operation is represented by an array with two positive integers a and b, which means M[i][j] should be added by one for all 0 <= i < a and 0 <= j < b.
# You need to count and return the number of maximum integers in the matrix after performing all the operations.

    def RangeAdditionMaxCount(self, R, C, ops):
        if not ops:
            return R * C
        x, y = zip(*ops)
        return min(x) * min(y)  #get the min(row) and min(column) in a overlapped area, return area value

    def RangeAdditionMaxCount2(self, R, C, ops):
        return min(op[0] for op in ops) * min(op[1] for op in ops) if ops else R * C
    
    def RangeAdditionMaxCount3(self, R, C, ops):
        grid = [[0] * C for _ in range(R)]
        for op in ops:
            for i in range(op[0]):
                for j in range(op[1]):
                    grid[i][j] += 1
        h = [ grid[i][j] for i in range(R) for j in range(C)]
        return collections.Counter(h)[max(h)]
    
    
#####################################################
# 599. Minimum index sum of two lists
# Input: ["Shogun", "Tapioca Express", "Burger King", "KFC"] ["KFC", "Shogun", "Burger King"]
#Output: ["Shogun"]
#Explanation: The restaurant they both like and have the least index sum is "Shogun" with index sum 1 (0+1).

    def FindCommonMiniumSumIndex(self, A, B):
        d = set(A) & set(B)
        c = collections.Counter({v: i for i, v in enumerate(A) if v in d}) + \
            collections.Counter({v: i for i, v in enumerate(B) if v in d})
        return [i for i, v in c.items() if v == min(c.values())]

    def FindCommonMiniumSumIndex2(self, A, B):
        Ad = {v: i for i, v in enumerate(A)}
        minv, res = 1e9, []
        for j, v in enumerate(B):
            i = Ad.get(v, 1e9)
            if i + j < minv:
                minv, res = i + j, [v]
            elif i + j == minv:
                res.append(v)
        return res

# a single path:
    def FindCommonMiniumSumIndex3(self, A, B):
        m, n, dic, minv, res = len(A), len(B), dict(), 1e9, []
        for i in range(max(m, n)):
            if i < m:
                if not dic[A[i]]:
                    dic[A[i]] == -1
                elif dic[A[i]] == -2:
                    dic[A[i]] += i + 2
                    if dic[A[i]] < minv:
                        res = [A[i]]
                    elif dic[A[i]] == minv:
                        res.append(A[i])
            if i < n:
                if not dic[B[i]]:
                    dic[B[i]] == -2
                elif dic[B[i]] == -1:
                    dic[B[i]] += i + 1
                    if dic[B[i]] < minv:
                        res = [B[i]]
                    elif dic[B[i]] == minv:
                        res.append(B[i])
        return res
        

#####################################################
# 657. Robot return to origin
# a robot starting at position (0, 0) with valid moves are R (right), L (left), U (up), and D (down).
# Example 1: Input: "UD" Output: true 
# Explanation: The robot moves up once, and then down once. All moves have the same magnitude, so it ended up at the origin where it started. Therefore, we return true.
 
# Example 2: Input: "LL" Output: false
# Explanation: The robot moves left twice. It ends up two "moves" to the left of the origin. We return false because it is not at the origin at the end of its moves.

    def JudgeCircle(self, moves):
        return not sum(1j ** 'RUL'.find(m) for m in moves)
# explain the coding above
# 'RUL'.find(m) returns: 0 for 'R', 1 for 'U', 2 for 'L', -1 for 'D'
# 1j**'RUL'.find(m) returns: 1 for 'R', 1j for 'U', -1 for 'L', -1j for 'D'
# sum will cancel off 'L' and 'R', 'U and 'D' separately.

    def JudgeCircle2(self, moves):
        pos = [0, 0]
        for m in moves:
            if m == 'U': pos[0] += 1
            elif m == 'D': pos[0] -= 1
            elif m == 'L': pos[1] -= 1
            elif m == 'R': pos[1] += 1
        return pos == [0, 0]

    def JudgeCircle3(self, moves):
        return moves.count('L') == moves.count('R') and moves.count('U') == moves.count('D')

#####################################################
# 661. Image smoother
# Given a 2D integer matrix M representing the gray scale of an image, you need to design a smoother to make the gray scale of each cell becomes the average gray scale (rounding down) of all the 8 surrounding cells and itself. If a cell has less than 8 surrounding cells, then use as many as you can.
    def ImageSmoother(self, M):
        m, n = len(M), len(M[0])
        grid = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                adj = [M[i+x][j+y] for x, y in ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,1),(1,-1)) \
                        if 0 <= i+x < m and 0 <= j+y < n]
                grid[i][j] = sum(adj) // len(adj)
        return grid

#####################################################
# 665. Non-decreasing Array
# Given an array with n integers, your task is to check if it could become non-decreasing by modifying at most 1 element.

    def CheckNonDecreasingArray(self, nums):
        cnt_dec = 0
        for i in range(len(nums) - 1):
            if nums[i] > nums[i+1]:
                cnt_dec += 1
                if i == 0:
                    nums[i] = nums[i+1]
                elif nums[i-1] <= nums[i+1]:
                    nums[i] = nums[i-1]
                else:
                    nums[i+1] = nums[i]
            if cnt_dec > 1:
                return False
        return True

# use count to remember how many descending pairs in the array, if count>=2, obviously we cannot modify a single element to obtain a non-descreaing array, so we return False. When count equals one, we check if we can modify nums[i] (the first one in the descending pair, by decreasing it) or nums[i+1] (the second one in the descending pair, by increasing it), if in both situations, we cannot, then we will also return False. In this way, the situation that return False is complete, the others will return True. And also in this way, we can return much earlier

    def CheckNonDecreasingArray2(self, nums):
        cnt = 0
        for i in range(len(nums)-1):
            if nums[i] > nums[i+1]:
                cnt += 1
                if cnt > 1 or ( (i-1 > 0 and nums[i-1] > nums[i+1]) and (i+2 < len(nums) and nums[i+2] < nums[i])):
                    return False
        return True



#####################################################
# 690. Employee Importance
#You are given a data structure of employee information, which includes the employee's unique id, his importance value and his direct subordinates' id.
#For example, employee 1 is the leader of employee 2, and employee 2 is the leader of employee 3. They have importance value 15, 10 and 5, respectively. Then employee 1 has a data structure like [1, 15, [2]], and employee 2 has [2, 10, [3]], and employee 3 has [3, 5, []]. Note that although employee 3 is also a subordinate of employee 1, the relationship is not direct.
#Now given the employee information of a company, and an employee id, you need to return the total importance value of this employee and all his subordinates.
#Example 1: Input: [[1, 5, [2, 3]], [2, 3, []], [3, 3, []]], 1
#Output: 11

    def EmployeeImportance(self, employees, id):
        employee = { e[0]: e for e in employees }
        def total(id):
            e = employee[id]
            return e[1] + sum(map(total, e[2]))
        return total(id)

    def EmployeeImportance2(self, employees, id):
        dic = {employee[0]: employee for employee in employees}
        def val(employee):
            return employee[1] + sum(val(dic[sub]) for sub in employee[2])
        return val(dic[id])

    def EmployeeImportance3(self, employees, id):
        em = [e for e in employees if e[0] == id][0]
        res, subor = em[1], em[2]
        if subor:
            return res + sum(e[1] for e in employees for sid in subor if sid == e[0])


#####################################################
# 697. Degree of an array
# Given a non-empty array of non-negative integers nums, the degree of this array is defined as the maximum frequency of any one of its elements.
# Your task is to find the smallest possible length of a (contiguous) subarray of nums, that has the same degree as nums.
    def FindShortesSubArray(self, nums):
        first, last = {}, {}
        for i, v in enumerate(nums):
            first.setdefault(v, i)
            last[v] = i
            c = collections.Counter(nums)
            degree = max(c.values())
            return min(last[v] - first[v] + 1 for v in c if c[v] == degree)

# a single path

    def FindShortesSubArray2(self, nums):
        first, counter, res, degree = {}, {}, 0, 0
        for i, v in enumerate(nums):
            first.setdefault(v, i)
            counter[v] = counter.get(v, 0) + 1
            if counter[v] > degree:
                degree = counter[v]
                res = i  - first[v] + 1
            elif counter[v] == degree:
                res = min(res, i - first[v] + 1)
        return res

    def FindShortesSubArray3(self, nums):
        cnt, res = collections.Counter(nums), collections.defaultdict(list)
        degree = max(cnt.values())
        for i, v in enumerate(nums):
            res[v].append(i)
        return min(res[v][-1] - res[v][0] + 1 for v in cnt if cnt[v] == degree)


#####################################################
# 704. Binary search
# Given a sorted (in ascending order) integer array nums of n elements and a target value, write a function to search target in nums. If target exists, then return its index, otherwise return -1.
# use binary search
    def BinarySearch(self, nums, target):
        l, r = 0, len(nums)-1
        while l < r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1 
            else:
                r = mid - 1
        return -1

    



#####################################################
# 733. Flood fill
# An image is represented by a 2-D array of integers, each integer representing the pixel value of the image (from 0 to 65535).
#Given a coordinate (sr, sc) representing the starting pixel (row and column) of the flood fill, and a pixel value newColor, "flood fill" the image.
#To perform a "flood fill", consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color as the starting pixel), and so on. Replace the color of all of the aforementioned pixels with the newColor.
#At the end, return the modified image.

#use BFS 

    def FloodFill(self, img, sr, sc, colorNew):
        colorOld, m, n = img[sr][sc], len(img), len(img[0])
        if colorOld != colorNew:
            q = collections.deque([sr, sc])
            while q:
                i, j = q.popleft()
                img[i][j] = colorNew
                for x, y in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if 0 <= x < m and 0 <= y < n and img[x][y] == colorOld:
                        q.append((x, y))
        return img

#use DFS
    def FloodFill2(self, img, sr, sc, colorNew):
        def dfs(i, j):
            img[i][j] = colorNew
            for x, y in  ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if 0 <= x < m and 0 <= y < n and img[x][y] == colorOld:
                    dfs(x, y)
        colorOld, m, n = img[sr][sc], len(img), len(img[0])
        if colorOld != colorNew:
            dfs(sr, sc)
        return img


#####################################################
# 746. Minimum cost climing stairs
# On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).
#Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.
# use dynamic programming
# refer to question 70. Climbing stairs

    def MinimumCostClimbingStairs(self, cost):
        s = [0]
        for c in cost:
            s = c + min(s), s[0]
        return min(s)

    def MinimumCostClimbingStairs2(self, cost):
        for i in range(2, len(cost)):
            cost[i] += min(cost[i-1], cost[i-2])
        return min(cost[-2:])
        #return min(cost[-1], cost[-2])

    def MinimumCostClimbingStairs3(self, cost):
        prev2, prev = cost[0], cost[1]
        for i in range(2, len(cost)):
            prev2, prev = prev, cost[i] + min(prev2, prev)
        return min(prev2, prev)

    













#####################################################
# 760. Find anagram mappings
# Two anagram arrays A & B, find the B index when reading  
# These lists A and B may contain duplicates. If there are multiple answers, output any of them.

    def FindAnagramMappings(self, a, b):
        return [ b.index(c) for c in a]

    def FindAnagramMappings2(self, a, b):
        res = []
        for i in range(len(a)):
            res.append(b.index(a[i]))
        return res

    def FindAnagramMappings3(self, a, b):
        lookup = {val: i for i, val in enumerate(b)}
        return [ lookup(val) for val in a]


#####################################################
# 766. Toeplitz matrix
# A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same element.
# Input:
#matrix = [
#  [1,2,3,4],
#  [5,1,2,3],
#  [9,5,1,2]
#]
#Output: True

    def IsToeplitzMatrix(self, m):
        for i in range(len(m)-1):
            for j in range(len(m[0]-1)):
                if m[i][j] != m[i+1][j+1]:
                    return False
        return True

    def IsToeplitzMatrix2(self, m):
        return all(m[i][j] == m[i-1][j-1] for i in range(1, len(m)) for j in range(1, len(m[0])))


    def IsToeplitzMatrix3(self, m):
        return all(r1[:-1] == r2[1:] for r1, r2 in zip(m[:-1], m[1:]))


#####################################################
# 771. Jewels and stones
# find how many of the stones you have are also jewels

    def JewelsInStones(self, J, S):
        return sum(s in J for s in S)

    def JewelsInStones2(self, J, S):
        return sum(map(S.count, J))

    def JewelsInStones3(self, J, S):
        return sum(map(J.count, S))

    def JewelsInStones4(self, J, S):
        counter = collections.Counter(S)
        count = 0
        for ch in J:
            count += counter[ch]
        return count


#####################################################
# 812. Largest triangle area
# You have a list of points in the plane. Return the area of the largest triangle that can be formed by any 3 of the points.

    def LargestTriangleArea(self, p):
        return max(0.5 * abs(i[0]*j[1] + j[0]*k[1] + k[0]*i[1] - j[0]*i[1] - k[0]*j[1] - i[0]*k[1]) \
            for i, j, k in itertools.combinations(p, 3))

    def LargestTriangleArea2(self, p):
        return max(0.5 * abs(xa*yb + xb*yc + xc*ya - xb*ya - xc*yb - xa*yc)
                   for (xa, ya), (xb, yb), (xc, yc) in itertools.combinations(p, 3))

    def largestTriangleArea3(self, points):
        """
        :type points: List[List[int]]
        :rtype: float
        """
        def area(p1,p2,p3):
            x1,y1=p1
            x2,y2=p2
            x3,y3=p3
            return 0.5*(x1*y2+x2*y3+x3*y1-x3*y2-x2*y1-x1*y3)

        def boundary(points):
            points=sorted(points,key=lambda p: (p[0],p[1]))
            def cross(o,a,b): return (o[0]-a[0])*(o[1]-b[1])-(o[1]-a[1])*(o[0]-b[0])
            lower=[]
            for p in points:
                while len(lower)>=2 and cross(lower[-2],lower[-1],p)<0: lower.pop()
                lower.append(tuple(p))
            upper=[]
            for p in reversed(points):
                while len(upper)>=2 and cross(upper[-2],upper[-1],p)<0: upper.pop()
                upper.append(tuple(p))
            return list(set(lower[:-1]+upper[:-1]))

        res=0
        bound=boundary(points)
        for p1,p2,p3 in itertools.permutations(bound,3):
            tmp=area(p1,p2,p3)
            if tmp>res: res=tmp
        return res



#####################################################
# 821. Shortest distance to a character
# Given a string S and a character C, return an array of integers representing the shortest distance from the character C in the string.

    def ShortestToChar(self, s, c):
        n = len(s)
        res, pos = [n]*n, -n
        for i in range(n) + range(n)[::-1]: #compare both sides
            if s[i] == c:
                pos = i
            res[i] = min(res[i], abs(i-pos))
        return res

    def ShortestToChar2(self, s, c):
        n = len(s)
        res = [0 if sc == c else n for sc in s]
        for i in range(n-1): res[i+1] = min(res[i+1], res[i]+1)
        for i in range(n-1)[::-1]: res[i] = min(res[i], res[i+1]+1)
        return res

#Using fwd, we track distance from closest matching character moving left-to-right.
#Using rev, we track distance from closest matching character moving right-to-left.
#We compute the answer in a single pass. The ~ operator allows us to index from the right of the string.

    def ShortestToChar3(self, s, c):
        res = [math.inf]*len(s)
        fwd, rev = math.inf, math.inf
        for i in range(len(s)):
            if s[i] == c:
                fwd = 0
            else:
                fwd += 1
            res[i] = min(res[i] + fwd)
            if s[~i] == c:
                rev = 0
            else:
                rev += 1
            res[~i] = min(res[~i], rev)
        return res

    def ShortestToChar4(self, s, c):
        index = [i for i, st in enumerate(s) if st == c]
        return [min(abs(idx-x) for x in index) for idx in range(len(s))]

    

#####################################################
# 832. Flipping an image
# Given a binary matrix A, we want to flip the image horizontally, then invert it, and return the resulting image.
#Input: [[1,1,0],[1,0,1],[0,0,0]]
#Output: [[1,0,0],[0,1,0],[1,1,1]]
#Explanation: First reverse each row: [[0,1,1],[1,0,1],[0,0,0]].
#Then, invert the image: [[1,0,0],[0,1,0],[1,1,1]]

    def FlipAndInvertImage(self, nums):
        return [ [1^i for i in n[::-1]] for n in nums]

#####################################################
# 833. Projection area of a 3D shapes
# On a N * N grid, we place some 1 * 1 * 1 cubes that are axis-aligned with the x, y, and z axes.
# Each value v = grid[i][j] represents a tower of v cubes placed on top of grid cell (i, j).
# Now we view the projection of these cubes onto the xy, yz, and zx planes.
# A projection is like a shadow, that maps our 3 dimensional figure to a 2 dimensional plane. 
# Here, we are viewing the "shadow" when looking at the cubes from the top, the front, and the side.
# Return the total area of all three projections.

# explanation:
#front-back projection area on xz = sum(max value for every col)
#right-left projection area on yz = sum(max value for every row)
#top-down projection area on xy = sum(1 for every v > 0)

#Example 1: Input: [[2]] Output: 5

    def ProjectionArea(self, grid):
        hor = sum(map(max, grid))
        ver = sum(map(max, zip(*grid)))
        top = sum(v > 0 for row in grid for v in row)
        return ver + hor + top

    def ProjectionArea2(self, grid):
        top = sum (v != 0 for row in grid for v in row)
        front = sum(max(row) for row in grid)
        side = sum(max(row[j] for row in grid) for j in range(len(grid[0])))
        return top + front + side




#####################################################
# 852. Peak index in a mountain array
# Given an array that is definitely a mountain, return any i such that A[0] < A[1] < ... A[i-1] < A[i] > A[i+1] > ... > A[A.length - 1].
# use binary search
    def PeakIndexInMountainArray(self, nums):
        l, r = 0, len(nums)-1
        while l < r:
            m = (l+r)//2
            if nums[m] < nums[m+1]:
                l = m + 1
            else:
                r = m
        return l

    def PeakIndexInMountainArray2(self, nums):
        return nums.index(max(nums))

#####################################################
# 867. Transpose matrix
# The transpose of a matrix is the matrix flipped over it's main diagonal, switching the row and column indices of the matrix.

    def TransposeMatrix(self, A):
        return zip(*A)

    def TransposeMatrix2(self, A):
        return [ list(item) for item in zip(*A)]

    def TransposeMatrix3(self, A):
        return [ [row[col] for row in A] for col in range(len(A[0]))]

    def TransposeMatrix4(self, A):
        return [ [A[i][j] for i in range(len(A))] for j in range(len(A[0])) ]

    def TransposeMatrix5(self, A):
        return numpy.transpose(A).tolist()

#####################################################
# 888. Fair candy swap
# refer to the online question description

    def FairCandySwap(self, A, B):
        diff = (sum(A) - sum(B)) //2  #diff can be a negative number
        A = set(A)
        for b in set(B):
            if diff + b in A:
                return [diff + b, b]

#####################################################
# 892. Surface area of 3D shapes
# On a N * N grid, we place some 1 * 1 * 1 cubes.
# Each value v = grid[i][j] represents a tower of v cubes placed on top of grid cell (i, j).
# Return the total surface area of the resulting shapes.

#Surface area of each cell is 4 * lateral area + upper and lower face unit areas.
#For each cell, neighbour cells blocks some area, which is the minimum of 2 cells taken into account.

    def SurfaceArea3D(self, grid):
        n, res = len(grid), 0
        for i in range(n):
            for j in range(n):
                if grid[i][j]:
                    res += grid[i][j]*4 + 2
                    res -= i and min(grid[i-1][j], grid[i][j])
                    res -= j and min(grid[i][j-1], grid[i][j])
                    res -= i < n - 1 and min(grid[i+1][j], grid[i][j])
                    res -= j < n - 1 and min(grid[i][j+1], grid[i][j])
        return res

    def SurfaceArea3D2(self, grid):
        n, res = len(grid), 0
        for i in range(n):
            for j in range(n):
                if grid[i][j]: 
                    res += 2 + grid[i][j]*4
                if i: res -= min(grid[i][j], grid[i - 1][j]) * 2
                if j: res -= min(grid[i][j], grid[i][j - 1]) * 2
        return res


    def SurfaceArea3D3(self, grid):
        n, res = len(grid), 0
        for i in range(n):
            for j in range(n):
                if grid[i][j]: 
                    res += 2
                    for x, y in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                        if 0 <= x < n and 0 <= y < n:
                            nval = grid[x][y]
                        else:
                            nval = 0
                        res += max(grid[x][y] - nval, 0)
        return res




#####################################################
# 905. Sort array by parity
# Given an array A of non-negative integers, return an array consisting of all the even elements of A, followed by all the odd elements of A.
# use in-place storage

    def SortArrayByParity(self, nums):
        return [ n for n in nums if not n%2] + [ n for n in nums if n%2]

    def SortArrayByParity2(self, nums):
        return sorted(nums, key = lambda x: x%2)
        #return sorted(nums, key = lambda x: x & 1)

    def SortArrayByParity3(self, nums):
        i = -1
        for j in range(len(nums)):
            if nums[j] % 2 == 0:
                i += 1
                if i < j:
                    nums[i], nums[j] = nums[j], nums[i]
        return nums

    def SortArrayByParity4(self, nums):
        left, right = 0, len(nums)-1
        while left < right:
            if not nums[left] % 2:
                left += 1
            elif nums[right] % 2:
                right -= 1
            else:
                nums[left], nums[right] = nums[right], nums[left]
        return nums

#nums=[6,6,6,3,1,2,4]
#a.sortArrayByParity(nums)

#####################################################
# 896. Monotonic array
# An array is monotonic if it is either monotone increasing or monotone decreasing.

    def MonotonicArray(self, nums):
        return all(a >= b for a, b in zip(nums, nums[1:])) or all(a <= b for a, b in zip(nums, nums[1:]))

    def MonotonicArray2(self, nums):
        return all(nums[i] <= nums[i-1] for i in range(1, len(nums))) or all(nums[i] >= nums[i-1] for i in range(1, len(nums)))
    
    #Python 2
    def MonotonicArray3(self, nums):
        #return not {cmp(i, j) for i, j in zip(nums, nums[1:])} >= {1, -1}
        return all( (a ^ b) >= 0 for a, b in zip(nums, nums[1:]))


# def oppositeSigns(x, y): 
#    return ((x ^ y) < 0); 

# check if n is positive, negative or zero
# 1 + (n>>31) – (-n>>31)

#####################################################
# 908. Smallest Range I
# Given an array A of integers, for each integer A[i] we may choose any x with -K <= x <= K, and add x to A[i].
# After this process, we have some array B.
# Return the smallest possible difference between the maximum value of B and the minimum value of B

#Intuition:
#If min(A) + K < max(A) - K, then return max(A) - min(A) - 2 * K
#If min(A) + K >= max(A) - K, then return 0

    def SmallestRangeI(self, A, K):
        return max(0, max(A)-min(A)-2*K)

    def SmallestRangeI2(self, A, K):
        l, r = min(A) + K, max(A) - K
        return 0 if l >= r else r - l

    

#####################################################
# 922. Sort array by parity II
# Given an array A of non-negative integers, half of the integers in A are odd, and half of the integers are even.
# Sort the array so that whenever A[i] is odd, i is odd; and whenever A[i] is even, i is even.

    def SortArrayByParityII(self, nums):
        even, odd = [a for a in nums if not a%2], [a for a in nums if a%2]
        return [even.pop() if not i%2 else odd.pop() for i in range(len(nums))]

    def SortArrayByParityII2(self, nums):
        res = [0]*len(nums)
        i, j = 0, 1
        for a in nums:
            if a%2 == 0:
                res[i] = a
                i += 2
            else:
                res[j] = a
                j += 2
        return res

# in-place storage
    def SortArrayByParityII23(self, nums):
        j = 1
        for i in range(0, len(nums), 2):
            if nums[i]%2 == 0:
                continue
            while nums[j]%2 == 1:
                j += 2
            nums[i], nums[j] = nums[j], nums[i]
        return nums

#####################################################
# 937. Recorder log files
# Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.

    def ReorderedLogFiles(self, logs):
        l = filter(lambda l: l[l.find(" ")+1].isalpha(), logs)
        d = filter(lambda l: l[l.find(" ")+1].isdigit(), logs)
        return sorted(l, key = lambda x: (x[x.find(" "):], x[:x.find(" ")])) + list(d)

    def ReorderedLogFiles2(self, logs):
        letter_logs, digit_logs = [], []
        for log in logs:
            if log.split()[1].isdigit():
                digit_logs.append(log)
            else:
                letter_logs.append(log)

        letter_logs.sort(key=lambda log: log.split()[0]) # when suffix is tie, sort by identifier
        letter_logs.sort(key=lambda log: log.split()[1:]) # sorted by suffix

        return letter_logs + digit_logs

    def ReorderedLogFiles3(self, logs):
        letterLogs, digitLogs = [], []
        for log in logs:
            logList = log.split()
            if logList[1][0].isalpha():
                letterLogs.append(log)
            else:
                digitLogs.append(log)
        letterLogsSorted = sorted(letterLogs, key = lambda x: ' '.join(x.split()[1:]))
        return letterLogsSorted + digitLogs

    



#####################################################
# 942. DI string match
# Given a string S that only contains "I" (increase) or "D" (decrease), let N = S.length.
# Return any permutation A of [0, 1, ..., N] such that for all i = 0, ..., N-1:
# If S[i] == "I", then A[i] < A[i+1]
# If S[i] == "D", then A[i] > A[i+1]
# use binary search

    def DIStringMatch(self, s):
        l, r, res = 0, 0, [0]
        for c in s:
            if c == 'I':
                r += 1
                res.append(r)
            else:
                l -= 1
                res.append(l)
        return [i - l for i in res] #note: i - L (L is the maximum negative index) to ensure the index from 0 to N-1
    
    def DIStringMatch2(self, s):
        res, inc, dec = [], 0, len(s)
        for c in s:
            if c == 'I':
                res += inc
                inc += 1
            else:
                res += dec
                dec -= 1
        res += inc
        return res


#####################################################
# 944. Delete columns to make sorted
# We are given an array A of N lowercase letter strings, all of the same length.
# Suppose we chose a set of deletion indices D such that after deletions, each remaining column in A is in non-decreasing sorted order.
# Return the minimum possible value of D.length.

# O(NlogNM)
    def MinimumDeletionSize(self, nums):
        return sum(list(col) != sorted(col) for col in zip(*nums))

# O(NM)
    def MinimumDeletionSize2(self, nums):
        return sum(any(a > b for a, b in zip(col, col[1:]) for col in zip(*nums)))
        
    def MinimumDeletionSize3(self, nums):
        return sum( any(a[j] > b[j] for a, b in zip(nums, nums[1:])) for j in range(len(nums[0])))




#####################################################
# 961. N-repeated element in size 2N array
# In a array A of size 2N, there are N+1 unique elements, and exactly one of these elements is repeated N times.

    def RepeatedNTimes(self, nums):
        rep_dic = dict(collections.Counter(nums))
        for key in rep_dic:
            if rep_dic[key] == len(nums)/2:
                return key

    def RepeatedNTimes2(self, nums):
        return int((sum(nums) - sum(set(nums))) // (len(nums)//2-1))

#####################################################
# 976. Largest perimeter triangle
# Given an array A of positive lengths, return the largest perimeter of a triangle with non-zero area, formed from 3 of these lengths.
#If it is impossible to form any triangle of non-zero area, return 0.

    def LargestPerimeterTriangel(self, A):
        A = sorted(A)[::-1]
        for i in range(len(A)-2):
            if A[i] < A[i+1] + A[i+2]:
                return A[i] + A[i+1] + A[i+2]
        return 0

    def LargestPerimeterTriangel2(self, A):
        return next((sum(A[i:i+3]) for A in [sorted(A)] for i in range(len(A)-3, -1, -1) if sum(A[i:i+2]) > A[i+2]), 0)



#####################################################
# 977. Squares of a sorted array
# Given an array of integers A sorted in non-decreasing order, return an array of the squares of each number, also in sorted non-decreasing order.
    def SquaresSortedArray(self, nums):
        return sorted( [ n*n for n in nums])
        #return list( map(lambda x: x*x, sorted(nums, key = lambda x: abs(x))) )

#####################################################
# 997. Find the town judge
# In a town, there are N people labelled from 1 to N.  There is a rumor that one of these people is secretly the town judge.
#If the town judge exists, then:
#The town judge trusts nobody.
#Everybody (except for the town judge) trusts the town judge.
#There is exactly one person that satisfies properties 1 and 2.
#You are given trust, an array of pairs trust[i] = [a, b] representing that the person labelled a trusts the person labelled b.
#If the town judge exists and can be identified, return the label of the town judge.  Otherwise, return -1.

    def FindJudge(self, N, trust):
        dic, map = collections.defaultdict(), set()
        for i, j in trust:
            if not dic[i]:
                dic[i].append(j)
                map.add(j)     
        return map - dic.keys() + 1

    def FindJudge2(self, N, trust):
        b = set()
        for n in trust:
            b. add(n[0])
        t = list(range(1,N+1))
        return [i for i in t if i not in b] or -1

    def FindJudge3(self, N, trust):
        cnt = [0] * (N + 1)
        for i, j in trust:
            cnt[i] -= 1
            cnt[j] += 1
        for i in range(1, N + 1):
            if cnt[i] == N -1:
                return i
        return -1

    


#####################################################
# 999. Available captures for rook
# On an 8 x 8 chessboard, there is one white rook.  There also may be empty squares, white bishops, and black pawns.  These are given as characters 'R', '.', 'B', and 'p' respectively. Uppercase characters represent white pieces, and lowercase characters represent black pieces.
# The rook moves as in the rules of Chess: it chooses one of four cardinal directions (north, east, west, and south), then moves in that direction until it chooses to stop, reaches the edge of the board, or captures an opposite colored pawn by moving to the same square it occupies.  Also, rooks cannot move into the same square as other friendly bishops.
# Return the number of pawns the rook can capture in one move.

# Just iterate over the board and stop when you see rook
# You will have 4 tasks to do at this point: checking each cardinal direction for facing pawn first
# 4 simple loops and you are done
    def NumRookCaptures(self, board, res=0):
        for i in range(8):
            for j in range(8):
                if board[i][j] == 'R':
                    for x in range(i - 1, -1, -1):
                        if board[x][j] in 'Bp':
                                res += board[x][j] == 'p'
                                break
                    for x in range(i + 1, 8):
                        if board[x][j] in 'Bp':
                            res += board[x][j] == 'p'
                            break
                    for y in range(j - 1, -1, -1):
                        if board[i][y] in 'Bp':
                            res += board[i][y] == 'p'
                            break
                    for y in range(j + 1, 8):
                        if board[i][y] in 'Bp':
                            res += board[i][y] == 'p'
                            break
                    return res


    def NumRookCaptures2(self, board):
            m, n, res = len(board), len(board[0]), 0
            (a, b) = [(a, b) for a in range(m) for b in range(n) if board[a][b] == 'R'][0]
            for j in range(b - 1, -1, -1):
                if board[a][j] == 'p': res += 1
                if board[a][j] != '.': break
            for j in range(b + 1, n):
                if board[a][j] == 'p': res += 1
                if board[a][j] != '.': break
            for i in range(a + 1, m):
                if board[i][b] == 'p': res += 1
                if board[i][b] != '.': break
            for i in range(a - 1, -1, -1):
                if board[i][b] == 'p': res += 1
                if board[i][b] != '.': break
            return res

    def NumRookCaptures3(self, board):
        for i in range(8):
            for j in range(8):
                if board[i][j] == 'R':
                    x0, y0 = i, j
                    i, j = 8, 8 #break the loop
                
        res = 0
        for i, j in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
            x, y = x0 + i, y0 + j
            while 0 <= x < 8 and 0 <= y < 8:
                if board[x][y] == 'p': res += 1
                if board[x][y] != '.': break
                x, y = x + i, y + j
        return res

#####################################################
# 1005. Maxium sum of array after K negations
# Given an array A of integers, we must modify the array in the following way: we choose an i and replace A[i] with -A[i], and we repeat this process K times in total.  (We may choose the same index i multiple times.)
# Return the largest possible sum of the array after modifying it in this way.

    def LargestSumAfterKNegations(self, A, K):
        A.sort()
        i = 0
        while i < len(A) and i < K and A[i] < 0:
            A[i] = -A[i]
            i += 1
        return sum(A) - (K - i) % 2 * min(A) * 2  
        #after run out of the negative elements, if K is still more than i, change the sign of the min(A)
        # back and force, until K runs out, either min(A) with + sign, or min(A) with - sign,
        # if min(A) is positive, then sum(A), otherwise sum(A) - 2*min(A)   

#use heapfy
    def LargestSumAfterKNegations2(self, A, K):
        heapq.heapify(A)
        for _ in range(K):
            heapq.heappush(A, -heapq.heappop(A))
        return sum(A)

    

#####################################################
# 1013. Partition array into three parts with equal sum
# Given an array A of integers, return true if and only if we can partition the array into three non-empty parts with equal sums.
# partition goes in order of 1 to len(A)

    def CanThreePartsEqualSum(self, A):
        if not sum(A)%3:
            return False
        expected, total, count = sum(A)//3, 0, 0
        for a in A:
            total += a
            if total == expected:
                count += 1
                total = 0
        return count == 3
    
    def CanThreePartsEqualSum2(self, A):
        a = list(itertools.accumulate(A))
        if not sum(A) % 3:
            return False
        one = sum(A) // 3
        two = one * 2
        if one in a:
            l_index = a.index(one)
        else:
            return False
        if two in a[l_index:]:
            return True
        else:
            return False
        


# partiton may not go in the order of A ? 
    def CanThreePartsEqualSum3(self, A):
        if not sum(A)%3:
            return False
        expected, total, count = sum(A)//3, 0, 0
        while A:
            res = []
            for a in A:
                total += a
                if total == expected:
                    count += 1
                    total = 0
                elif total > expected:
                    res.append(a)
                    total -= a
                else:
                    continue
            A = res
        return count == 3






#####################################################
# 1025. Divisor game
# Choosing any x with 0 < x < N and N % x == 0.
# Replacing the number N on the chalkboard with N - x.
# Also, if a player cannot make a move, they lose the game.
# Return True if and only if Alice wins the game, assuming both players play optimally.

#Conclusion
#If N is even, can win.
#If N is odd, will lose.


    def DivisorGame(self, N):
        return N%2 == 0
    
# use dynamic programming

    def DivisorGame2(self, N):
        dp = [False for i in range(N+1)]
        for i in range(1, N+1):
            for x in range(1, i//2 + 1):
                if (i%x == 0) and (not dp[i-x]):
                    dp[i] = True
                    break
    
    def DivisorGame3(self, N):
        dp = [False]*(N+1)
        for i in range(2, N+1):
            dp[i] = not all(dp[i-j] for j in range(1, int(i**0.5+1)) if not i%j)
        return dp[N]

    def DivisorGame4(self, N):
        return self.solve(N, {})

    def solve(self, N, m):
        if N in m:
            return m[N]
        if N == 1:
            return False
        m[N] = any(not self.solve(N-i, m) for i in range(1,N) if N%i == 0)
        return m[N]


#####################################################
# 1029. Two city scheduling
# There are 2N people a company is planning to interview. The cost of flying the i-th person to city A is costs[i][0], and the cost of flying the i-th person to city B is costs[i][1].
# Return the minimum cost to fly every person to a city such that exactly N people arrive in each city.
# Example 1: Input: [[10,20],[30,200],[400,50],[30,20]] Output: 110
# Explanation: 
#The first person goes to city A for a cost of 10.
#The second person goes to city A for a cost of 30.
#The third person goes to city B for a cost of 50.
#The fourth person goes to city B for a cost of 20.
#The total minimum cost is 10 + 30 + 50 + 20 = 110 to have half the people interviewing in each city.

    def TwoCitySchedulingCost(self, costs):
        N = len(costs) // 2
        costs.sort(key = lambda x: x[0]-x[1])
        return sum(i[0] for i in costs[:N]) + sum(j[1] for j in costs[N:])
        #return sum(a if i < N else b for i, (a, b) in enumerate(costs))

#use dynamic programming

    def TwoCitySchedulingCost2(self, costs):
        N = len(costs) // 2
        dic = {(0, 0): 0}
        for cost in costs:
            newdic = {}
            for i, j in dic.keys():  # key (i, j) => ith people in the A city group, jth in the B city group
                if i + 1 <= N:
                    newdic[i+1, j] = min(newdic.get((i+1, j), float('inf')), dic[i,j] + cost[0])
                if j + 1 <= N:
                    newdic[i, j+1] = min(newdic.get((i, j+1), float('inf')), dic[i,j] + cost[1])
            dic = newdic
        return dic[N, N]


#####################################################
# 1030. Matrix cells in distance order
# We are given a matrix with R rows and C columns has cells with integer coordinates (r, c), where 0 <= r < R and 0 <= c < C.
# Additionally, we are given a cell in that matrix with coordinates (r0, c0).
# Return the coordinates of all cells in the matrix, sorted by their distance from (r0, c0) from smallest distance to largest distance.  Here, the distance between two cells (r1, c1) and (r2, c2) is the Manhattan distance, |r1 - r2| + |c1 - c2|.  (You may return the answer in any order that satisfies this condition.)

# Input: R = 2, C = 3, r0 = 1, c0 = 2
# Output: [[1,2],[0,2],[1,1],[0,1],[1,0],[0,0]]
# Explanation: The distances from (r0, c0) to other cells are: [0,1,1,2,2,3]
# There are other answers that would also be accepted as correct, such as [[1,2],[1,1],[0,2],[1,0],[0,1],[0,0]].

    def MatrixCellDistanceOrder(self, R, C, r0, c0):
        return sorted([(i, j) for i in range(R) for j in range(C)], key= lambda x: abs(x[0]-r0)+abs(x[1]-c0))


#####################################################
# 1042. Flower planting with no adjacent
#You have N gardens, labelled 1 to N.  In each garden, you want to plant one of 4 types of flowers.
#paths[i] = [x, y] describes the existence of a bidirectional path from garden x to garden y.
#Also, there is no garden that has more than 3 paths coming into or leaving it.
#Your task is to choose a flower type for each garden such that, for any two gardens connected by a path, they have different types of flowers.
#Return any such a choice as an array answer, where answer[i] is the type of flower planted in the (i+1)-th garden.  The flower types are denoted 1, 2, 3, or 4.  It is guaranteed an answer exists.

#Input: N = 4, paths = [[1,2],[2,3],[3,4],[4,1],[1,3],[2,4]]
#Output: [1,2,3,4]

    def GardenNoAdjacentPlant(self, N, paths):
        res = [0] * N
        dic = {}
        for x, y in paths:
            if x in dic: 
                dic[x].append(y)
            else:
                dic[x] = [y]
            if y in dic:
                dic[y].append(x)
            else:
                dic[y] = [x]
        for k, v in dic.items():
            s = set()
            for i in v:
                s.add(res[i - 1])
            res[k-1] = list({1, 2, 3, 4} - s)[0]
            s.clear()
        
        return [ i if i != 0 else 1 for i in res]

    def GardenNoAdjacentPlant2(self, N, paths):
        res = [0]* N
        G = [ [] for i in range(N)]
        for x, y in paths:
            G[x-1].append(y-1)
            G[y-1].append(x-1)
        for i in range(N):
            res[i] = ({1,2,3,4} - {res[j] for j in G[i]}).pop()
        return res

    def GardenNoAdjacentPlant3(self, N, paths):
        dic = collections.defaultdict(list)
        res = [0] * (N+1)
        psorted = [sorted(p) for p in paths]
        print(psorted)
        for p in psorted:
            dic[p[0]].append(p[1])
            dic[p[1]].append(p[0])
        for i in range(1,N+1):
            res[i] = ({1,2,3,4} - {res[j] for j in dic[i]}).pop()
        return res[1:]
    

#####################################################
# 1046. Last stone weight
#We have a collection of rocks, each rock has a positive integer weight.
#Each turn, we choose the two heaviest rocks and smash them together.  Suppose the stones have weights x and y with x <= y.  The result of this smash is:

#If x == y, both stones are totally destroyed;
#If x != y, the stone of weight x is totally destroyed, and the stone of weight y has new weight y-x.
#At the end, there is at most 1 stone left.  Return the weight of this stone (or 0 if there are no stones left.)

    def LastStoneWeight(self, stones):
        stones = sorted(stones)
        for _ in range(len(stones) - 1):
            x, y = stones.pop(), stones.pop()
            if abs(x - y):
                bisect.insort(stones, abs(x - y))
        return stones.pop()

    def LastStoneWeight2(self, stones):
        if len(stones) == 1:
            return stones[0]
        stones.sort()
        x  = abs(stones.pop() - stones.pop())
        return self.LastStoneWeight2([x] + stones) if x else self.LastStoneWeight2(stones)



#####################################################
# 1051. Height checker
# Students are asked to stand in non-decreasing order of heights for an annual photo.
# Return the minimum number of students not standing in the right positions.

    def HeightChecker(self, heights):
        return sum(h1 != h2 for h1, h2 in zip(heights, sorted(heights)))







    



#####################################################
# 1086. High five
# Given a list of scores of different students, return the average score of each student's top five scores in the order of each student's id.
# Each entry items[i] has items[i][0] the student's id, and items[i][1] the student's score. The average score is calculated using integer division.
# Input: [[1,91],[1,92],[2,93],[2,97],[1,60],[2,77],[1,65],[1,87],[1,100],[2,100],[2,76]]
# Output: [[1,87],[2,88]]

    def HighFive(self, nums):
        sd = list(set(map(lambda x: x[0], nums)))
        r1 = [ [y[1] for y in nums if y[0]==x] for x in sd ]
        r11 = [ sorted(n, reverse=True)[:5] for n in r1 ]
        r2 = [ int(sum(n)/len(n)) for n in r11 ]
        return [ [sd[i], r2[i]] for i in range(len(sd)) ]

#nums= [[1,91],[1,92],[2,93],[2,97],[1,60],[2,77],[1,65],[1,87],[1,100],[2,100],[2,76]]
#a=solution()
#a.HighFive(nums)

#####################################################
# 1089. Duplicate zeros
# Given a fixed length array arr of integers, duplicate each occurrence of zero, shifting the remaining elements to the right.
# use in-place storage
    def DuplicateZeros(self, nums):
        i= 0
        while i < len(nums):
            if nums[i] == 0:
                nums.insert(i, 0)
                nums.pop()  #remove the last element
                i += 1
            i += 1





#####################################################
# 1103. Distribute candies to people
# We distribute some number of candies, to a row of n = num_people people in the following way:
# We then give 1 candy to the first person, 2 candies to the second person, and so on until we give n candies to the last person.
# Then, we go back to the start of the row, giving n + 1 candies to the first person, n + 2 candies to the second person, and so on until we give 2 * n candies to the last person.
# This process repeats (with us giving one more candy each time, and moving to the start of the row after we reach the end) until we run out of candies.  The last person will receive all of our remaining candies (not necessarily one more than the previous gift).
# Return an array (of length num_people and sum candies) that represents the final distribution of candies.

    def DistributeCandiesToPeople(self, candies, n):
        res = [0]*n
        i = 0
        while candies > 0:
            res[i % n] += min(candies, i+1)
            candies -= i + 1
            i += 1
        return res





#####################################################
# 1122. Relative sort array
# Given two arrays arr1 and arr2, the elements of arr2 are distinct, and all elements in arr2 are also in arr1.
# Sort the elements of arr1 such that the relative ordering of items in arr1 are the same as in arr2.  Elements that don't appear in arr2 should be placed at the end of arr1 in ascending order.

    def RelativeSortArray(self, arr1, arr2):
        return sorted(arr1, key = lambda x: arr2.index(x) if x in arr2 else len(arr2) + x)

    def RelativeSortArray2(self, arr1, arr2):
        last = len(arr2)
        d = { k: i for i, k in enumerate(arr2)}
        return sorted(arr1, key=lambda elem: d.get(elem, last+elem))

#####################################################
# 1164. Fixed point
# Given an array A of distinct integers sorted in ascending order, 
# return the smallest index i that satisfies A[i] == i.  Return -1 if no such i exists.
# use binary search

    def FixedPoint(self, A):
        left, right = 0, len(A)-1
        while left < right:
            mid = (left + right)//2
            if A[mid] >= mid:
                right = mid
            else:
                left = mid + 1
        if A[left] == left:
            return left
        return -1































#####################################################
# 5000. Greater common divider
#

    def GreatCommonDivider(self, seq):
        aseq = [abs(e) for e in seq]
        amat = [ [0]*i + [1] + [0]*(len(seq)-i-1) for i in range(len(seq)) ]  #initialize the divider list for each number
        seq2 = [ (v, i) for i, v in enumerate(aseq) ]
        nz = [ (v, i) for v, i in seq2 if v != 0]  #remove number with zero value 
        while len(nz) != 1:
            mv, mi = min(nz)  #pull the minimum number
            seq3 = [ (v%mv, v//mv, i) if i != mi else (v, 0, i) for v, i in seq2 ]
            amat = [ [v - n*u for v, u in zip(amat[i], amat[mi])] for v, n, i in seq3 ]
            seq2 = [ (v, i) for v, n, i in seq3 ]
            nz = [ (v, i) for v, i in seq2 if v != 0 ]  #remove evenly divided numbers
        
        mat = [ [-a if v < 0 else a for a, v in zip(ax, seq)] for ax in amat ]
        vert = [ e for i, e in enumerate(mat) if i !=mi ]

        return mv, mat[mi], vert

#if __name__ == *__main__*
#    seq = [4, 26, -20, 94]
#    res = GreatCommonDivider(seq)
#    print(res)

#seq = [4, 26, -20, 94]
#a.solution()
#a.GreatCommonDivider(seq)
#(2, [-6, 1, 0, 0], [[13, -2, 0, 0], [-5, 0, -1, 0], [-17, -1, 0, 1]])

# How to call main()
#def mainp():
#    print("Hello World!")

#if __name__ == "__main__":
#    mainp()
    
