import re
import itertools
import collections
#from string import maketrans #1009
import math
from functools import lru_cache # for 509


#####################################################
# 7. Reverse integer
# use bit operation - mod and carry
class solution:
    def ReverseInt(self, numb):
        x = abs(numb)
        r = 0
        while x > 0:
            y = int(x % 10)  # x, y = divmod(x, 10)
            r = r*10 + y
            x = int(x/10)
            if (r > 2**31-1) or (r < 2**31): #overflow case
                print("overflow")
                break
            return (r*(-1 if numb<0 else 1))

    def ReverseInt2(self, nums):
        a, res = abs(nums), 0
        while a > 0:
            a, b = divmod(a, 10)
            res = res*10 + b
            if res > 2**31-1 or res < -2**31:
                return ("overflow")
        return res * (-1 if nums < 0 else 1)



#a.ReverseInt(-123450)

#####################################################
# 8. String to integer
# use int() and regular expression
    def atoi(self, s):
        s = s.strip()
        s = re.findall('(^[\+\-0]*\d+)\D*',s)  # s = re.findall("(^[\+\-0]*\d+)\D*", s)
        try:
            numb = int(''.join(s))
            if numb > 2**31-1:
                return 2**31-1
            elif numb < -2**31:
                return -2**31
            else:
                return numb
        except:
            return 0

#a.atoi("   -42")
#a.atoi("4193 with words")
#a.atoi("words with 4193")
#a.atoi("-91283472332")


#####################################################
# 9. Palindrome number
# An integer is a palindrome when it reads the same backward as forward.
# use str() to convert to string

    def PalindromNumb(self, numb):
        if (numb<0):
            return False
        s = str(numb)
        if(len(s)%2 == 0):
            r = int(len(s)/2)
        else:
            r = int(len(s)/2) + 1
        if s[0: int(len(s)/2)][::-1] == s[r:]:
            return True
        else:
            return False

#n = 10246064201
#a.PalindromNumb(n)


#####################################################
# 12. Integer to Roman
# use number to char mapping

    def IntToR(self, n):
        # rc - roman char
        rc = ["I","IV","V","IX","X","XL","L","XC","C","CD","D","CM","M"]
        mp =  [1,4,5,9,10,40,50,90,100,400,500,900,1000]
        if n <= 0 or n >  3999:
            print("number out of range")
            return
        k = len(mp)-1
        s = ""
        while n > 0:
            if (n >= mp[k]):
                n = n - mp[k]
                s = s + rc[k]
            else:
                k -= 1
        return

#n = 1994
#a.IntToR(n)


#####################################################
# 12. Roman to Integer
# use number to char mapping
# note the way to handle the cases with two-character: IV, IX, XL, XC, CD, CM
    def RToInt(self, s):
        # ri - dict of roman char => value
        ri = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)
        prev = "M"
        numb = 0
        for c in s:
            if (ri[c] > ri[prev]): # cases of IV, IX, XL, XC, CD, CM
                numb = numb + (ri[c]-2*ri[prev])  
                # IV = V - I, in addition, - I (minus the I added in previous step) = V - 2 * I
            else:
                numb = numb + ri[c]
            prev = c


#strs = "MCMXCIV"
#a.RToInt(strs)

#####################################################
# 1. Two Sum - return index + value
# use hashmap

    def TwoSum(self, n, target):
        mp, res = {}, []
        for i in range(len(n)):
            m = target - n[i]
            if m in mp:
                #return (i, n[i], mp.get(m), m)
                res.append([m, n[i]])
            else:
                mp[n[i]]= i
        print(mp)
        return res

#n = [2, 7, 11, 15]
#target = 9
#a.TwoSum(n, target)

    def TwoSumII(self,nums,target):
        for i in range(len(nums)):
            l,r,tmp=i+1,len(nums)-1,target-nums[i]
            while l<=r:
                mid = (l+r)//2
                if nums[mid]==tmp:
                    return [i+1,mid+1]
                elif nums[mid]<tmp:
                    l = mid+1
                else:
                    r = mid-1
            return []

def tsum(L,target):
    a=sorted(L)
    left = 0
    right = len(L)-1
    while left < right:
        right1 = right
        while a[left]+a[right1] > target:
            right1 -= 1
        if a[left]+a[right1] == target:
            return True
        elif right1 < right:
            right = right1+1
        left += 1
    return False


#####################################################
# 15. Three Sum - sum() = 0, based on 2Sum
# sort, brute force, each loop is applied with 2Sum
# 
    def ThreeSum(self, n):
        res = []
        if len(n) < 3:
            return res
        if n == [0]*len(n):
            return [[0, 0, 0]]
        n.sort()
        if n[-1] <= 0:  # n[-1]==n[len(n)-1]
            return res
        while n[0] <= 0:
            twosum = self.TwoSum(n[1:], -n[0])
            for t in twosum:
                t.append(n[0])
                res.append(t)
            n = n[1:]
        return (list(set([tuple(i) for i in res])))

#s = [-9, -1, 0, 1, 2, -1, -4]
#a.ThreeSum(s)


#####################################################
# 16. 3Sum closest - sum() -> target
# sort, use binary search, from ends (left, right) to the middle element
    def ThreeSumClosest(self, n, target):
        n.sort()
        res = sum(n[:3])
        for i in range(len(n)):
            l, r = i+1, len(n)-1
            while l < r:
                s = sum(n[i], n[l], n[r])
                if abs(s - target) < abs(res - target):
                    res = s
                if s < target:
                    l += 1
                elif s > target:
                    r -= 1
                else: 
                    return res
        return res

#nums = [-1, 2, 1, -4, 7, 8]
#target = 1
#a.ThreeSumClosest(nums,target)

#####################################################
# 17. Letter combinations of a phone number
# output all the letter combos
# 
    def CombPhoneNumb(self, sd):
        phone = {'2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']}
        
        def CombChar(subcomb, dgt):
            if (len(dgt) == 0):
                combstr.append(subcomb)
            else:
                for c in phone[dgt[0]]:
                    CombChar(subcomb+c, dgt[1:])
        
        combstr = []
        if sd:
            sd = sd.replace("1", "").replace("0","") #exclde 0 and 1 cases
            CombChar("", sd)
        return combstr


#dt="0213"
#a.CombPhoneNumb(dt)

    def CombPhoneNumb2(self, sd):
        phone2 = {"1":"", "2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs","8":"tuv","9":"wxyz","0":""}
        if sd:
            sd = sd.replace("1", "").replace("0","")
        res = [""]
        for _d in sd:
            lst = phone2[_d]
            nres = []
            for c in lst:
                for r in res:
                    nres.append(r+c)
            res = nres
        return res

#dt="0213"
#a.CombPhoneNumb2(dt)

#####################################################
# 18. 4Sum - sum() = target
# based on twoSum

    def FourSum(self, n, target):
        twosum = collections.defaultdict(list)  #dict value is a list
        res = set()
        for (i1, v1), (i2, v2) in itertools.combinations(enumerate(n), 2): # i is the index, v is the value
            twosum[v1+v2].append({i1, i2})
        for t in list(twosum.keys()):
            if not twosum[target - t]:
                continue  #go to the next for t loop
            for p1 in twosum[t]:
                for p2 in twosum[target - t]:
                    if p1.isdisjoint(p2): #no commen elements, supported by Python 3 only
                        res.add(tuple(sorted(n[i] for i in p1 | p2)))
            del twosum[t]  #remove this processed key
        return [list(r) for r in res]
        

#note: isdisjoint() is supported in Python 3 only
#for Python 2, use either the two funcs below:
    def dicts_disjoint(self, a, b):
        return not any(k in b for k in a)
#Or:

    def dicts_disjoint2(self, a, b):
        return all(k not in b for k in a)

#n = [1, 0, -1, 0, -2, 2]
#target = 0
#a.FourSum(n,target)

#####################################################
# 29. Divide two integers without using the "/"
# use "minus the numerator", slow method
    def DivInt(self, a, b):
        MAX_INT, MIN_INT=2**31-1, -2**31
        if(a>MAX_INT or a<MIN_INT or b>MAX_INT or b<MIN_INT or b==0):
            print("out of range")
            return
        count = 0
        a1, b1 = abs(a), abs(b)
        while a1 > b1:
            a1 -= b1
            count += 1
        if (a < 0 and b > 0) or (a>0 and b<0):
            count = 0 - count
        return count

# use "minus the looping-doubled numerator", faster method
    def DivInt2(self, a, b):
        MAX_INT, MIN_INT=2**31-1, -2**31
        if(a>MAX_INT or a<MIN_INT or b>MAX_INT or b<MIN_INT or b==0):
            print("out of range")
            return
        count = 0
        a1, b1 = abs(a), abs(b)
        while a1 > b1:
            x, i = b1, 1
            while a1 > x + x:
                x += x
                i += i
            a1 = a1 - x
            count += i
        
        if (a < 0 and b > 0) or (a>0 and b<0):
            count = 0 - count
        return count


#a, b = 10, 3
#a.DivInt(a, b)
#a, b = 7, -3
#a.DivInt(a, b)

#####################################################
# 35. Count and Say
# 1.     1
#2.     11
#3.     21
#4.     1211
#5.     111221
#1 is read off as "one 1" or 11.
#11 is read off as "two 1s" or 21.
#21 is read off as "one 2, then one 1" or 1211.


    def CountandSay(self, n):
        res="1"
        for _ in range(n-1):
            ch, tmp, count = res[0], '', 0
            for s in res:
                if s == ch:
                    count += 1
                else:
                    tmp += str(count) + ch
                    ch = s
                    count = 1
            res = tmp + str(count) + ch
        return res

#n=5
#a=Solution()
#a.CountandSay(n)

#####################################################
# 66. Plus one
#Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
# use divmod

    def PlusOne(self, nums):
        carry = 1
        for i in range(len(nums)-1, -1, -1):  #from the lowest digit
            carry, nums[i] = divmod(nums[i]+carry, 10)
            if carry == 0:
                return nums
        return [1] + nums

#nums = [4,3,2,1]
#a=Solution()
#a.PlusOne(nums)

#####################################################
# 67. Add binary
# Given two binary strings, return their sum (also a binary string)
# use divmod

    def AddBinary(self, a, b):
        i, j, carry, res = len(a)-1, len(b)-1, 0, ""
        while i >= 0  or j >= 0 or carry:
            if i >= 0:
                carry += int(a[i])
                i -= 1
            if j >=0:
                carry += int(b[j])
                j -= 1
            carry, d = divmod(carry, 2)
            res = str(d) + res

#b1,b2="1010","1011"
#a=Solution()
#a.AddBinary(b1,b2)

#####################################################
# 69. Sqrt(x)
# use binary search

    def sqrt(self, x):
        l, r = 0, x
        while l <= r:
            mid = (l + r)//2
            if mid * mid <= x < (mid+1) * (mid+1):
                return mid
            elif mid * mid < x:
                l = mid + 1
            else:
                r = mid
    
#x=8
#a=Solution()
#a.sqrt(x)

#####################################################
# 118. Pascal's triangle
# Given a non-negative integer numRows, generate the first numRows of Pascal's triangle.

    def PascalTriangle(self, n):
        res = [ [1 for _ in range(i+1)] for i in range(n)]
        for i in range(2, n):
            for j in range(1, i):
                res[i][j] = res[i-1][j-1] + res[i-1][j]
        return res
    
#n=5
#a=solution()
#a.PascalTriangle(n)
#[[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]

#####################################################
# 119. Pascal's triangle II
# Given a non-negative index k where k ≤ 33, return the kth index row of the Pascal's triangle.

    def PascalTriangleII(self, k):
        res = [ [1 for _ in range(i+1)] for i in range(k+1)]
        for i in range(2, k+1):
            for j in range(1, i):
                res[i][j] = res[i-1][j-1] + res[i-1][j]
        return res[-1]

#k=3
#a=solution()
#a.PascalTriangle(k)
#[1, 3, 3, 1]


#####################################################
# 136. Single number
# Given a non-empty array of integers, every element appears twice except for one. Find that single one.
    def SingleNumber(self, nums):
        res = 0
        for n in nums:
            res ^= n
        return res

    def SingleNumber2(self, nums):
        return 2*sum(set(nums))-sum(nums)



    


#####################################################
# 168. Excel sheet column title
# use divmod

    def ExcelColumnTitle(self, n):
        dic = [ chr(x) for x in range(ord('A'), ord('Z')+1) ]
        res = ''
        while n > 0:
            n, rem = divmod(n-1, 26)
            res = dic[rem] + res
        return res

#n=701
#a.ExcelColumnTitle(n)


#####################################################
# 169. Majority element
# Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
# You may assume that the array is non-empty and the majority element always exist in the array.


#Boyer Moore voting algorithm:
    def MajorityElement(self, nums):
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)
        return candidate

    def MajorityElement2(self, nums):
        return sorted(nums)[len(nums)/2]

    def MajorityElement3(self, nums):
        return collections.Counter(nums).most_common(1)[0][0]
    

#####################################################
# 171. Excel sheet column number - based on input of char(s)
# output the column number

    def ExcelColumnNumber(self, s):
        res = 0
        for c in s:
            res = res*26 + ord(c) - ord('A')+1
        return res

#strs='ZY'
#a=solution()
#a.ExcelColumnNumber(strs)

#####################################################
# 172. Factorial trailing zeros
# Given an integer n, return the number of trailing zeroes in n!
# zero is caused by 2*5, 4*5, 6*5, 8*5, 10, 20...
# 5**2 will have the additional 1 zero

    def FactorialTrailingZeroes(self, n):
        k, count = 5, 0
        while k <= n:
            count += n//k
            k = k*5
        return count

#n=5
#a.FactorialTrailingZeroes(n)

#####################################################
# 190. Reverse bits
# Reverse bits of a given 32 bits unsigned integer
# use divmod
    def ReverseBits(self, n):
        res = 0
        for _ in range(32):
            res = res*2 + n%2
            n = n//2
        return res

#faster than using the operator above
    def ReverseBits2(self, n):
        res = 0
        for _ in range(32):
            res = (res << 1) + (n & 1)
            n >>= 1
        return res

#####################################################
# 191. Number of 1 bits
# also known as the Hamming weight
# use divmod(n, 2)

    def NumberOneBits(self, n):
        res = 0
        for _ in range(32):
            res += n%2
            n = n//2


#####################################################
# 202. Happy number
# A happy number: starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.
# use str(n)
    def HappyNumber(self, n):
        dic = {}
        while n != 1 and n not in dic: # use dic to detect the cycle of the n
            dic[n] = dic.get(n, 0) + 1  # put 1 to all dic[n]
            n = sum(int(c)**2 for c in str(n))
        return n == 1

#n=19
#a.HappyNumber(n)


#####################################################
# 204. Count primes
# Count the number of prime numbers less than a non-negative number, n.
# use sqrt(n)

    def CountPrime(self, n):
        if n <= 2:
            return 0
        res = [1] * n
        res[0] = res[1] = 0
        for i in range(2, int(n**1/2) + 1):
            if res[i]:
                res[i*i:n:i]=[0]*len(res[i*i:n:i]) 
                #i is the common divider => assign those positions which can be divided by i, means they are not prime
        return sum(res)

#use i as common divider but loop in range of (2, n), so it's slower
        
    def CountPrime2(self, n):
        if n <= 2:
            return 0
        res = [True]*n
        res[0] = res[1] = False
        for i in range(2, n):
            if res[i] == True:
                for j in range(2, (n-1)//i+1):
                    res[i*j] = False
        return sum(res)



    def CountPrime4(self, n):
        if n <= 2: 
            return 0
        res = []
        for num in range(2, n+1):
            for i in range(2, int(num**0.5) + 1):
                if num % i == 0:
                    break
            else:
                res.append(num)
        return len(res)

#optimal
    def CountPrime5(self, n):
        if n <= 2: 
            return 0
        res = [2]
        for num in range(3, n, 2):
            if all(num%i for i in range(3, int(math.sqrt(num))+1, 2)):
                res.append(num)
        return len(res)

#n=10
#a.CountPrimes(n)
#a.CountPrimes2(n)

    def is_prime(self, N):
        return all(N%j for j in range(2, int(N**0.5) + 1)) and N > 1

#####################################################
# 231. Power of two
# Given an integer, write a function to determine if it is a power of two
# use n & n-1

    def PowerofTwo(self, n):
        return n > 0 and not (n & n-1)

    def PowerofTwo2(self, n):
        if n <=0: 
            return False
        while n%2 == 0:
            n = n//2
        return n == 1

#n=218
#a.PowerofTwo2(n)

#####################################################
# 246. Strobogrammatic number
# A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
#Example 1: Input:  "69" Output: true
#Example 2: Input:  "88" Output: true
#Example 3: Input:  "962" Output: false
# use binary search, process from both ends (left and right) to the center

    def StrobogrammaticNumber(self, n):
        num = str(n)
        dic, l, r = {'0':'0','1':'1','6':'9','8':'8','9':'6'}, 0, len(num)-1
        while l <= r:
            if num[l] not in dic or dic[num[l]] != num[r]:
                return False
            l += 1
            r -= 1
        return True

#num=10869801
#a.StrobogrammaticNumber(num)

#####################################################
# 258. Add digits
#Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

#Example: Input: 38 Output: 2 
#Explanation: The process is like: 3 + 8 = 11, 1 + 1 = 2.
# use divmod

# recursive

    def AddDigits(self, n):
        if 0 <= n <= 9:
            return n
        tmp = 0
        while n:
            tmp += n%10
            n //= 10
        return self.AddDigits(tmp)

# without recursion or interation
# n = 456 4*(99+1)+5*(9+1)+6=(4+5+6)+9*m=15+9*m=(1+5)+9*m2....

    def AddDigits2(self, n):
        if 0 <= n <= 9:
            return n
        res = n%9
        return res if res != 0 else 9

#n=38
#a.AddDigits2(38)

#####################################################
# 263. Ugly number
# Ugly numbers are positive numbers whose prime factors only include 2, 3, 5.

    def UglyNumber(self, n):
        if n == 0:
            return False
        while n%2 == 0:
            n = n//2
        while n%3 == 0:
            n = n//3
        while n%5 == 0:
            n = n//5
        return n == 1

#n=14
#a.UglyNumber(n)
#n=75*12
#a.UglyNumber(n)


#####################################################
# 268. Missing number
# Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
# use bit operation

    def MissingNumber(self, nums):
        n = len(nums)
        return n*(n+1)/2 - sum(nums)

    def MissingNumber2(self, nums):
        res = 0
        for i in range(len(nums)+1):
            res = res^i
        for n in nums:
            res = res^n
        return res
    
    def MissingNumber3(self, nums):
        return [x for x in range(1, len(nums)+1) if x not in nums]

    def MissingNumber4(self, nums):
        return set(list(range(1, len(nums)+1))) - set(nums)

#nums=[9,6,4,2,3,5,7,0,1]
#a=solution()
#a.MissingNumber2(nums)

#####################################################
# 339. Nested list weight sum
# Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

    def NestedListWeightSum(self, nestedlist):
        depth, res = 1, 0
        while nestedlist:
            res += depth*sum([i.getInteger() for i in nestedlist if i.isInteger()])
            newlist = []
            for i in nestedlist:
                if not i.isInteger():
                    newlist += i.getList()
            nestedlist = newlist
            depth += 1
        return res
    
#####################################################
# 371. Sum of two integers
# Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.
#Input: a = -2, b = 3
#Output: 1

    def SumOfTwoIntegers(self, a, b):
        max_int = 0x7FFFFFFF
        min_int = 0x80000000
        mask = 0x100000000 # bin(1 << 8)
        while b:
            a, b = (a ^ b) % mask, ((a & b) << 1) % mask
        return a if a <=  max_int else ~((a % min_int) ^ max_int)

    def SumOfTwoIntegers2(self, a, b):
        max_int = 0x7FFFFFFF
        #min_int = 0x80000000
        mask = 0xFFFFFFFF # mask to get last 32 bits
        while b:
            # ^ get different bits and & gets double 1s, << moves carry
            a, b = (a ^ b) & mask, ((a & b) << 1) & mask
        # if a is negative, get a's 32 bits complement positive first
        # then get 32-bit positive's Python complement negative
        return a if a <= max_int else ~(a ^ mask)

# add 
    def Add(self, x, y): 
    
        # Iterate till there is no carry  
        while (y != 0): 
        
            # carry now contains common 
            # set bits of x and y 
            carry = x & y 
    
            # Sum of bits of x and y where at 
            # least one of the bits is not set 
            x = x ^ y 
    
            # Carry is shifted by one so that    
            # adding it to x gives the required sum 
            y = carry << 1
        
        return x 
# subtraction

    def Subtract(self, x, y): 
  
        # Iterate till there is no carry 
        while y: 
        
            # borrow contains common  
            # set bits of y and unset 
            # bits of x 
            borrow = (~x) & y
            
            # Subtraction of bits of x 
            # and y where at least one 
            # of the bits is not set 
            x = x ^ y 
    
            # Borrow is shifted by one  
            # so that subtracting it from  
            # x gives the required sum 
            y = borrow << 1
        
        return x 





#####################################################
# 412. Fizz Buzz
# output 1 to n, but for %3 == 0, use Fizz and for %5 == 0, use Buzz

    def fizzBuzz(self, n):
        return ['Fizz' * (not i % 3) + 'Buzz' * (not i % 5) or str(i) for i in range(1, n+1)]


#####################################################
# 414. Third maximum number
# Given a non-empty array of integers, return the third maximum number in this array. If it does not exist, return the maximum number. The time complexity must be in O(n).

    def ThirdMaximumNumber(self, nums):
        v = [float('-inf'), float('-inf'), float('-inf')]
        for num in nums:
            if num not in v:
                if num > v[0]: v = [num, v[0], v[1]]  # v[0] is the prev maximum
                elif num > v[1]: v = [v[0], num, v[1]]
                elif num > v[2]: v = [v[0], v[1], num]
        return max(nums) if float('-inf') in v else v[2]

    def ThirdMaximumNumber2(self, nums):
        m1 = m2 = m3 = -float('inf')
        for num in nums:
            if num == m1 or num == m2 or num == m3: 
                continue
            if num > m1: m1, m2, m3 = num, m1, m2
            elif num > m2: m2, m3 = num, m2
            elif num > m3: m3 = num
        if m2 == -float('inf'):
            return m1
        return m3
    

    

#####################################################
# 461. Hamming distance
# The Hamming distance between two integers is the number of positions at which the corresponding bits are different.
# use bit operation ^
    def HammingDistance(self, x, y):
        x = x ^ y
        y = 0
        while x:
            y += 1
            x = x & (x - 1)  # mask the non-zero bit
        return y

    def HammingDistance2(self, x, y):
        return bin(x ^ y).count('1')

# We can find the i-th bit (from the right) of a number by dividing by 2 i times, then taking the number mod 2.
# Using this, lets compare each of the i-th bits, adding 1 to our answer when they are different.

    def HammingDistance3(self, x, y):
        res = 0
        while x or y:
            res += (x % 2) ^ (y % 2)
            x /= 2
            y /= 2
        return res

#####################################################
# 476. Number complement
# Given a positive integer, output its complement number. The complement strategy is to flip the bits of its binary representation.
# use divmod or bit operation


# Find the power of 2 number larger than num. 
# Then i-1 will unset the last set bit and set all bits to its right to 1. Then just XOR with this number.

    def FindComplement(self, num):
        i = 1
        while i <= num:
            i = i << 1
        return (i - 1) ^ num

    def FindComplement2(self, num):
        mask = 1
        while mask < num:
            mask = (mask << 1) + 1
        return mask ^ num

    def FindComplement3(self, num):
        mask = 1 << (len(bin(num)) - 2 ) # mask = 1 << bin(num).bit_length()
        return (mask - 1) ^ num

    def FindComplement4(self, num):
        x = bin(num)[2:]
        return int(''.join([str(abs(1-int(i))) for i in x]),2)

#print(1<< len(bin(8))-2)
#print(bin(8),bin(16))
#16
#0b1000 0b10000


#####################################################
# 492. Construct the rectangle
#1. The area of the rectangular web page you designed must equal to the given target area.
#2. The width W should not be larger than the length L, which means L >= W.
#3. The difference between length L and width W should be as small as possible.
# You need to output the length L and the width W of the web page you designed in sequence.
    def ConstructRectangle(self, area):
        mid = int(math.sqrt(area))
        while mid > 0:
            if area % mid == 0:
                return [int(area / mid), mid]
            mid -= 1

    def ConstructRectangle2(self, area):
        W = int(math.sqrt(area))
        while area % W and W >= 1:
            W -= 1
        return [ area // W, W]

    


#####################################################
# 496. Next greater element I
# You are given two arrays (without duplicates) nums1 and nums2 where nums1’s elements are subset of nums2. Find all the next greater numbers for nums1's elements in the corresponding places of nums2.
# The Next Greater Number of a number x in nums1 is the first greater number to its right in nums2. If it does not exist, output -1 for this number.

    def NextGreaterElement(self, nums1, nums2):
        d, res = {}, [-1]*len(nums1)
        for i, n1 in enumerate(nums1):
            d[n1] = i
        stack = []
        for n2 in nums2:
            while stack and stack[-1] < n2:
                top = stack.pop()
                if top in d:
                    res[d[top]] = n2
            stack.append(n2)
        return res
    
    def NextGreaterElement2(self, nums1, nums2):
        def helper(n1):
            for n2 in nums2[nums2.index(n1):]:
                if n2 > n1:
                    return n2
            return -1
        return map(helper, nums1)

#####################################################
# 509. Fibonacci number
# use dynamic programming

    def FibonacciNumber(self, n):
        return self.FibonacciNumber(n-1) + self.FibonacciNumber(n-2) if n > 1 else n
    
    def FibonacciNumber2(self, n):
        l, r = -1, 1
        for _ in range(n+1):
            l, r = r, l + r
        return r

    def FibonacciNumber3(self, n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a+b
        return a

    @lru_cache(maxsize=512)
    def FibonacciNumber4(self, n):
        if n < 2:
            return n
        return self.FibonacciNumber4(n-1) + self.FibonacciNumber4(n-2)

    def FibonacciNumber5(self, n):
        if n == 0 or n==1:
            return n
        res = [0 , 1]
        for _ in range(1, n):
            res.append(sum(res[-2:])) # add sum of last 2 numbers to results
        return res[-1] # return last result from the list

#####################################################
# 693. Binary number with alternating bits
# 
    def HasAlternatingBits(self, n):
        return not any(a == b for a, b in zip(bin(n)[2:], bin(n)[3:]))
    
    def HasAlternatingBits2(self, n):
        return all(dup not in bin(n) for dup in ("11", "00"))

#####################################################
# 696. Count binary substrings
# Give a string s, count the number of non-empty (contiguous) substrings that have the same number of 0's and 1's, and all the 0's and all the 1's in these substrings are grouped consecutively.
#Substrings that occur multiple times are counted the number of times they occur.

#Example 1: Input: "00110011" Output: 6
#Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".

#First, I count the number of 1 or 0 grouped consecutively.
#For example "0110001111" will be [1, 2, 3, 4].
#Second, for any possible substrings with 1 and 0 grouped consecutively, the number of valid substring will be the minimum number of 0 and 1.
#For example "0001111", will be min(3, 4) = 3, ("01", "0011", "000111")

    def CountBinarySubstrings(self, s):
        s = map(len, s.replace('01', '0 1').replace('10', '1 0').split())
        s  = list(s)
        return sum(min(a, b) for a, b in zip(s, s[1:]))

    def CountBinarySubstrings2(self, s):
        pre, cur, res = 0, 1, 0
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                cur += 1
            else:
                pre = cur
                cur = 1
            if pre > cur:
                res += 1
        return res

    def CountBinarySubstrings3(self, s):
        pre, cur, res = 0, 1, 0
        for i in range(1, len(s)):
            if s[i] == s[i-1]:
                cur += 1
            else:
                res += min(pre, cur)
                pre = cur
                cur = 1
        return res + min(pre, cur)

#####################################################
# 728. Self dividing numbers
# A self-dividing number is a number that is divisible by every digit it contains.
# For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.
# Given a lower and upper number bound, output a list of every possible self dividing number, including the bounds if possible.
# Also, a self-dividing number is not allowed to contain the digit zero.

    def SelfDividingNumbers(self, left, right):
        is_self_dividing = lambda num: '0' not in str(num) and all([num % int(digit) == 0 for digit in str(num)])
        return filter(is_self_dividing, range(left, right+1))

#[num % int(digit) == 0 for digit in str(num)] creates an entire list which is not necessary. 
#By leaving out the [ and ], we can make use of generators which are lazy and allows for short-circuit evaluation, i.e. all will terminate as soon as one of the digits fail the check.

#The answer below improves the run time from 128 ms to 95 ms:

    def SelfDividingNumbers2(self, left, right):
        is_self_dividing = lambda num: '0' not in str(num) and all(num % int(digit) == 0 for digit in str(num))
        return filter(is_self_dividing, range(left, right + 1))


    def SelfDividingNumbers3(self, left, right):
        def is_valid(num):
            n = num
            while n:
                n, i = divmod(n, 10)
                if i == 0 or num % i != 0:
                    return False
            return True

        return filter(is_valid, range(left, right + 1))




#####################################################
# 751. IP to CIDR
# refer to the question description online
# x & -x => tail 1
    def IpToInt(self, ip):
        res = 0
        for x in ip.split('.'):
            res = 256*res + int(x)
        return res

    def IntToIp(self, x):
        return ".".join(str((x>>i)%256) for i in (24, 16, 8, 0))

    
    def IpToCIDR(self, ip, n):
        start = self.IpToInt(ip)
        res = []
        while n:
            mask = max(33-(start & -start).bit_length(), 33-n.bit_length())
            res.append(self.IntToIp(start) + '/' + str(mask))
            start += 1 << (32 - mask)
            n -= 1 << (32-mask)
        return res

#ip="255.0.0.7"
#n=10
#a=solution()
#a.ipToCIDR(ip,n)

#####################################################
# 762. Prime number of set bits in binary representation
# Given two integers L and R, find the count of numbers in the range [L, R] (inclusive) having a prime number of set bits in their binary representation.

    def CountPrimeSetBits(self, L, R):
        return sum(bin(i).count('1') in (2,3,5,7,11,13,17,19,23,29,31) for i in range(L, R+1))

    def CountPrimeSetBits2(self, L, R):
        def is_prime(n):
            if n > 1:
                for i in range(2, n): 
                    if n % i == 0:
                        return False
                return True
            else:
                return False
        count = 0
        for j in range(L, R + 1):
            n_ones = len([1 for bit in list(bin(j))[2:] if bit == '1'])
            if is_prime(n_ones):
                count += 1
        return count

    def CountPrimeSetBits3(self, L, R):
        def is_prime(n):
            return all(n%j for j in range(2, int(n**0.5)+1)) and n > 1
        t = map(is_prime, [bin(i).count('1') for i in range(L, R + 1)])
        return sum(t)

#L,R=10,15
#a=solution()
#a.countPrimeSetBits3(L,R)


#####################################################
# 788. Rotated digits
#X is a good number if after rotating each digit individually by 180 degrees, we get a valid number that is different from X.  Each digit must be rotated - we cannot choose to leave it alone.
#A number is valid if each digit remains a digit after rotation. 0, 1, and 8 rotate to themselves; 2 and 5 rotate to each other; 6 and 9 rotate to each other, and the rest of the numbers do not rotate to any other number and become invalid.
#Now given a positive number N, how many numbers X from 1 to N are good?

    def RotateDigits(self, N):
        return len([num for num in range(1, N+1) if not set(str(num)) & {"3", "4", "7"} and set(str(num)) & {"2", "5", "6", "9"}])             



    
#####################################################
# 800. Similar RGB color
# every capital letter represents some hexadecimal digit from 0 to f.
#Example 1:
#Input: color = "#09f166"
#Output: "#11ee66"
#Explanation:  
#The similarity is -(0x09 - 0x11)^2 -(0xf1 - 0xee)^2 - (0x66 - 0x66)^2 = -64 -9 -0 = -73.
#This is the highest among any shorthand color.

    def SimilarRGB(self, color):
        def getClosest(s):
            m = ['00','11','22','33','44','55','66','77','88','99','aa','bb','cc','dd','ee','ff']
            return min(m, key = lambda x: abs(int(x, 16) - int(s, 16)))

        res = ''
        color = color[1:] #skip the first char of '#'
        for i in range(3): 
            s = color[2*i:2*i+2]
            res += getClosest(s)
        return '#' + res

    def SimilarRGB2(self, color):
        res = "#"
        for i in range(1, len(color), 2):
            q, r = divmod(int(color[i:i+2], 16), 17)  #11, 22, 33, 44 => diff()=int(11, 16)=17
            if r > 8:   #  17//2 = 8
                q += 1   
            res += '{:02x}'.format(17*q)
        return res

#####################################################
# 860. Lemonde change
# At a lemonade stand, each lemonade costs $5. 
#Customers are standing in a queue to buy from you, and order one at a time (in the order specified by bills).
#Each customer will only buy one lemonade and pay with either a $5, $10, or $20 bill.  You must provide the correct change to each customer, so that the net transaction is that the customer pays $5.
#Note that you don't have any change in hand at first.
#Return true if and only if you can provide every customer with correct change.

    def LemonadeChange(self, bills):
        five = ten = 0
        for i in bills:
            if i == 5:
                five += 1
            elif i == 10:
                five, ten = five - 1, ten + 1
            elif ten > 0: # if it's a $20 bill
                five, ten = five - 1, ten - 1
            else: 
                five -= 3
            if five < 0:
                return False
        return True

    def LemonadeChange2(self, bills):
        five = ten = 0
        for i in bills:
            if i == 5:
                five += 1
            elif i == 10 and five:
                five, ten = five - 1, ten + 1
            elif i == 20 and five and ten:
                five, ten = five - 1, ten - 1
            elif i == 20 and five >= 3:
                five -= 3
            else:
                return False
        return True


            


#####################################################
# 868. Binary gap 
# Given a positive integer N, find and return the longest distance between two consecutive 1's in the binary representation of N.

    def BinaryGap(self, N):
        res = [ len(s)+1 for s in str(bin(N))[2:].split('1')[:-1]]
        return max(res)

    def BinaryGap2(self, N):
        pre = dist = 0
        for i, c in enumerate(bin(N)[2:]):
            if c == '1':
                dist = max(dist, i - pre)
                pre = i
            return dist
        

    def BinaryGap3(self, N):
        index = [i for i, v in enumerate(bin(N)) if v == '1']
        return max([ b - a for a, b in zip(index, index[1:])] or [0])

    

#####################################################
# 985. Sum of even numbers after queries
# We have an array A of integers, and an array queries of queries.
# For the i-th query val = queries[i][0], index = queries[i][1], we add val to A[index].  Then, the answer to the i-th query is the sum of the even values of A.
    def SumEvenAfterQueries(self, A, queries):
        res = []
        for query in queries:
            A[query[1]] += query[0]
            res.append(sum(x for x in A if not x%2))

        return res

#####################################################
# 1009. Complement of base 10 integer
# 

    def BitwiseComplement(self, N):
        x =1 
        while N > x:
            x = x * 2 + 1
        return x - N   #x = 11111...1111

    def BitwiseComplement2(self, N):
        x =1 
        while N > x:
            x = x * 2 + 1
        return N ^ x
    
    def BitwiseComplement3(self, N):
        return (1 << len(bin(N)) >> 2) - N - 1

    #def BitwiseComplement4(self, N):
        #return int(bin(N)[2:].translate(maketrans('01','10')), 2)

    def BitwiseComplement5(self, N):
        res = 0
        for c in bin(N)[2:]:
            res = (res << 1) + 1 - int(c)
            return res

    def BitwiseComplement6(self, N):
        return int(''.join(str(1-int(c)) for c in bin(N)[2:]), 2)

    def BitwiseComplement7(self, N):
        mask = 1
        while mask < N:
            mask = (mask << 1) + 1
        return N ^ mask


#####################################################
# 1056. Confusing number
# Given a number N, return true if and only if it is a confusing number, which satisfies the following condition:
#We can rotate digits by 180 degrees to form new digits. When 0, 1, 6, 8, 9 are rotated 180 degrees, they become 0, 1, 9, 8, 6 respectively. When 2, 3, 4, 5 and 7 are rotated 180 degrees, they become invalid. 
#A confusing number is a number that when rotated 180 degrees becomes a different number with each digit valid.
# refer to the question 788

    def ConfusingNumber(self, N):
        dic = {'0':'0','1':'1','6':'9','8':'8','9':'6'}
        t = ''.join(dic[k] for k in str(N) if k in dic)
        return str(N) != t[::-1]

    def ConfusingNumber2(self, N):
        mapping = {0:0, 1:1, 6:9, 8:8, 9:6}
        invalid = [2,3,4,5,7]
        tmp, res = 0, []
        while(N):
            N, tmp = divmod(N, 10)
            if tmp in invalid:
                return False
            res.append(mapping[tmp])
        res = res[::-1]
        r = 0
        for i, x in enumerate(res):
            r += 10 ** i * x
        return r != N

    




#####################################################
# 1085. Sum of digits in the minimum number
# Given an array A of positive integers, let S be the sum of the digits of the minimal element of A.
# Return 0 if S is odd, otherwise return 1.
# use divmod()

    def SumOfDigitsMinimumNumber(self, A):
        if not A:
            return 0
        res, mini = 0, min(A)
        while mini > 0:
            quo = min%10   # mini, quo = divmod(mini, 10) 
            mini = mini//10
            res += quo
        return 0 if res%2 else 1

    def SumOfDigitsMinimumNumber2(self, A):
        if not A:
            return 0
        res, N = 0, min(A)
        while N:
            N, carry = divmod(N, 10)
            res += carry
        return 0 if res%2 else 1





#####################################################
# 1099. Two sum less than K
# Given an array A of integers and integer K, return the maximum S such that there exists i < j with A[i] + A[j] = S and S < K. If no i, j exist satisfying this equation, return -1.
# use binary search

    def TwoSumLessThanK(self, A, K):
        res = -1
        for i in range(len(A)-1):
            for j in range(len(A)-1):
                if A[i] + A[j] < K and i != j:
                    res = max(res, A[i]+A[j])
        return res

    def TwoSumLessThanK2(self, A, K):
        A.sort()
        l, r, res = 0, len(A)-1, 0
        while l < r:
            if A[l] + A[r] < K:
                res = max(res, A[l] + A[r])
                l += 1
            else:
                r -= 1
        return res if res > 0 else -1



#####################################################
# 1118. Number of days in a month
# Given a year Y and a month M, return how many days there are in that month.

    def DaysInMonth(self, Y, M):
        if M == 2:
            return 29 if Y%4 == 0 else 28
        elif (M < 8 and M%2) or (M >= 8 and not M%2): 
            return 31
        else:
            return 30
        


#####################################################
# 1133. Largest unique number

    def LargestUniqueNumber(self, nums):
        d = [ k for k, v in collections.Counter(nums).items() if v == 1]
        return max(d) if len(d) > 0 else -1

    def LargestUniqueNumber2(self, nums):
        res, map = [], {}
        for n in nums:
            if map[n] is None:
                map[n] = 1
                res.append(n)
            else:
                if n in res:
                    res.remove(n)
        return max(res) if len(res) > 0 else -1
        

                

#####################################################
# 1134. Armstrong number
# The k-digit number N is an Armstrong number if and only if the k-th power of each digit sums to N
# Input: 153 Output: true 153 = 1^3 + 5^3 + 3^3
# use divmod

    def ArmstrongNumber(self, N):
        if sum(map(lambda n: n**len(str(N)), (int(m) for m in str(N)) )) == N:
            return 1
        return 0
    
    def ArmstrongNumber2(self, N):
        res, n = 0, len(str(N))
        while N:
            N, carry = divmod(N, 10)
            res += carry**n
        return res == N


#####################################################
# 1137. N-th Tribonacci number
#The Tribonacci sequence Tn is defined as follows: 
#T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.
#Given n, return the value of Tn.

    def TribonacciNumber(self, n):
        a, b, c = 1, 1, 0
        for _ in range(n):
            a, b, c = b, c, a+b+c
    
    def TribonacciNumber2(self, n):
        if n <= 1:
            return n
        if n == 2:
            return 1
        a, b, c = 0, 1, 1
        for _ in range(3, n+1):
            d = a + b + c
            a, b, c = b, c, d
        return d

    def TribonacciNumber3(self, n):
        m = [0, 1, 1]
        for _ in range(3, n+1):
            m.append(m[-1] + m[-2] + m[-3])
        return m[n]


