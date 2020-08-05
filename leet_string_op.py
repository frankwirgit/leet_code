import itertools
import collections
import re
import sys  #243
import functools #1047
import math #1071
import operator
import string
import os

#####################################################
# 3. Longest substring without repeating characters
# use the hashmap to detect dups
class solution:
    def MaxLenNRepStr(self, s):
        map1, maxmap = {}, {}
        i = 0
        for c in s:
            if (c in map1): #found dup
                if len(map1) > len(maxmap):
                    maxmap = map1.copy()
                map1.clear()
            map1[c] = i  #add new char in map
            i += 1
        return (''.join(maxmap.keys()))

    def lengthOfLongestSubstring2(self, s):
        dic, res, start, = {}, 0, 0
        for i, ch in enumerate(s):
            if ch in dic:
                # update the res
                res = max(res, i-start)
                # here should be careful, like "abba"
                start = max(start, dic[ch]+1)
            dic[ch] = i
        # return should consider the last 
        # non-repeated substring
        return max(res, len(s)-start)

#strs = "abcabcbb"
#a = solution()
#a.MaxLenNRepStr(strs)


#####################################################
# 5. Longest palindromic substring
# palindrome - string with symatric chars
# use brute force

    def centralExp(self, s, l, r):
        while (l >= 0) and (r < len(s)) and s[l] == s[r]:
            l -= 1
            r += 1
        return (r - l)

    def PalindromStr(self, s):
        start, end = 0, 0
        for i in range(len(s)):
            len1 = self.centralExp(s, i, i)  #with central char s[i]
            len2 = self.centralExp(s, i, i+1)  #without central char
            len_max = max(len1, len2)
            if(len_max > (end - start)):
                end = i + int((len_max)/2)
                start = i - int((len_max-1)/2)
            print(s[start:end])
        return s[start:end]


    def longestPalindrome(self, s):
        res = ""
        for i in range(len(s)):
            # odd case, like "aba"
            tmp = self.helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # even case, like "abba"
            tmp = self.helper(s, i, i+1)
            if len(tmp) > len(res):
                res = tmp
        return res
 
    # get the longest palindrome, l, r are the middle indexes   
    # from inner to outer
    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]

#strs="bababd"
#a = solution()
#a.PalindromStr(strs)


#####################################################
# 6. Zig zag conversion
# string "PAYPALISHIRING", number_of_row = 3 
# P   A   H   N
# A P L S I I G
# Y   I   R
#
# directly parse out
    def ZigZag(self, s, nrow): 
        if(nrow <= 1):
            print("invalid zigzag")
            return
        direction = 1
        dt = [""]*nrow
        row = 0
        for c in s:
            dt[row] = dt[row]+c #concatenate the row string
            if row == 0:
                direction = 1
            elif row == (nrow - 1):
                direction = -1
            row = row + direction
        return ("".join(dt))

#strs="PAYPALISHIRING"
#a = solution()
#a.ZigZag(strs,4)

#####################################################
# 10. Regular expression matching for Dot (one) and Asterisk (zero or many)
# s, p are two strings, p has '.' and '*' chars
# dynamic programming recursive()

    def RegularMatch(self, s, p):
        if not p:
            return (not s)
        firstm = bool(s) and p[0] in {s[0, "."]}
        if len(p) > 2 and p[1] in {"*"}:
            #self.RegularMatch(s, p[2:]) - zero char occurs before "*"
            #self.RegularMatch(s[1:], p) - any char in s matches "*"
            return firstm or self.RegularMatch(s[1:], p) or self.RegularMatch(s, p[2:])
        else:
            return firstm or self.RegularMatch(s[1:], p[1:])

#s  = "mississippi"
#p = "missis*ippi"
#a.RegularMatch(s,p)

#####################################################
# 14. Longest common prefix string
# use brute force

    def LongestPrefix(self, s):
        s = sorted(s, key=len)
        smin = s[0]  #string with the shortest length
        pref = ""
        for i in range(len(smin)):
            for j in range(1, len(s)):
                if smin[i] != s[j][i]:
                    i = len(smin)  # end the two for loops, no pref found
                    break
                elif j == len(s) - 1:  #the ith char matches with all strings
                    pref += smin[i]
        return pref

#s = ["flower","flow","flight"]
#a.LongestPrefix(s)

#####################################################
# 20. Valid parentheses
# use stack, and mapping

    def isValid(self, s):
        stack = []
        mp = {"(":")","[":"]","{":"}"}
        ls = [")","]","}"]
        for c in s:
            if c in mp:
                stack.append(c)  #push left-p into the stack
            elif c in ls:
                if stack:
                    tmp = stack.pop()  #if there is right-p, pop the stack to pair with it
                    if c != mp[tmp]: #if not the pair, then it is invalid
                        return False
                else:
                    return False
            else:
                continue
        return not stack

#a = solution()
#s1="([)]"
#print(a.isValid(s1))
#s2="{[]}"
#print(a.isValid(s2))

#####################################################
# 22. Generate parentheses
# use dynamic programming

    def GeneParen(self, n):
        def geneP(p, l, r, res=[]):
            if l:
                geneP(p + "(", l-1, r)
            if l < r:
                geneP(p + ")", l, r-1)
            if not r:
                res += p,
            return res
        return geneP("", n, n)

#n=3
#a.GeneParen(n)

#####################################################
# 28. Implement strStr()
# search the substring in a string and return the first match's index
# brute force O(mn)

    def strStr1(self, haystack, needle):
        l1, l2 = len(haystack), len(needle)
        if l1 < l2:
            return -1
        for i in range(l1 - l2 + 1):
            if needle == haystack[i:i+l2]: #compare substring
                return i
        return -1

    def strStr2(self, haystack, needle):
        l1, l2 = len(haystack), len(needle)
        if l1 < l2:
            return -1
        for i in range(l1 - l2 + 1):
            j = 0
            while j < l2 and needle[j] == haystack[i+j]: #compare every char
                j += 1
            if j == l2:
                return i
        return -1

#s= "hello"
#st = "ll"
#s = "aaaaa"
#st = "bba"
#a.strStr(s,st)

#####################################################
# 30. Substring with concatenation of all words
# given a string, s, and a list of words, words, 
# that are all of the same length. 
# Find all starting indices of substring(s) in s that is a concatenation of each word in words exactly once and without any intervening characters.
#

#use internal function - itertools.permutations
    def SubstrAllWords(self, s, words):
        aw = list(itertools.permutations(words))
        #print(a)
        res = []
        for w in aw:
            fdx = s.find(''.join(w))
            if fdx != -1:
                res.append(fdx)
        return res

#use brute force
#use left and right to define a moving window with window size = k
#  
    def SubstrAllWords2(self, s, words):
        if not words:
            return []
        k = len(words[0])
        res = []
        for left in range(k):
            d = collections.Counter(words)
            for right in range(left+k, len(s)+1, k): #each word in words having the same length k
                word = s[right - k: right]
                d[word] -= 1
                while d[word] < 0:
                    d[s[left:left+k]] += 1
                    left += k
                if left + k*len(words) == right: #found one index with whole match
                    res.append(left)
            return res

#s = "barfoothefoobarman"
#words = ["foo","bar"]
#s = "wordgoodgoodgoodbestword"
#words = ["word","good","best","word"]
#a.SubstrAllWords(s, words)

#####################################################
# 59. Length of last word
# Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.

    def LengthLastWord(self, s):
        if len(s.split(())) == 0:
            return 0
        else:
            return len(s.split()[-1])


#####################################################
# 125. Valid palindrome
# empty string is defined as valid palindrome.

    def ValidPalindrome(self, s):
        ss = "".join(re.findall("[a-zA-Z0-9]*", s)).lower()
        return ss == ss[::-1]

#strs="A man, a plan, a canal: Panama"
#a.ValidPalindrome(strs)

#####################################################
# 150. Read N characters given Read4
# the API: int read4(char *buf) reads 4 characters at a time from a file.
# by using the read4 API, implement the function int read(char *buf, int n) that reads n characters from the file.
    def read4(self, buf):
        #pass
        data = ""
        dev = os.open("/dev/rtlightsensor0", os.O_RDWR)
        data = os.read(dev,4)
        data.decode("utf-8")
        return data

    def read(self, buf, n):
        size, buf4 = 0, ['']*4
        while size < n:
            size4 = self.read4(buf4)
            if not size4:
                break
            size4 = min(size4, n-size) #the last batch assigned from read4()
            buf[size:] = buf4[:size4]
            size += size4
        return size

#buf, n = "abcde", 5
#a=Solution()
#a.read(buf,n)

#####################################################
# 193. Valid phone number
# a valid phone number must appear in one of the following two formats: (xxx) xxx-xxxx or xxx-xxx-xxxx. (x means a digit)
# shell script - regular expression

#egrep -o "^( ([0-9]{3}\-) | (\([0-9]{3}\)) ){1} [0-9]{3}\-[0-9]{4}$" file.txt

#####################################################
# 195. Tenth Line
# Given a text file file.txt, print just the 10th line of the file

#awk 'NR==10' file.txt
#sed -n '10p'< file.txt
#tail -n +10 file.txt | head -n 1


#####################################################
# 205. Isomorphic string
# Isomorpic: if the characters in s can be replaced to get t.

    def IsomorphicStrings(self, s, t):
        return map(s.find, s) == map(t.find, t)

    def IsomorphicStrings2(self, s, t):
        return [s.find(i) for i in s] == [t.find(j) for j in t]
    
    def IsomorphicStrings3(self, s, t):
        d1, d2 = {}, {}
        for i, val in enumerate(s):
            d1[val] = d1.get(val, []) + [i]
        for i, val in enumerate(t):
            d2[val] = d2.get(val, []) + [i]

        return sorted(d1.values()) == sorted(d2.values())

    def IsomorphicStrings4(self, s, t):
        d1, d2 = [ [] for _ in range(26)], [ [] for _ in range(26)]
        for i, val in enumerate(s.lower()):
            d1[ord(val)-ord('a')].append(i)
        for i, val in enumerate(t.slower()):
            d2[ord(val)-ord('a')].append(i)
        return sorted(d1) == sorted(d2)

#s, t='paper', 'title'
#a.IsomorphicStrings(s,t)

#####################################################
# 242. Valid Anagram
# Given two strings s and t , write a function to determine if t is an anagram of s.

    def ValidAnagram(self, s, t):
        return sorted(s) == sorted(t)

    def ValidAnagram2(self, s, t):
        if len(s) != len(t):
            return False
        return collections.Counter(s) == collections.Counter(t)

#s, t="anagram", "nagaram"
#a.ValidAnagram(s,t)

#####################################################
# 243. Shortest word distance
# You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.

    def ShortestWordDistance(self, words, w1, w2):
        idx1, idx2, minD = -1, -1, sys.maxsize
        for i in range(len(words)):
            if words[i] == w1:
                idx1 = i
            elif words[i] == w2:
                idx2 = i
            if idx1 >= 0 and idx2 >= 0:
                minD = min(minD, abs(idx1 - idx2))
        return (minD if minD != sys.maxsize else None)


#words, w1,w2=["practice", "makes", "perfect", "coding", "makes"],"coding","practice"
#a.ShortestWordDistance(words,w1,w2)

#####################################################
# 266. Palindrome permutation
# Given a string, determine if a permutation of the string could form a palindrome
# For example, "code" -> False, "aab" -> True, "carerac" -> True.

    def PalindromePermutation(self, s):
        #return list(collections.Counter(s).values()).count(2) == len(s)%2
        #return (not len(s)%2 and list(collections.Counter(s).values()).count(2) == len(s)/2) or (len(s)%2 and list(collections.Counter(s).values()).count(2) == (len(s)-1)/2)
        #  in case of a palindrome, the number of characters with odd number of occurences can't exceed 1
        ls = [ v for v in list(collections.Counter(s).values()) if v % 2 ]
        return len(ls) < 2
        


    def PalindromePermutation2(self, s):
        left = set()
        for c in s:
            if c in left:
                left.remove(c)
            else:
                left.add(c)
        return len(left) < 2


#####################################################
# 290. Word pattern
#  Given a pattern and a string str, find if str follows the same pattern.

    def WordPattern(self, p, words):
        ws = words.split()
        #return [p.find(i) for i in p] == [ws.index(i) for i in ws]
        return list(map(p.find, p)) == list(map(ws.index, ws))

#p,words = "abba", "dog cat cat dog"
#a.WordPattern(p,words)



#####################################################
# 383. Randome Note
# Given an arbitrary ransom note string and another string containing letters from all the magazines, write a function that will return true if the ransom note can be constructed from the magazines ; otherwise, it will return false.
# Each letter in the magazine string can only be used once in your ransom note.
    def RansomConstruct(self, ransomNote, magazine):
        return not collections.Counter(ransomNote) - collections.Counter(magazine)




#####################################################
# 387. First unique character in a string
# Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
    def FirstUniqueCharacter(self, s):
        dt, res = {}, -1
        for i, c in enumerate(s[::-1]):
            if c not in dt:
                res = len(s) - i - 1
            dt[c] = i
        return res

    def FirstUniqueCharacter2(self, s):
        return min([s.find(c) for c in string.ascii_lowercase if s.count(c)==1]) or -1
    
    def FirstUniqueCharacter3(self, s):
        #return min([s.find(c) for c, v in collections.Counter(s).iteritems() if v == 1]) or -1
        return min([s.find(c) for c, v in collections.Counter(s).items() if v == 1]) or -1



#####################################################
# 389. Find the difference
# Given two strings s and t which consist of only lowercase letters.
# String t is generated by random shuffling string s and then add one more letter at a random position.
# Find the letter that was added in t.
    def FindTheDifference(self, s, t):
        return chr(sum(map(ord, t)) -sum(map(ord, s)))
    
    def FindTheDifference2(self, s, t):
        return chr(functools.reduce(int.__xor__, map(ord, s+t)))

    def FindTheDifference3(self, s, t):
        return chr(functools.reduce(operator.xor, map(ord, s+t)))

    def FindTheDifference4(self, s, t):
        dic_s = collections.Counter(s)
        dic_t = collections.Counter(t)
        for key, val in dic_t.items():
            if key not in dic_s or val > dic_s[key]:
                return key

    def FindTheDifference5(self, s, t):
        return list(iter(collections.Counter(t)-collections.Counter(s)))[0]


#####################################################
# 408. Valid word abbreviation
# Given a non-empty string s and an abbreviation abbr, return whether the string matches with the given abbreviation.

    def solution ValidWordAbbreviation(self, word, abbr):
        n = len(word)
        count, loc = 0,  0
        for w in abbr:
            if w.isdigit():
                if w == '0' and count == 0:
                    return False
                count = count * 10 + int(w)
            else:
                loc += count
                count = 0
                if loc >= size or word[loc] != w:
                    return False
                loc += 1
        return loc + count == n



#####################################################
# 409. Longest palindrome
# Given a string which consists of lowercase or uppercase letters, find the length of the longest palindromes that can be built with those letters.
# This is case sensitive, for example "Aa" is not considered a palindrome here.

    def LongestPalindrome(self, strs):
        s = collections.Counter(strs)
        return sum(i - (i%2) for i in s.values()) + 1 * (any(i%2==1 for i in s.values()))
        # sum of the even counters plus 1 (if there is at least one odd counter)




#####################################################
# 500. Keyboard Row
# Given a List of words, return the words that can be typed using letters of alphabet on only one row's of American keyboard like the image below.

    def KeyboardRow(self, words):
        return filter(re.compile('(?i)([qwertyuiop]*|[asdfghjkl]*|[zxcvbnm]*)$').match, words)

    def KeyboardRow2(self, words):
        rows = ['qwertyuiopQWERTYUIOP', 'asdfghjklASDFGHJKL', 'zxcvbnmZXCVBNM']
        return [word for word in words if any([all([ch in row for ch in word]) for row in rows])]

    def KeyboardRow3(self, words):
        row1 = set("qwertyuiopQWERTYUIOP")
        row2 = set("asdfghjklASDFGHJKL")
        row3 = set("zxcvbnmZXCVBNM")
        return [word for word in words if set(word.lower())<=row1 or set(word.lower())<=row2 or set(word.lower())<=row3]

    def KeyboardRow4(self, words):
        return [word for row in [set('qwertyuiop'), set('asdfghjkl'), set('zxcvbnm')] \
                for word in words if set(word.lower()) <= row]


#####################################################
# 520. Detect capital
#All letters in this word are capitals, like "USA".
#All letters in this word are not capitals, like "leetcode".
#Only the first letter in this word is capital, like "Google".
#Otherwise, we define that this word doesn't use capitals in a right way.

    def DetectCaptialUse(self, word):
        return word.isupper() or word.islower() or word.istitle()

    def DetectCaptialUse2(self, word):
        return word[1:] == word[1:].lower() or word == word.upper()

    def DetectCaptialUse3(self, word):
        return all(ord('A') <= ord(w) <= ord('Z') for w in word) or all(ord('a') <= ord(w) <= ord('Z') for w in w[1:])
    
#use regular expression
    def DetectCaptialUse4(self, word):
        pw = re.compile(r"^[A-Z]*$|^[a-z]*$|^[A-Z][a-z]+$")
        return word.match(pw)
    



#####################################################
# 557. Reverse words in a string III
# Given a string, you need to reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.

    def ReverseWords(self, s):
        return " ".join(s.split()[::-1])[::-1]  #first reverse the order of the words and then reverse the entire string.

    def ReverseWords2(self, s):
        return " ".join(map(lambda x: x[::-1], s.split()))

    def ReverseWords3(self, s):
        return " ".join(w[::-1] for w in s.split())

#####################################################
# 682. Baseball game
# Integer (one round's score): Directly represents the number of points you get in this round.
#"+" (one round's score): Represents that the points you get in this round are the sum of the last two valid round's points.
#"D" (one round's score): Represents that the points you get in this round are the doubled data of the last valid round's points.
#"C" (an operation, which isn't a round's score): Represents the last valid round's points you get were invalid and should be removed.
#Each round's operation is permanent and could have an impact on the round before and the round after.
#You need to return the sum of the points you could get in all the rounds.

    def CalulatePoints(self, ops):
        res = []
        for op in ops:
            if op == 'C':
                res.pop()
            elif op == 'D':
                res.append(res[-1]**2)
            elif op == '+':
                res.append(sum(res[-2:]))
            else:
                res.append(int(op))
        return sum(res)






#####################################################
# 709. To lower case
# Implement function ToLowerCase() that has a string parameter str, and returns the same string in lowercase.
    def ToLowerCase(self, s):
        return s.lower()

    def ToLowerCase2(self, s):
        return ''.join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)

#####################################################
# 717. 1-bit and 2-bit characters
# We have two special characters. The first character can be represented by one bit 0. The second character can be represented by two bits (10 or 11).
# Now given a string represented by several bits. Return whether the last character must be a one-bit character or not. The given string will always end with a zero.

#Just go through the bits. For every 1-bit, skip the following bit. Return whether the last seen start bit was zero.
    def IsOneBitCharacter(self, bits):
        bits = iter(bits)
        for bit in bits:
            if bit:
                next(bits)
        return not bit

    def IsOneBitCharacter2(self, bits):
        for i, bit in enumerate(bits):
            if bit:
                bits.pop(i + 1)
        return not bit

    def IsOneBitCharacter3(self, bits):
        bit = ''.join(str(i) for i in bits)
        res = bit.replace('10','').replace('11','')
        return res == "" or res[:-1] == '0'

    def isOneBitCharacter(self, bits):
            pos = 0
            while pos < len(bits)-1:
                if bits[pos]:
                    pos+=2
                else:
                    pos+=1
            return pos==len(bits)-1
    





#####################################################
# 748. Shortest completing word
# refer to the online question description

    def ShortestCompletingWord(self, lp, words):
        cnt = collections.Counter("".join( c for c in lp.lower() if c.isalpha()))
        return min([w for w in words if not cnt - collections.Counter(w)], key=len)


    def ShortestCompletingWord2(self, lp, words):
        def a_in_b(c1, c2):
            for k in c1.keys():
                if c2[k] < c1[k]:
                    return False
            return True
        cnt = collections.Counter("".join( c for c in lp.lower() if c.isalpha()))
        min_len = 1001
        res = ""
        for _, word in enumerate(words):
            w = collections.Counter(word.lower())
            if a_in_b(cnt, w):
                if len(word) < min_len:
                    min_len = len(word)
                    res = word
        return res
        





#####################################################
# 784. Letter case permutation
# Given a string S, we can transform every letter individually to be lowercase or uppercase to create another string.  Return a list of all possible strings we could create.

    def LetterCasePermutation(self, S):
        L = [ [i.lower(), i.upper()] if i.isalpha() else i for i in S]
        return [''.join(i) for i in itertools.product(*L)]

#Track every character in S and double result items with swapped case for the character in new item.

    def LetterCasePermutation2(self, S):
        res, curr = [str(S)], 0
        while curr < len(S):
            if S[curr].isalpha():
                mx, i = len(res), 0
                while i < mx:
                    res.append(str(res[i][:curr] + res[i][curr].swapcase()+res[i][curr+1:]))
                    i += 1
            curr += 1
        return res

    def LetterCasePermutation3(self, S, res = {""}):
        for s in S:
            res = { r + t for r in res for t in (s.lower(), s.upper())}
        return list(res)

#####################################################
# 796. Rotate string
# We are given two strings, A and B.
# A shift on A consists of taking string A and moving the leftmost character to the rightmost position. For example, if A = 'abcde', then it will be 'bcdea' after one shift on A. Return True if and only if A can become B after some number of shifts on A.

    def RotateString(self, A, B):
        return len(A) == len(B) and B in A + A
    



#####################################################
# 804. Unique Morse code words
# Example: Input: words = ["gin", "zen", "gig", "msg"] Output: 2
# Explanation: The transformation of each word is:
#"gin" -> "--...-."
#"zen" -> "--...-."
#"gig" -> "--...--."
#"msg" -> "--...--."
# There are 2 different transformations, "--...-." and "--...--.".

    def UniqueMorseCodeWords(self, words):
        Morse = [".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--",
             "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--.."]
        return len({"".join(Morse[ord(i)-ord('a')] for i in w) for w in words })  #use set to dedup

#####################################################
# 806. Number of lines to write string
#We are to write the letters of a given string S, from left to right into lines. Each line has maximum width 100 units, and if writing a letter would cause the width of the line to exceed 100 units, it is written on the next line. We are given an array widths, an array where widths[0] is the width of 'a', widths[1] is the width of 'b', ..., and widths[25] is the width of 'z'.
#Now answer two questions: how many lines have at least one character from S, and what is the width used by the last such line? Return your answer as an integer list of length 2.

#widths = [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
#S = "bbbcccdddaaa"
#Output: [2, 4]
#Explanation: 
#All letters except 'a' have the same length of 10, and 
#"bbbcccdddaa" will cover 9 * 10 + 2 * 4 = 98 units.
#For the last 'a', it is written on the second line because
#there is only 2 units left in the first line.
#So the answer is 2 lines, plus 4 units in the second line.

    def NumberOfCurs(self, widths, s):
        res, cur = 1, 0
        for i in s:
            width = widths[ord(i) - ord('a')]
            res += 1 if cur + width > 100 else 0
            cur = width if cur + width > 100 else cur + width
        return [res, cur]

    


#####################################################
# 811. Subdomain visit count
#
    def SubdomainVisits(self, mydomains):
        c = collections.Counter()
        for nd in mydomains:
            n, d = nd.split()
            c[d] += int(n)
            for i in range(len(d)):
                if d[i] == '.':
                    c[d[(i+1):]] += int(n)
        return ["%d %s" % (c[k], k) for k in c]

    def SubdomainVisits2(self, mydomains):
        c = collections.Counter()
        for nd in mydomains:
            n, *d = nd.replace(" ", ".").split(".")  #*d is a list of d
            for i in range(len(d)):
                c[".".join(d[i:])] += int(n)
        return ["%d %s" % (c[k], k) for k in c]

    def SubdomainVisits3(self, mydomains):
        res, dic = [], {}
        for nd in mydomains:
            n, d = nd.split()
            dt = d.split(".")
            loc = dt[-1]
            if loc in dic:
                dic[loc] += int(n)
            else:
                dic[loc] = int(n)
            for i in range(len(dt)-2, -1, -1):
                loc = dt[i] + "." + loc
                if loc in dic:
                    dic[loc] += int(n)
                else:
                    dic[loc] = int(n)
        for key, val in dic.items():
            res.append(str(val)+" "+key)
        return res

#####################################################
# 824. Goat latin
#The rules of Goat Latin are as follows:
#If a word begins with a vowel (a, e, i, o, or u), append "ma" to the end of the word.
#For example, the word 'apple' becomes 'applema'.
 
#If a word begins with a consonant (i.e. not a vowel), remove the first letter and append it to the end, then add "ma".
#For example, the word "goat" becomes "oatgma".
 
#Add one letter 'a' to the end of each word per its word index in the sentence, starting with 1.
#For example, the first word gets "a" added to the end, the second word gets "aa" added to the end and so on.
#Return the final sentence representing the conversion from S to Goat Latin. 

    def ToGoatLatin(self, S):
        vowel = set('aeiouAEIOU')
        def latin(w, i):
            if w[0] not in vowel:
                w = w[1:] + w[0]
            return w + 'ma' + 'a'*(i + 1)
        return ' '.join(latin(w, i) for i, w in enumerate(S.split()))


#####################################################
# 830. Position of large groups
#In a string S of lowercase letters, these letters form consecutive groups of the same character.
#For example, a string like S = "abbxxxxzyy" has the groups "a", "bb", "xxxx", "z" and "yy".
#Call a group large if it has 3 or more characters.  We would like the starting and ending positions of every large group.
#The final answer should be in lexicographic order.

    def LargeGroupPositions(self, s):
        i, j, N = 0, 0, len(s)
        res = []
        while i < N:
            while j < N and s[j] == s[i]:
                j += 1
            if j - i >= 3: 
                res.append((i, j -1))
            i = j
        return res

    def LargeGroupPositions2(self, s):
        curIndex, res = 0, []
        for i in itertools.groupby(s):
            length = len(list(i[1]))
            if length >= 3:
                res.append([curIndex, curIndex + length -1])
            curIndex += length
        return res


#####################################################
# 859. Buddy strings
# Given two strings A and B of lowercase letters, return true if and only if we can swap two letters in A so that the result equals B.

    def BuddyStrings(self, A, B):
        if len(A) != len(B) or set(A) != set(B):
            return False
        if A == B:
            return len(A) - len(set(A)) >= 1  #A=aabb, B=aabb?
        else:
            cnt, res = 0, []
            for i in range(len(A)):
                if A[i] != B[i]:
                    cnt += 1
                    res.append(i)
                if cnt > 2:
                    return False
            return A[res[0]] == B[res[1]] and A[res[1]] == B[res[0]]
        
    def BuddyStrings2(self, A, B):
        if len(A) != len(B):
            return False
        dif, dup = [[s, B[i]] for i, s in enumerate(A) if s != B[i]], len(A) != len(set(A))
        return len(dif) == 2 and dif[0] == dif[1][::-1] or (not dif and dup)

#####################################################
# 884. Uncommon words from two sentences
# We are given two sentences A and B.  (A sentence is a string of space separated words.  Each word consists only of lowercase letters.)
# A word is uncommon if it appears exactly once in one of the sentences, and does not appear in the other sentence.
# Return a list of all uncommon words. 

    def UncommonFromSentences(self, A, B):
        c = collections.Counter( (A + " " + B).split())
        return [w for w in c if c[w] == 1]




#####################################################
# 893. Groups of special-equivalent strings
# You are given an array A of strings.
#Two strings S and T are special-equivalent if after any number of moves, S == T.
#A move consists of choosing two indices i and j with i % 2 == j % 2, and swapping S[i] with S[j].
#Now, a group of special-equivalent strings from A is a non-empty subset S of A such that any string not in S is not special-equivalent with any string in S.
#Return the number of groups of special-equivalent strings from A.

#Example 3:
#Input: ["abc","acb","bac","bca","cab","cba"]
#Output: 3
##Explanation: 3 groups ["abc","cba"], ["acb","bca"], ["bac","cab"]

#Example 4:
#Input: ["abcd","cdab","adcb","cbad"]
#Output: 1
#Explanation: 1 group ["abcd","cdab","adcb","cbad"]

    def NumSpecialEquivGroups(self, A):
        return set( "".join(sorted(s[0::2]))  + "".join(sorted(s[1::2]))  for s in A)

#####################################################
# 917. Reverse only letters
# Given a string S, return the "reversed" string where all characters that are not a letter stay in the same place, and all letters reverse their positions.
# use in-place storage
    def ReverseOnlyLetters(self, S):
        S, i, j = list(S), 0, len(S)-1
        while i < j:
            if not S[i].isalpha():
                i += 1
            elif not S[j].isalpha():
                j -= 1
            else:
                S[i], S[j] = S[j], S[i]
                i, j = i + 1, j - 1
        return "".join(S)

    def ReverseOnlyLetters2(self, S):
        return re.sub(r'[A-Za-z]', "{}", S).format(*[c for c in S[::-1] if c.isalpha()])
#t = re.sub(r'[A-Za-z]', "{}", S)
#print(t)
#{}-{}{}-{}{}{}-{}{}{}{}

    def ReverseOnlyLetters3(self, S):
        r = [s for s in S if s.isalpha()]
        return "".join(S[i] if not S[i].isalpha() else r.pop() for i in range(len(S)))

    def ReverseOnlyLetters4(self, S):
        r = [s for s in S if s.isalpha()]
        return "".join(r.pop() if c.isalpha() else c for c in S)





#####################################################
# 929. Unique email address
# rules:
# If you add periods ('.') between some characters in the local name part of an email address, mail sent there will be forwarded to the same address without dots in the local name.  For example, "alice.z@leetcode.com" and "alicez@leetcode.com" forward to the same email address.  (Note that this rule does not apply for domain names.)
# If you add a plus ('+') in the local name, everything after the first plus sign will be ignored. This allows certain emails to be filtered, for example m.y+name@email.com will be forwarded to my@email.com.  (Again, this rule does not apply for domain names.)

# Example 1:
#Input: ["test.email+alex@leetcode.com","test.e.mail+bob.cathy@leetcode.com","testemail+david@lee.tcode.com"]
# Output: 2
# Explanation: "testemail@leetcode.com" and "testemail@lee.tcode.com" actually receive mails
 
    def NumberUniqueEmails(self, emails):
        addresses = set()
        for email in emails:
            local, domain = email.split("@")
            local = local.split("+")[0].replace(".","")
            addresses.add(local + "@" + domain)
        return len(addresses)


#####################################################
# 953. Verifying an alien dictionary
# Given a sequence of words written in the alien language, and the order of the alphabet, return true if and only if the given words are sorted lexicographicaly in this alien language.
    def IsAlienSorted(self, words, order):
        m = {c: i for i, c in enumerate(order)}
        words = [[m[c] for c in w] for w in words]
        return all(w1 <= w2 for w1, w2 in zip(words, words[1:]))

    def IsAlienSorted2(self, words, order):
        return words == sorted(words, key=lambda w: map(order.index, w))






#####################################################
# 1002. Find common characters#
#Given an array A of strings made only from lowercase letters, return a list of all characters that show up in all strings within the list (including duplicates).  For example, if a character occurs 3 times in all strings but not 4 times, you need to include that character three times in the final answer.
#You may return the answer in any order.


    def CommonChars(self, strs):
        res = collections.Counter(strs[0])
        for st in strs:
            res &=  collections.Counter(st)

        return list(res.elements())
    
    def CommonChars2(self, strs):
        dictt = {i:strs[0].count(i) for i in strs[0]}
        for st in strs[1:]:
            for j in dictt.keys():
                if st.count(j) < dictt[j]:
                    dictt[j] = st.count(j)
        n = ''
        for j in dictt.keys():
            n += j*dictt[j]
        return [j for j in n]

    #wrong
    def CommonChars3(self, strs):
        dictchar = collections.Counter(sorted(strs, key=len)[0])  #get the shortes element
        #return [ char for char in dictchar.keys() if st.count(char) > 0 for st in strs[1:] ]
        return [ char for char in dictchar.keys() for st in strs[1:] if st.count(char) > 0 ]
        













#####################################################
# 1021. Remove outermost parentheses
# Return S after removing the outermost parentheses of every primitive string in the primitive decomposition of S.
# Example 1: Input: "(()())(())" Output: "()()()"
# Example 2: Input: "(()())(())(()(()))" Output: "()()()()(())"
# Example 3: Input: "()()" Output: ""

    def RemoveOutermostParentheses(self, s):
        res, opened= [], 0
        for c in s:
            if c == '(' and opened > 0: res.append(c)
            if c == ')' and opened > 1: res.append(c)
            opened += 1 if c == '(' else -1
        return ''.join(res)

#s = "((()))"
#a.RemoveOutermostParentheses(s)

#####################################################
# 1047. Remove all adjacent duplicates in string
# Example 1: Input: "abbaca" Output: "ca"

    def RemoveAdjacentDuplicates(self, s):
        res = []
        for c in s:
            if res and res[-1] == c:
                res.pop()
            else:
                res.append(c)
        return "".join(res)

    def RemoveAdjacentDuplicates2(self, S):
        return functools.reduce(lambda s, c: s[:-1] if s[-1] == c else s + c, S, "#")[1:]  #?

    def RemoveAdjacentDuplicates3(self, s):
        i = 0
        while i < len(s)-1:
            if s[i] == s[i+1]:
                s = s[:i] + s[i+2:]
                if i > 0:
                    i -= 2  # i step back two indices in every case except i=0
                else:
                    i -= 1 # for i=0 to remain on 0, only -1 is necessary
            i += 1
        return s

    def RemoveAdjacentDuplicates4(self, s):
        last = None
        while last != s:
            last, s = s, re.sub(r'(.)\1', '', s) # "." any character,  "(.)" capturing group, "(.)\1" contents of group 1 - repeated (.),  re.sub() => remove any adjacent dup
        return s
    
#####################################################
# 1065. Index paris of a string
# Given a text string and words (a list of strings), return all index pairs [i, j] so that the substring text[i]...text[j] is in the list of words.
#Input: text = "ababa", words = ["aba","ab"]
#Output: [[0,1],[0,2],[2,3],[2,4]]
#Explanation: 
#Notice that matches can overlap, see "aba" is found in [0,2] and [2,4].
    def IndexPairs(self, text, words):
        if not words:
            return []
        res = []
        for word in words:
            startings = [ index for index in range(len(text)-len(word)) if text.startswith(word, index)]
            for start in startings:
                res.append([start, start+len(word)-1])
        return res.sort()

#####################################################
# 1071. Greatest common divisor of strings
# For strings S and T, we say "T divides S" if and only if S = T + ... + T  (T concatenated with itself 1 or more times)
#Return the largest string X such that X divides str1 and X divides str2.

    def GcdOfStrings(self, str1, str2):
        return math.gcd(len(str1), len(str2))

    def GcdOfStrings2(self, str1, str2):
        l1, l2 = len(str1), len(str2)
        str = str1 if l1 < l2 else str2
        for l in range(min(l1, l2), 0, -1):
            if l1%l == 0 and l2%l == 0:
                if str[:l]*(l1//l) == str1 and str[:l]*(l2//l) == str2:
                    return str[:l]
        return ""

        
#str1 = "ABCABC"
#str2 = "ABC"
#print(a.gcdOfStrings(str1, str2))
    
    def GcdOfStrings3(self, str1, str2):
        res = ''.join([a for a, b in zip(str1, str2) if a == b])
        for i in range(1, len(res) + 1)[::-1]:
            if not ''.join(str1.split(res[:i])) and not ''.join(str2.split(res[:i])):
                return res[:i]

#print(str1.split('ABC'))
#['', '', '']



#####################################################
# 1078. Occurrences after bigram
#Given words first and second, consider occurrences in some text of the form "first second third", where second comes immediately after first, and third comes immediately after second.
# For each such occurrence, add "third" to the answer, and return the answer.


    def FindOccurrences(self, text, first, second):
        res = []
        if not text:
            return res
        text = text.split()
        for i in range(2, len(text)):
            if text[i-2] == first and text[i-1] == second:
                res.append(text[i])
        return res

    def FindOccurrences2(self, text, first, second):
        text = text.split()
        return [ c for a, b, c in zip(text, text[1:], text[2:]) if a == first and b == second]





#####################################################
# 1108. Defanging an IP address
# A defanged IP address replaces every period "." with "[.]"

    def DefangIPAddress(self, s):
        return s.replace('.', '[.]')


#####################################################
# 1119. Remove vowels from a string

    def RemoveVowels(self, s):
        vows = ['a','e','i','o','u']
        return ''.join([ c for c in s if c.lower not in vows ])




