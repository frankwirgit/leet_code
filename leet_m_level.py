def f807(grid):
    max_row = [max(row) for row in grid]
    max_col = [max(col) for col in zip(*grid)]
    res  =0
    for i, row in enumerate(grid):
        for j, h in enumerate(row):
            res += min(max_row[i], max_col[j]) - h
    return res

grid = [ [3, 0, 8, 4], 
  [2, 4, 5, 7],
  [9, 2, 6, 3],
  [0, 3, 1, 0] ]

print(f807(grid))

###############################
#reversed inorder traversal
def f1038(root):
    if not root:
        return None
    q, res, presum = [], root, 0
    while q:
        while root:
            q.append(root)
            root = root.right
        node = q.pop()
        presum += node.val
        node.val = presum
        root = node.left
    return res

###############################

import string
import random
class f535:
    alphac = string.ascii_letters + string.digits
    
    def __init__(self):
        self.url2code = {}
        self.code2url = {}
    
    def encode(self, longUrl):
        while longUrl not in self.url2code:
            code = ''.join(random.choice(f535.alphac) for _ in range(6))
            if code not in self.code2url:
                self.code2url[code] = longUrl
                self.url2code[longUrl] = code
            return 'http://tinyurl.com/' + self.url2code[longUrl]
        
    def decode(self, shortUrl):
        return self.code2url[shortUrl[-6:]]
    

###############################

class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def f108(N):
    if not N: 
        return None
    l, r = 0, len(N)-1
    if l <= r:
        mid = (l + r) // 2

        node = TreeNode(N[mid])
        node.left = f654(N[:mid])
        node.right = f654(N[mid+1:])
    
        return node


###############################

def f654(N):
    if not N: 
        return None
    #l, r = 0, len(N)-1
    mid = N.index(max(N))

    node = TreeNode(N[mid])
    node.left = f654(N[:mid])
    node.right = f654(N[mid+1:])

    return node

###############################

def f701(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        if not root.elft:
            root.left = TreeNode(val)
        else:
            f701(root.left, val)
    else:
        if not root.right:
            root.right = TreeNode(val)
        else:
            f701(root.right, val)
    return root

###############################

import itertools

def f1079(tiles):
    return sum(len(set(itertools.permutations(tiles, i))) for i in range(1, len(tiles)+1))

#OR

def f1079_2(tiles):
    res = {""}
    for c in tiles:
        res |= {d[:i] + c + d[i:] for d in res for i in range(len(d)+1)}
    return len(res) - 1




###############################

def f544(N):
    team = list(map(str, range(1, N+1)))
    while N > 1:
        for i in range(N//2):
            team[i] = "({}, {})".format(team[i], team.pop())
        N //= 2
    return team[0]


###############################

import bisect
def f1008(A):
    def helper(i, j):
        if i == j: return None
        root = TreeNode(A[i])
        mid = bisect.bisect(A, A[i], i+1, j)
        root.left = helper(i+1, mid)
        root.right = helper(mid, j)
        return root
    return helper(0, len(A))

def f1008_1(A):
    if not A:
        return None
    node = TreeNode(A.pop(0))
    i, n = 0, len(A)
    while i < n and A[i] < node.val:
        i += 1
    node.left = f1008_1(A[:i])
    node.right = f1008_1(A[i:])
    return node
    
    

def f1008_2(A):
    root = TreeNode(A[0])
    q = [root]
    for v in A[1:]:
        if v < q[-1].val:
            q[-1].left = TreeNode(v)
            q.append(q[-1].left)
        else:
            while q and v > q[-1].val:
                node = q.pop()
            node.right = TreeNode(v)
            q.append(node.right)
        return root


###############################

import collections
def f950(deck):
    d = collections.deque()
    for x in sorted(deck)[::-1]:
        d.rotate()
        d.appendleft(x)
    return list(d)

def f950_2(deck):
    ind = list(range(len(deck)))
    for n in sorted(deck):
        deck[ind[0]] = n
        ind = ind[2:] + [ind[1]] if len(ind) > 1 else []
    return deck


###############################

def f1100(strs, k):
    if len(strs)< k:
        return 0
    res = []
    for i in range(len(strs)-k+1):
        if len(set(strs[i:i+k])) == k:
            res.append(strs[i:i+k])
    return res

###############################

def f894(N):
    N = N - 1
    if N == 0:
        return [TreeNode(0)]
    res = []
    for i in range(1, min(20, N), 2):
        for left in f894(i):
            for right in f894(N - i):
                root = TreeNode(0)
                root.left = left
                root.right = right
                res += [root]
    return res


def f894_2(N):
    res = {}
    def dfs(A):
        if len(A) == 1: return [TreeNode(0)]
        if str(A) in res: return res[str(A)]
        current = []
        for i in range(len(A)):
            if len(A[:i])%2== 1 and len(A[i+1:])%2 == 1:
                lefty = dfs(A[:i])
                righty = dfs(A[i+1:])
                for l in lefty:
                    for r in righty:
                        curr = TreeNode(0)
                        curr.left = l
                        curr.right = r
                        current.append(curr)
        res[str(A)] = current
        return current
    return dfs(list(range(N)))

###############################

def f763(strs):
    start, res = 0, []
    while start < len(strs):
        end = len(strs) - strs[::-1].find(strs[start])
        if end > start:
            end = max(len(strs) - strs[::-1].find(i) for i in set(strs[start:end]))
            res.append(end - start)
            start = end - 1
        start += 1
    return res

def f763_2(strs):
    d, res, l, r = {}, [], 0, 0
    for i, v in enumerate(strs):
        d[v] = i
    for i in range(len(strs)):
        r = max(r, d[strs[i]])
        if i == r:
            res.append(i-l+1)
            l = i + 1
        return res

    
###############################

def f814(root):
    if not root:
        return None
    root.left = f814(root.left)
    root.right = f814(root.right)
    if not root.left and not root.right and not root.val: return None
    return root

###############################

def f890(words, pat):
    res = []
    for w in words:
        if len(w) == len(pat) and [w.find(i) for i in w] == [pat.find(j) for j in pat]:
            res.append(w)
        return res

#OR

#return [w for w in words if [w.index(c) for c in w] == [pat.index(c) for c in pat]]



###############################


def f797(graph):
    def dfs(cur, path):
        if cur == len(graph) - 1:
            res.append(path)
        else:
            for i in graph[cur]:
                dfs(i, path + [i])
    res = []
    dfs(0, [0])
    return res



###############################


def f1104(label):
    level, tot = -1, 0
    while label > tot:
        level += 1
        tot += (2 ** level)
    
    level -= 1
    cur = label // 2
    res = [label]
    while level > -1:
        st, end = 2 ** level, (2 **(level+1)) - 1
        cur = st + end - cur
        res.append(cur)
        level -= 1
        cur = cur // 2
    return res[::-1]

###############################

def f921(S):
    r, l = 0, []
    for s in S:
        if s == "(":
            l.append(s)
        elif l:   # s == ")" and l
            l.pop()
        else:
            r += 1
    return r + len(l)

###############################

def f861(A):
    for i in range(len(A)):
        if A[i][0] == 0:
            for j in range(len(A[0])):
                A[i][j] = 1 - A[i][j] #flip all the rows, that has a 0 in the front.
    
    base = 1
    sumtotal  = 0
    for j in range(len(A[0])-1, -1, -1):
        sumcol = sum([A[i][j] for j in range(len(A))])
        sumtotal += base * max(sumcol, len(A) - sumcol)
        base = base * 2
        
    return sumtotal
        

###############################

def f979(self, root):
    self.res = 0
    def dfs(node):
        if not node: return 0
        left = dfs(root.left)
        right = dfs(root.right)
        self.res += abs(left) + abs(right)
        return node.val + left + right - 1
    dfs(root)
    return self.res

def f979_2(root):
    def dfs(node):
        if not node: return 0, 9
        (lbal, lcnt), (rbal, rcnt) = dfs(node.left), dfs(node.right)  
        #(value, accumulated moves)
        bal = node.val + lbal + rbal - 1
        return bal, lcnt + rcnt + abs(bal)
    return dfs(root)[1]

        
        
###############################

def f968(root):
    if not root.left and not root.right:
        return 1
    res = []
    d = {root: None}
    q = collections.deque([root])
    while q:
        node = q.popleft()
        if node.left:
            d[node.left] = node
            q.append(node.left)
        if node.right:
            d[node.right] = node
            q.append(node.right)
        res.append(node)
        
    dp = {}
    for i in range(len(res) -1, -1, -1):
        parent = d[res[i]]
        if res[i] in dp or res[i].left in dp or res[i].right in dp:
            continue
        dp[parent] = 1
    return sum(dp.values())

###############################

def f517(machines):
    total, n = sum(machines), len(machines)
    if total % n: return -1
    target, res, toRight = total/n, 0, 0 
    #toRight means the clothes that we need to pass to the right number.
    for m in machines:
        toRight = m + toRight - target
        res = max(res, abs(toRight), m - target)
    return res


def f517_2(machines):
    total, n = sum(machines), len(machines)
    if total % n: return -1
    target, ans, send_out = total/n, 0, [0]*n 
    for i in range(n-1):
        if machines[i] > target:
            send_out[i] += machines[i] - target
        elif machines[i] < target:
            send_out[i+1] = target - machines[i]
        machines[i+1] += machines[i] - target
        ans = max(ans, send_out[i], send_out[i+1])
    return ans

###############################

###############################


def f366(root):
    def dfs(node):
        if not node:
            return -1
        i = 1 + max(dfs(node.left), dfs(node.right))  #i is the level from leaves
        if i == len(res):
            res.append([])
        res[i].append(node.val)
        return i
    res = []
    dfs(root)
    return res

###############################

def f419(board):
    if len(board) == 0: return 0
    m, n = len(board), len(board[0])
    cnt  = 0
    for i in range(m):
        for j in range(n):
            if board[i][j] == 'X' and \
            (i == 0 or board[i-1][j] == '.') and \
            (j == 0 or board[i][j-1] == '.'):
                cnt += 1
    return cnt

###############################

def f537(a, b):
    a1, a2 = map(int, a[:-1].split('+'))
    b1, b2 = map(int, b[:-1].split('+'))
    return '%d+d%i' %(a1*b1 - a2*b2, a1*b2 + a2*b1)

###############################

def f338(N):
    res = []
    for n in range(N+1):
        cnt  = 0
        while n:
            if n%2: cnt += 1
            n //= 2
        res.append(cnt)
    return res

#OR
    for n in range(N+1):
        res += bin(n).count('1')
    return res

#OR
    res = [0]
    while len(res) < N+1:
        res += [ i + 1 for i in res]
    return res[:N+1]

###############################

def f54(M):
    if not M: return []
    l, r, u, d, res = 0, len(M[0])-1, 0, len(M)-1, []
    while l<=r and u <=d:
        res.extend(M[u][l:r+1]) #left to right
        u += 1
        for i in range(u, d+1):
            res.append(M[i][r]) #up to down
        r -= 1
        if u <= d:
            res.extend(M[d][l:r+1][::-1]) #right to left
            d -= 1
        if l <= r:
            for i in range(d, u-1, -1):
                res.append(M[i][l]) #down to up
            l += 1
    return res

def f54_2(M):
    return M and [*M.pop(0)] + f54_2([*zip(*M)][::-1])

###############################
            
def f59(N):
    A, low = [], N*N+1
    while low > 1:
        low, high = low - len(A), low
        A = [range(low, high)] + zip(*A[::-1])
    return A


def f59_2(N):
    if not N:
        return []
    res = [[0]*N]*N
    l, r, u, d, n = 0, N-1, 0, N-1, 1
    while l <= r and u <= d:
        for i in range(l, r+1):
            res[u][i] = n
            n += 1
        u += 1
        
        for i in range(u, d+1):
            res[i][r] = n
            n += 1
        r -= 1
        
        for i in range(r, l-1, -1):
            res[d][i] = n
            n += 1
        d -= 1
        
        for i in range(d, u-1, -1):
            res[i][l] = n
            n += 1
        l += 1
        
    return res
    
###############################

def f885(R, C, r0, c0):
    res, n, k = [[r0, c0]], R*C, 1
    direct, ind = [(-1, 0), (0, 1), (1, 0), (0, -1)], 1
    while len(res) < n:
        for _ in range(2):
            for _ in range(k):
                r0 += direct[ind][0]
                c0 += direct[ind][1]
                if 0<=r0<R and 0<=c0<C:
                    res.append([r0,c0])
            ind = (ind + 1)%4
        k += 1  # k*2 path = 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, ....
    return res

###############################

##07/10

def f750(grid):
    if not grid or len(grid) < 2 or len(grid[0]) < 2:
        return 0
    m, n, res = len(grid), len(grid[0]), 0
    for i in range(m):
        for k in range(n):
            if grid[i][k] == 1:
                cnt = 0
                for j in range(i+1, m):
                    if grid[j][k]==1:
                        cnt += 1
                res += cnt * (cnt - 1) / 2;
    return res

###############################

def f1111(seq):
    q, res = [], [0]*len(seq)
    for i, c in enumerate(seq):
        if c == "(":
            q.append(i if not q or q[-1]<0 else -i)
        else:
            ind = q.pop()
            if ind >= 0:
                res[i] = res[ind] = 1
                
#( is 1 point, ) is -1 point.We try to keep total points of two groups even, by distributing parentheses alternatively.
def f1111_2(self, seq):
    A = B = 0
    res = [0] * len(seq)
    for i, c in enumerate(seq):
        v = 1 if c == '(' else -1
        if (v > 0) == (A < B):
            A += v
        else:
            B += v
            res[i] = 1
    return res

###############################

def f1123(root):
    def helper(node):
        if not node:
            return 0, None
        h1, lca1 = helper(node.left)
        h2, lca2 = helper(node.right)
        if h1 > h2:
            return h1 + 1, lca1
        if h1 < h2:
            return h2 + 1, lca2
        return h1 + 1, root
    return helper(root)[1]

###############################

def f723(board):
    R, C = len(board), len(board[0])
    changed = True
    
    while changed:
        changed = False
        for r in range(R):
            for c in range(C-2):
                if abs(board[r][c]) == abs(board[r][c+1]) == abs(board[r][c+2]) != 0:
                    board[r][c] = board[r][c+1] = board[r][c+2] = 0
                    change = True
                    
        for r in range(R-2):
            for c in range(C):
                if abs(board[r][c]) == abs(board[r+1][c]) == abs(board[r+2][c]) != 0:
                    board[r][c] = board[r+1][c] = board[r+2][c] = 0
                    change = True


        for c in range(C):
            i = R - 1
            for r in range(R-1, -1, -1):  #reversed(range(R))
                if board[r][c]>0:
                    board[i][c] = board[r][c]
                    i -= 1
            for r in reversed(range(i+1)):
                board[r][c] = 0
                
    return board

###############################

def f986(a, b):
    i, j, res = 0, 0, []
    while i < len(a) and j < len(b):
        if a[i][1] < b[j][0]:
            i += 1
        elif b[j][1] < a[i][0]:
            j += 1
        else:
            res.append([max(a[i][0], b[j][0]), min(a[i][1], b[j][1])])
            if a[i][1] > b[j][1]:
                j += 1
            else:
                i += 1
    return res

def f986_2(a, b):
    c = sorted(a+b, key=lambda x: x[0])
    res = []
    for i in range(1, len(c)):
        if c[i-1][1] < c[i][0]:
            continue
        else:
            res.append([c[i][0], min(c[i-1][1], c[i][1])])
            
    return res

###############################

def f912_quicksort(N):
    if len(N) < 2:
        return N
    pivot = random.choice(N)
    lt = [v for v in N if v < pivot]
    eq = [v for v in N if v == pivot]
    gt = [v for v in N if v > pivot]
    
    return f912_quicksort(lt) + eq + f912_quicksort(gt)

def f912_insertsort(N):
    for i in range(1, len(N)):
        key = N[i]
        j = i - 1
        while j >= 0 and key < N[j]:
            N[j+1] = N[j]
            j -= 1
        N[j+1] = key
    return N


def f913_mergesort(N):
    def merge(A, B):
        c = []
        while A and B:
            c.append(A.pop(0)) if A[0] < B[0] else c.append(B.pop(0))
        return c + (A or B)
    
    n = len(N)
    return N if n < 2 else merge(f913_mergesort(N[:n//2]), f913_mergesort(N[n//2:]))


###############################

import functools

def f1101(self,logs):
    
    def findp(res,a,b):
        for i in range(len(res)):
            if a in res[i] or b in res[i]:
                res[i].add(a)
                res[i].add(b)
                break
        else:
            res.append({a,b})
        #print(res)
        #check if there is a common person across all groups
        #if there is, this is the first time all people got connected
        return functools.reduce(lambda a,b:a&b, res)
    
    logs = sorted(logs,key=lambda x: x[0]) #sort by timestamp
    
    res=[{logs[0][1],logs[0][2]}]
    for k in range(1, len(logs)):
        if findp(res,logs[k][1],logs[k][2]):
            return logs[k][0]
    return -1



p = {} # p is the parent dict
def find(x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return p[x]
def union(x, y):
    p[x] = p.setdefault(x, x)
    p[y] = p.setdefault(y, y)
    print(p, find(x), find(y))
    p[find(x)] = find(y)

union((0,1,2,3),(3,4,5))
print(p)

#{(0, 1, 2, 3): (0, 1, 2, 3), (3, 4, 5): (3, 4, 5)} (0, 1, 2, 3) (3, 4, 5)
#{(0, 1, 2, 3): (3, 4, 5), (3, 4, 5): (3, 4, 5)}
###############################

def f791(S, T):
    return ''.join(sorted(T, key=lambda x: S.find(x)))

###############################

def f1110(root, to_delete):
    to_del = set(to_delete)
    res = []
    def helper(root, is_root):
        if not root:
            return None
        root_deleted = root.val in to_del
        if is_root and not root_deleted:
            res.append(root)
        root.left = helper(root.left, root_deleted)
        root.right = helper(root.right,root_deleted)
        return None if root_deleted else root
    
    helper(root, True)
    return res

###############################

def f959(self, grid):
    f = {}
    def find(x):
        f.setdefault(x, x)
        if x != f[x]:
            f[x] = find(f[x])
        return f[x]
    def union(x, y):
        f[find(x)] = find(y)

    for i in range(len(grid)):
        for j in range(len(grid)):
            if i:  # i > 0  union to the top area
                union((i - 1, j, 2), (i, j, 0))
            if j:  # j > 0  union to the left area
                union((i, j - 1, 1), (i, j, 3))
            if grid[i][j] != "/":
                union((i, j, 0), (i, j, 1))
                union((i, j, 2), (i, j, 3))
            if grid[i][j] != "\\":
                union((i, j, 3), (i, j, 0))
                union((i, j, 1), (i, j, 2))
    return len(set(map(find, f)))


###############################


#To get dp[i], we will try to change k last numbers separately to the maximum of them,

def f1043(A, K):
    n = len(A)
    dp = [0] * n
    curMax = 0
    for i in range(n):
        if i < K: 
            curMax = max(curMax, A[i])
            dp[i] = curMax * (i + 1)
        else:
            curMax = 0
            for j in range(1, K + 1):
                curMax = max(A[i - j + 1], curMax)
                dp[i] = max(dp[i], dp[i - j] + curMax * j)
    return dp[n - 1]



def f1043_2(A, K):
    N = len(A)
    dp = [0] * (N + 1)
    for i in range(N):
        curMax = 0
        for k in range(1, min(K, i + 1) + 1):
            curMax = max(curMax, A[i - k + 1])
            dp[i] = max(dp[i], dp[i - k] + curMax * k)
    return dp[N - 1]

###############################

def f969(A):
    res = []
    for x in range(len(A), 1, -1):
        i = A.index(x)
        res.extend([i+1, x])
        A = A[:i:-1] + A[:i]
    return res

###############################


def f1061(A, B, S):
    f = {}
    def find(x):
        f.setdefault(x, x)
        if x != f[x]:
            f[x] = find(f[x])
        return f[x]
    def union(x, y):
        if find(x) < find(y):
            f[find(y)] = find(x)
        else:
            f[find(x)] = find(y)
    #########
    p = dict()
    def find1(c):
        p.setdefault(c,c)
        if c!=p[c]:
            p[c]=find(p[c])
        return p[c]

    def union1(a,b):
        c1,c2=find(a),find(b)
        if(c1<c2):
            p[c2]=c1
        else:
            p[c1]=c2
    ########
        
    for i in range(len(A)):
        if find(A[i]) != find(B[i]):
            union(A[i],B[i])
        res = ""
    for j in range(len(S)):
        res += find(S[j])
    print(res)
    return res

A,B,S = "parker","morris","parser"
print(f1061(A,B,S))

###############################

def f877(piles):
    cache = {}
    piles = tuple(piles)
    def firstscore(i,j):
        if i>=j: return 0
        if j==i+1 and j < len(piles): return piles[i]
        if (i,j) in cache: return cache[i,j]
        res = max(piles[i]+min(firstscore(i+2,j), firstscore(i+1,j-1)) , piles[j-1] + min(firstscore(i+1,j-1), firstscore(i,j-2)))
        cache[i,j] = res
        return res

    Alex = firstscore(0,len(piles))
    Lee = sum(piles) - Alex
    return Alex > Lee


###############################

def f107(root):
    res = []
    if not root:
        return res
    q = collections.deque([root])
    while q:
        rec = []
        for i in range(len(q)):
            node = q.popleft()
            rec.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        res.append(rec)
    return res[::-1]

###############################

###############################
    
def f103(root):
    res = []
    if not root:
        return res
    q=collections.deque([root])
    level = 1
    while q:
        rec = []
        for i in range(len(q)):
            node = q.popleft()
            rec.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        if level%2: res.append(rec)
        else: res.append(rec[::-1])
        level += 1
        
    return res

###############################

def f542(M):
    m, n = len(M), len(M[0])
    q = collections.deque([])
    visited = set()
    for i in range(m):
        for j in range(n):
            if M[i][j] == 0:
                q.append((i, j))
                visited.add((i, j))
    while q:
        i, j = q.popleft()
        for x, y in [(i+1,j),(i-1,j),(i,j-1),(i,j+1)]:
            if 0<=x<m and 0<=y<n and (x, y) not in visited:
                M[x][y] = M[i][j] + 1
                visited.add((x,y))
                q.append((x,y))
    return M


###############################

def f279(N):
    q = collections.deque([N])
    visited = set()
    level = 0
    while q:
        for i in range(len(q)):
            n = q.popleft()
            if n == 0:
                return level
            for i in range(1, int(n**0.5)+1):
                val = n - i**2
                if val in visited:
                    continue
                q.append(val)
                visited.add(val)
            level += 1
        return -1

###############################


def f752(deadend, target):
    q = collections.deque(['0000'])
    visited = set(deadend)
    level = 0
    while q:
        for i in range(len(q)):
            s = q.popleft()
            if s == target:
                return level
            #if s in visited: continue
            #visited.add(s)
            for i, c in enumerate(s):
                v1 = s[:i] + str(abs((int(c)+1)%10)) + s[i+1:]
                v2 = s[:i] + str(abs((int(c)-1)%10)) + s[i+1:]
                if (v1==target and v1 not in visited) or (v2==target and v2 not in visited): 
                    return level + 1
                else:
                    if v1 not in visited:
                        q.append(v1)
                        visited.add(v1)
                    if v2 not in visited:
                        q.append(v2)
                        visited.add(v2)
        level += 1
    return -1

###############################

def f127(word1, word2, wlist):
    d = collections.defaultdict(list)
    visited = set()
    for s in wlist:
        for i in range(len(s)):
            d[s[i:]+"_"+s[i+1:]].append(s)
    
    q = collections.deque([word1])
    level = 1
    while q:
        for i in range(len(q)):
            s = q.popleft()
            if s == word2:
                return level
            for j in range(len(s)):
                for s1 in d[s[:j]+"_"+s[j+1:]]:
                    if s1 not in visited:
                        q.append(s1)
                        visited.add(s1)
        level += 1
    return 0

import string
def f127_2(word1, word2, wlist):
        wordList = set(wlist)
        queue = collections.deque([[word1, 1]])
        while queue:
            s, level = queue.popleft()
            if s == word2:
                return level
            for i in range(len(s)):
                for c in string.ascii_lowercase:
                    next_word = s[:i] + c + s[i+1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        queue.append([next_word, level + 1])
        return 0

###############################

def f815(routes, S, T):
    d = collections.defaultdict(set)
    visited = set([S])
    for i in range(len(routes)):
        for val in routes[i]:
            d[val].add(i)
    level = 0
    q = collections.deque([S])
    while q:
        for i in range(len(q)):
            node = q.popleft()
            if node == T: return level
            for idx in d[node]:
                for st in routes[idx]:
                    if st not in visited:
                        visited.add(st)
                        q.append(st)
        level += 1
    return -1

###############################

def f1034(grid, r0, c0, color):
    m, n = len(grid), len(grid[0])
    oldcolor = grid[r0][c0]
    connected = set()
    
    #identify the connected area
    def dfs(grid, i, j):
        for x, y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
            if 0<=x<m and 0<=y<n and (x, y) not in connected and \
            grid[x][y] == oldcolor:
                connected.add((x, y))
                dfs(grid, x, y)

    connected.add((r0, c0))
    dfs(grid, r0, c0)
    
    #check if the i, j on the border
    def checkborder(i, j):
        if i in (0, m-1) or j in (0, n-1):
            return True
        for x, y in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
            if grid[x][y] != oldcolor:
                return True
        return False
    
    res = []
    for x, y in connected:
        if checkborder(x, y):
            res.append((x, y))
    
    for x, y in res: 
        grid[x][y]= color
        
    return grid


###############################


def f996(self, A):
    c = collections.Counter(A)
    cand = {i: {j for j in c if int((i + j)**0.5) ** 2 == i + j} for i in c}
    self.res = 0
    def dfs(x, left=len(A) - 1):
        c[x] -= 1
        if left == 0: self.res += 1
        for y in cand[x]:
            if c[y]: dfs(y, left - 1)
        c[x] += 1
    for x in c: dfs(x)
    return self.res

###############################

#07/12/2020

import bisect
def f995(A, K):
    res, s = 0, []
    for i in range(len(A)):
        idx = bisect.bisect_right(s, i)
        if (len(s)-idx) % 2 :
            A[i] = 1 - A[i]
        if A[i] ==1: continue
        if len(A) - i < K:
            return -1
        res += 1
        s.append(i + K)
    return res

def f995_2(A, K):
    cur = res = 0
    for i in range(len(A)):
        if i >= K and A[i-K] == 2:  #when window moves forward, reduce its cur by minus those flips falling out behind the window
            cur -= 1
        if (cur % 2) == A[i]: # cur is even and A[i] == 0 or cur is odd and A[i] == 1
            if i + K > len(A):
                return -1
            A[i] = 2
            cur, res = cur + 1, res + 1
    return res


###############################

def f1001(N, lamps, queries):
    lampon = set()
    rowon = colon = diagTL = diagBL = dict()
    for r, c in lamps:
        lampon.add((r,c))
        rowon[r] += 1
        colon[c] += 1
        diagTL[c-r] += 1
        diagBL[c+r-N] += 1
    res = []
    for r, c in queries:
        if rowon[r] > 0 or colon[c] > 0 or diagTL[c-r] > 0 or diagBL[c+r-N] >0:
            res.append(1)
        else:
            res.append(0)
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                x, y = r+dx, c+dy
                if (x, y) in lampon:
                    rowon[r] -= 1
                    colon[c] -= 1
                    diagTL[c-r] -= 1
                    diagBL[c+r-N] -= 1
                    lampon.remove((r,c))
    return res

###############################

def ff654(nums):
    if not nums:
        return None
    i = nums.index(max(nums))
    node = TreeNode(nums[i])
    node.left = ff654(nums[:i])
    node.right = ff654(nums[i+1:])
                      
    return node
                      
###############################
                      
def f998(root, val):
    if not root: return TreeNode(val)
    if val > root.val:
        node = TreeNode(val)
        node.left = root
        return node
    else:
        root.right = f998(root.right, val)
        return root
    
def f998_2(root, val):
    pre, cur = None, root
    while cur and cur.val > val:
        pre, cur = cur, cur.right
    node = TreeNode(val)
    node.left = cur
    if pre: pre.right = node
    return root if root.val > val else node
    
###############################

def f999(board):
    rook, m, n = [], len(board), len(board[0])
    for i in range(m):
        for j in range(n):
            if board[m][n] == 'R':
                rook.append([i, j])
                i, j = m, n
                
    res = 0
    x0, y0 = rook.pop()
    for i, j in [[0,-1],[-1,0],[0,1],[1,0]]:
        x, y = i + x0, j + y0
        while 0 <= x < m and 0 <= y < n:
            if board[x][y] == 'p': res += 1
            if board[x][y] != '.': break
            x, y = x + i, y + j
    return 

###############################


def f997(N, trust):
    dic, map = collections.defaultdict(list), set()
    for i, j in trust:
        if not dic[i]:
            dic[i].append(j)
            map.add(j)
    return map - dic.keys() + 1
    
def f997_2(N, trust):
    count = [0] * (N + 1)
    for i, j in trust:
        count[i] -= 1
        count[j] += 1
    for i in range(1, N + 1):
        if count[i] == N - 1:
            return i
    return -1


###############################

def f992(A, K):
    def helper(A, K):
        i, res = 0, 0
        d = collections.Counter()
        for j in range(len(A)):
            d[A[j]] += 1
            if d[A[j]] == 1: 
                K -= 1
                
            while K < 0:    #while there are more than K distinct char 
                d[A[i]] -= 1
                if d[A[i]] == 0:
                    K += 1
                i += 1  #remove the char from the left side
                res += j - i + 1 #get the window sides == the number of new substring
        return res
    return helper(A, K) - helper(A, K-1)
                  
def f992_2(A, K):
    def atMostK(A, K):
        count = collections.Counter()
        res = i = 0
        for j in range(len(A)):
            if count[A[j]] == 0: K -= 1
            count[A[j]] += 1
            while K < 0:
                count[A[i]] -= 1
                if count[A[i]] == 0: K += 1
                i += 1
            res += j - i + 1
        return res
    return atMostK(A, K) - atMostK(A, K - 1)
                  

###############################

def f991(X, Y):
    if X >= Y:
        return X - Y
    res = 0
    while X < Y:
        if Y % 2 == 1:
            Y += 1
            res += 1
        Y //= 2
        res += 1
    return res + X - Y

#The difference between DP and greedy is: Greedy algo requires that the result of the whole question is 
# determined by the result of sub-question. For this question, 
# we can prove that f(X, Y) = f(X, Y/2) + 1, if Y is even or f(X, Y + 1) + 1 if Y is odd. 
# Therefore, it could be solved simply by greedy algorithm. 
# If you consider it as a dp problem, the formula should be f(X, Y) = min(f(2X, Y), f(X-1, Y)) + 1. 
# If you want to calculate f(X, Y), you need to calculate f(2X, Y) and f(X-1, Y).



###############################



