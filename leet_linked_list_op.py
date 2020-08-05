import queue

#single linked list helper functions
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
    
    def has_value(self,value):
        if self.val == value:
            return True
        else:
            return False

class SingleLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        return
    
    def append_list_item(self,item):
        if not isinstance(item, ListNode):
            item = ListNode(item)
        if self.head is None:
            self.head = item
        else:
            self.tail.next = item
        self.tail = item
        return

    def list_length(self):
        count = 0
        tmp = self.head
        while tmp is not None:
            count += 1
            tmp = tmp.next
        return count

    def output_list(self):
        tmp = self.head
        while tmp is not None:
            print(tmp.val)
            tmp = tmp.next
        return

    def unordered_search(self, value):
        tmp = self.head
        count = 0
        outs = []
        while tmp is not None:
            count += 1
            if tmp.has_value(value):
                outs.append(count)
            tmp = tmp.next
        return outs

    def remove_list_item(self, nid):
        tmp = self.head
        prev = None
        count = 0
        while tmp is not None:
            count += 1
            if count == nid:
                if prev is not None:
                    prev.next = tmp.next
                else:
                    self.head = tmp.next #remove the head
                return
            prev = tmp
            tmp = tmp.next
        return

    def insert_list_item(self, item, nid):
        if not isinstance(item, ListNode):
            item = ListNode(item)
        tmp = self.head
        if tmp is None:
            self.head = item
            self.tail = item
            return
        count = 0
        while tmp is not None:
            if count == nid - 1:
                save_next = tmp.next
                tmp.next = item
                item.next = save_next
                return
            tmp = tmp.next
            count += 1
    



##################################################################
# 2. Add two numbers which digits are connected in linked lists
# use mod(10) and carry

class solution:
    def AddTwoNumbers(self, n1, n2):
        head = tmp = ListNode(0)
        carry = 0
        while n1 or n2 or carry:
            tmp1 = n1.val if n1 else 0
            tmp2 = n2.val if n2 else 0
            sumtmp = tmp1 + tmp2 + carry
            tmp.next = ListNode(int(sumtmp % 10))
            tmp = tmp.next
            carry = sumtmp // 10
            if n1:
                n1 = n1.next
            if n2:
                n2 = n2.next
        return head.next


#node1 = ListNode(2) 
#node2 = ListNode(4) 
#node3 = ListNode(3) 
#node1.next=node2
#node2.next=node3

#node11 = ListNode(5) 
#node12 = ListNode(6) 
#node13 = ListNode(4) 
#node11.next=node12
#node12.next=node13

#n = a.AddTwoNumbers(node1,node11)
#while n is not None:
#    print("current linked digit value=", n.val)
#    n = n.next

##################################################################
# 10. Remove the Nth node from end of list
#use N steps of delay

    def RemoveNfromEnd(self, n, sl):
        slow = fast = hd = sl.head
        while fast.next:
            if (n):    #n steps of delay between slow and fast
                n -= 1
            else:
                slow = slow.next
            fast = fast.next
        #when fast reaches to the end of the list, remove the node
        slow.next = slow.next.next
        return hd
    

#node1 = ListNode(1) 
#node2 = ListNode(2) 
#node3 = ListNode(3) 
#node4 = ListNode(4) 
#node5 = ListNode(5) 
#t = SingleLinkedList()
#for i in [node1, node2, node3, node4, node5]:  
#    t.append_list_item(i)
#t.output_list()
#print("===")
#n=2
#a.RemoveNfromEnd(n, t)
#t.output_list()

##################################################################
# 21. Merge two sorted lists
# use two ways - recursive and iterative methods

#recursive
    def MergeTwoSL(self, n1, n2):
        if not n1 or not n2:
            return n1 or n2
        if n1.val < n2.val:
            n1.next = self.MergeTwoSL(n1.next, n2)
            return n1
        else:
            n2.next = self.MergeTwoSL(n1, n2.next)
            return n2

#iterative
    def MergeTwoSL2(self, n1, n2):
        head = tmp = ListNode(0)
        while n1 and n2:
            if n1.val < n2.val:
                tmp.next = n1
                n1 = n1.next
            else:
                tmp.next = n2
                n2 = n2.next
            tmp = tmp.next
        tmp.next = n1 or n2
        return head.next

#node1 = ListNode(1) 
#node2 = ListNode(3) 
#node3 = ListNode(4) 
#node4 = ListNode(5) 
#node1.next=node2
#node2.next=node3
#node3.next=node4

#node11 = ListNode(2) 
#node12 = ListNode(4) 
#node13 = ListNode(6) 
#node14 = ListNode(8) 
#node11.next=node12
#node12.next=node13
#node13.next=node14

#t = a.MergeTwoSL(node1,node11)
#t = a.MergeTwoSL2(node1,node11)
#while t is not None:
#    print("current linked digit value=", t.val)
#    t = t.next

##################################################################
# 22. Merge k sorted lists
# use brute force
    def MergeKSL(self, ls):
        nodes = []   
        for k in ls:
            while k:
                nodes.append(k.val)   #append all value of the nodes
                k = k.next
        head = tmp = ListNode(0)
        for t in sorted(nodes):   #sort the value and output the linked list
            tmp.next = ListNode(t)
            tmp = tmp.next
        return head.next

# alternative - use priority queue
    def MergeKSL2(self, ls):
        pq = queue.PriorityQueue()
        for k in ls:
            if k:
                pq.put((k.val, k))  #priority queue can auto-sort the nodes
        head = tmp = ListNode(0)
        while not pq.empty():
            val, nd = pq.get()  #pull out the node and link it to the list
            tmp.next = ListNode(val)
            tmp = tmp.next
            nd = nd.next  #break the old link and put the next node back to the queue
            if nd:
                pq.put(nd.val, nd)
        return head.next


##################################################################
# 24. Swap nodes in pairs

    def swapPair(self, head):
        dummy = pre = ListNode(0)
        pre.next = head
        while pre.next and pre.next.next:
            a = pre.next   #define a, b pairs
            b = a.next
            pre.next, a.next, b.next = b, b.next, a
            pre = a
        return dummy.next


##################################################################
# 25. Reverse node in k groups
# form one group for every k nodes, reverse the nodes in each group 
# use recursive

    def reverseKGroup(self, head, k):
        if head:
            node = head
            for _ in range(k - 1):
                node = node.next
                if not node:
                    return head  #if length is shorter than k, return head
            prev = self.reverseKGroup(node.next, k) #call for the next group
            while prev is not node: #reverse the head to the node
                prev, head.next, head = head, prev, head.next
                #head = head.next
                #head.next = prev
                #prev = head
            return prev

#####################################################
# 83. Remove dups from sorted list
#

    def RemoveDupsSortedList(self, head):
        node = head
        while node and node.next:
            if node.val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        return head


#####################################################
# 141. Linked list cycle
#Given a linked list, determine if it has a cycle in it.
# use fast and slow - slow will be caught up by fast if there is a cycle in it
    def LinkedListCycle(self, head):
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                return True
        return False

#####################################################
# 160. Intersection of two linked list
# find the node at which the intersection of two singly linked lists begins

    def IntersectionTwoLinkedLists(self, h1, h2):
        if h1 is None or h2 is None:
            return None
        p1, p2 = h1, h2
        while p1 is not p2:
            p1 = h2 if p1 is None else p1.next
            p2 = h1 if p2 is None else p2.next
        return p1


#####################################################
# 203. Remove linked list elements

    def RemoveLinkedList(self, head, val):
        pre = ListNode(0)
        pre.next = current = head
        while current:
            if current.val == val:
                if pre:
                    pre.next = current.next
                else:
                    head = current.next  #remove head
            pre = current
            current = current.next
        return head

#####################################################
# 206. Reverse linked list

    def ReverseLinkedList(self, head):
        tail = None
        while head:
            tmp = head.next
            head.next = tail
            tail = head
            head = tmp
        return tail

    def ReverseLinkedList2(self, head):
        new_head = None
        while head:
            nexte = head.next
            head.next = new_head
            new_head = head
            head = nexte
        return new_head

    def reverse_recursively(self, head, new_head):
        if not head:
            return new_head
        next = head.next
        head.next = new_head
        return self.reverse_recursively(next, head)

    def ReverseLinkedList3(self, head):
        return self.reverse_recursively(head, None)

    def ReverseLinkedList4(self, head, prev=None):
        if not head:
            return prev
        next = head.next
        head.next = prev
        return self.ReverseLinkedList4(next, head)



#####################################################
# 234. Palindrome linked list
# Given a singly linked list, determine if it is a palindrome.
# use list to list each value in sequence

    def PalindromeLinkedList(self, head):
        vals = []
        while head:
            vals += head.val
            head = head.next
        return vals == vals[::-1]

#####################################################
# 237. Delete node in a linked list
# write a function to delete a node (except the tail) in a singly linked list

    def DeleteNodeLinkedList(self, node):
        if node == None: 
            return None
        if node.next:
            node.val = node.next.val
            node.next = node.next.next

#####################################################
# 707. Design linked list
# Design your implementation of the linked list. You can choose to use the singly linked list or the doubly linked list. A node in a singly linked list should have two attributes: val and next. val is the value of the current node, and next is a pointer/reference to the next node. If you want to use the doubly linked list, you will need one more attribute prev to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.
#Implement these functions in your linked list class:

#get(index) : Get the value of the index-th node in the linked list. If the index is invalid, return -1.
#addAtHead(val) : Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
#addAtTail(val) : Append a node of value val to the last element of the linked list.
#addAtIndex(index, val) : Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. If index is negative, the node will be inserted at the head of the list.
#deleteAtIndex(index) : Delete the index-th node in the linked list, if the index is valid.
#Example:

#MyLinkedList linkedList = new MyLinkedList();
#linkedList.addAtHead(1);
#linkedList.addAtTail(3);
#linkedList.addAtIndex(1, 2);  // linked list becomes 1->2->3
#linkedList.get(1);            // returns 2
#linkedList.deleteAtIndex(1);  // now the linked list is 1->3
#linkedList.get(1);            // returns 3

# double linked list
class Node:
    def __init__(self, value):
        self.val = value
        self.next = self.pre = None

class MyLinkedList:
    def __init__(self):
        self.head = self.tail = Node(-1)
        self.head.next = self.tail
        self.tail.pre = self.head
        self.size = 0
    
    def add(self, preNode, val):
        node = Node(val)
        node.pre = preNode
        node.next = preNode.next
        node.pre.next = node.next.pre = node
        self.size += 1

    def remove(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre
        self.size -= 1

    def forward(self, start, end, cur):
        while start != end:
            start += 1
            cur = cur.next
        return cur

    def backward(self, start, end, cur):
        while start != end:
            start -= 1
            cur = cur.pre
        return cur

    def get(self, index):
        if 0 <= index <= self.size // 2:
            return self.forward(0, index, self.head.next).val
        elif self.size // 2 < index < self.size:
            return self.backward(self.size -1, index, self.tail.pre).val
        return -1

    def addAtHead(self, val):
        self.add(self.head, val)
    
    def addAtTail(self, val):
        self.add(self.tail.pre, val)

    def addAtIndex(self, index, val):
        if 0 <= index <= self.size //2:
            self.add(self.forward(0, index, self.head.next).pre, val)
        elif self.size // 2 < index <= self.size:
            self.add(self.backward(self.size, index, self.tail).pre, val)

    def deleteAtIndex(self, index):
        if 0 <= index <= self.size // 2:
            self.remove(self.forward(0, index, self.head.next))
        elif self.size // 2 < index < self.size:
            self.remove(self.backward(self.size - 1, index, self.tail.pre))

        
## alternative single linked list 

class ListNode2:
    def __init__(self, x):
        self.val = x
        self.next = None

class MyLinkedList2:
    def __init__(self):
        self.head = None
        self.size = 0

    def get(self, index):
        if index < 0 or index >= self.size or self.head is None:
            return -1
        return self.findIndex(index).val
    
    def addAtHead(self, val):
        self.addAtIndex(0, val)

    def addAtTail(self, val):
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index, val):
        if index > self.size:
            return -1
        elif index == 0:
            head = ListNode2(val)
            head.next, self.head = self.head, head
        else:
            pre = self.findIndex(index - 1)
            cur = ListNode2(0)
            cur.next, pre.next = pre.next, cur
        self.size += 1

    def deleteAtIndex(self, index):
        if index < 0 or index >= self.size:
            return -1
        cur = self.findIndex(index - 1)
        cur.next = cur.next.next
        self.size -= 1

    def findIndex(self, index):
        cur = self.head
        for _ in range(index):
            cur = cur.next
        return cur

    
        
        



#####################################################
# 876. Middle of the linked list
# Given a non-empty, singly linked list with head node head, return a middle node of linked list.
# If there are two middle nodes, return the second middle node.

    def MiddleNode(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    