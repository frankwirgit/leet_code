======== Python database =========

======== Matplotlib =========

======== Pandas =========
#Merge two data frames append the second data frame as a new column to the first data frame.
import pandas as pd

Car_Price = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'Price': [23845, 17995, 135925 , 71400]}
carPriceDf = pd.DataFrame.from_dict(Car_Price)

car_Horsepower = {'Company': ['Toyota', 'Honda', 'BMV', 'Audi'], 'horsepower': [141, 80, 182 , 160]}
carsHorsepowerDf = pd.DataFrame.from_dict(car_Horsepower)

carsDf = pd.merge(carPriceDf, carsHorsepowerDf, on="Company")
print(carsDf)
  Company   Price  horsepower
0  Toyota   23845         141
1   Honda   17995          80
2     BMV  135925         182
3    Audi   71400         160

#Concatenate two data frames
import pandas as pd

GermanCars = {'Company': ['Ford', 'Mercedes', 'BMV', 'Audi'], 'Price': [23845, 171995, 135925 , 71400]}
carsDf1 = pd.DataFrame.from_dict(GermanCars)

japaneseCars = {'Company': ['Toyota', 'Honda', 'Nissan', 'Mitsubishi '], 'Price': [29995, 23600, 61500 , 58900]}
carsDf2 = pd.DataFrame.from_dict(japaneseCars)

carsDf = pd.concat([carsDf1, carsDf2], keys=["Germany", "Japan"])
print(carsDf)

               Company   Price                                           
Germany 0         Ford   23845
        1     Mercedes  171995
        2          BMV  135925
        3         Audi   71400
Japan   0       Toyota   29995
        1        Honda   23600
        2       Nissan   61500
        3  Mitsubishi    58900


#Sort all cars by Price column
carsDf = carsDf.sort_values(by=['price', 'horsepower'], ascending=False)
carsDf.head(5)

#Find the average mileage of each car making company
car_Manufacturers = df.groupby('company')
mileageDf = car_Manufacturers['company','average-mileage'].mean()
mileageDf

#Find each company’s Higesht price car
car_Manufacturers = df.groupby('company')
priceDf = car_Manufacturers['company','price'].max()
priceDf

#Count total cars per company
df['company'].value_counts()

#Print All Toyota Cars details
car_Manufacturers = df.groupby('company')
toyotaDf = car_Manufacturers.get_group('toyota')
toyotaDf

#Find the most expensive car company name
df = df [['company','price']][df.price==df['price'].max()]



#Replace all column values which contain ‘?’ and n.a with NaN.
df = pd.read_csv("D:\\Python\\Articles\\pandas\\automobile-dataset\\Automobile_data.csv", na_values={
'price':["?","n.a"],
'stroke':["?","n.a"],
'horsepower':["?","n.a"],
'peak-rpm':["?","n.a"],
'average-mileage':["?","n.a"]})
print (df)

df.to_csv("D:\\Python\\Articles\\pandas\\automobile-dataset\\Automobile_data.csv")

import pandas as pd
df = pd.read_csv("D:\\Python\\Articles\\pandas\\automobile-dataset\\Automobile_data.csv")
df.head(5)
df.tail(5)


dfObj = pd.DataFrame(columns=['User_ID', 'UserName', 'Action'])
column_names = ["a", "b", "c"]

df = pd.DataFrame(columns = column_names)



#Creating an empty dataframe with the same index and columns as another dataframe:

import pandas as pd
df_copy = pd.DataFrame().reindex_like(df_original)

======== Numpy =========
#create 2D array and plot
import numpy as np
import matplotlib.pyplot as plt

samplear=np.arange(100,200,10)
samplear=samplear.reshape(5,2)
print(samplear)
x=samplear[:,0:1]
y=samplear[:,1:2]

plt.plot(x,y)
plt.show()



#delete and insert column
import numpy

print("Printing Original array")
sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
print (sampleArray)

print("Array after deleting column 2 on axis 1")
sampleArray = numpy.delete(sampleArray , 1, axis = 1) 
print (sampleArray)

arr = numpy.array([[10,10,10]])

print("Array after inserting column 2 on axis 1")
sampleArray = numpy.insert(sampleArray , 1, arr, axis = 1) 
print (sampleArray)
Printing Original array
[[34 43 73]
 [82 22 12]
 [53 94 66]]
Array after deleting column 2 on axis 1
[[34 73]
 [82 12]
 [53 66]]
Array after inserting column 2 on axis 1
[[34 10 73]
 [82 10 12]
 [53 10 66]]


#2-D array. Print max from axis 0 and min from axis 1

import numpy

print("Printing Original array")
sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
print (sampleArray)


minOfAxisOne = numpy.amin(sampleArray, 1) 
print("Printing amin along columns - each row")
print(minOfAxisOne)

maxOfAxisZero = numpy.amax(sampleArray, 0) 
print("Printing amax along rows - each column")
print(maxOfAxisZero)

Printing Original array
[[34 43 73]
 [82 22 12]
 [53 94 66]]
Printing amin along columns - each row
[34 12 53]
Printing amax along rows - each column
[82 94 73]

#sort arry by row and column
import numpy

print("Printing Original array")
sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
print (sampleArray)

sortArrayByRow = sampleArray[:,sampleArray[1,:].argsort()]
print("Sorting Original array by secoond row")
print(sortArrayByRow)

print("Sorting Original array by secoond column")
sortArrayByColumn = sampleArray[sampleArray[:,1].argsort()]
print(sortArrayByColumn)

Printing Original array
[[34 43 73]
 [82 22 12]
 [53 94 66]]
Sorting Original array by secoond row
[[73 43 34]
 [12 22 82]
 [66 94 53]]
Sorting Original array by secoond column
[[82 22 12]
 [34 43 73]
 [53 94 66]]


#split array equally
import numpy

print("Creating 8X3 array using numpy.arange")
sampleArray = numpy.arange(10, 34, 1)
sampleArray = sampleArray.reshape(8,3)
print (sampleArray)

print("\nDividing 8X3 array into 4 sub array\n")
subArrays = numpy.split(sampleArray, 4) 
print(subArrays)
Creating 8X3 array using numpy.arange
[[10 11 12]
 [13 14 15]
 [16 17 18]
 [19 20 21]
 [22 23 24]
 [25 26 27]
 [28 29 30]
 [31 32 33]]

Dividing 8X3 array into 4 sub array

[array([[10, 11, 12],
       [13, 14, 15]]), array([[16, 17, 18],
       [19, 20, 21]]), array([[22, 23, 24],
       [25, 26, 27]]), array([[28, 29, 30],
       [31, 32, 33]])]


import numpy

arrayOne = numpy.array([[5, 6, 9], [21 ,18, 27]])
arrayTwo = numpy.array([[15 ,33, 24], [4 ,7, 1]])

resultArray  = arrayOne + arrayTwo
print("addition of two arrays is \n")
print(resultArray)

for num in numpy.nditer(resultArray, op_flags = ['readwrite']):
   num[...] = num*num
print("\nResult array after calculating the square root of all elements\n")
print(resultArray)

addition of two arrays is 

[[20 39 33]
 [25 25 28]]

Result array after calculating the square root of all elements

[[ 400 1521 1089]
 [ 625  625  784]]



import numpy

sampleArray = numpy.array([[3 ,6, 9, 12], [15 ,18, 21, 24], 
[27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]]) 
print("Printing Input Array")
print(sampleArray)

print("\n Printing array of odd rows and even columns")
newArray = sampleArray[::2, 1::2]
print(newArray)

Printing Input Array
[[ 3  6  9 12]
 [15 18 21 24]
 [27 30 33 36]
 [39 42 45 48]
 [51 54 57 60]]

 Printing array of odd rows and even columns
[[ 6 12]
 [30 36]
 [54 60]]



import numpy as np
x = np.array([[[1],[2],[3]], [[4],[5],[6]]])
#print(x)
#print(x.shape, x.ndim)
print(x[:,1:3,...])

[[[2]
  [3]]

 [[5]
  [6]]]


import numpy

sampleArray = numpy.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]]) 
print("Printing Input Array")
print(sampleArray)

print("\n Printing array of items in the third column from all rows")
newArray = sampleArray[...,2]
print(newArray)
Printing Input Array
[[11 22 33]
 [44 55 66]
 [77 88 99]]

 Printing array of items in the third column from all rows
[33 66 99]


import numpy

sampleArray = numpy.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]]) 
print("Printing Input Array")
print(sampleArray)

print("\n Printing array of items in the third column from all rows")
newArray = sampleArray[:,[0,2]]
print(newArray)
Printing Input Array
[[11 22 33]
 [44 55 66]
 [77 88 99]]

 Printing array of items in the third column from all rows
[[11 33]
 [44 66]
 [77 99]]

import numpy

print("Creating 5X2 array using numpy.arange")
sampleArray = numpy.arange(100, 200, 10)
print(sampleArray)
sampleArray = sampleArray.reshape(5,2)
print (sampleArray)

Creating 5X2 array using numpy.arange
[100 110 120 130 140 150 160 170 180 190]
[[100 110]
 [120 130]
 [140 150]
 [160 170]
 [180 190]]

import numpy

firstArray = numpy.empty([4,2], dtype = numpy.uint16) 
print("Printing Array")
print(firstArray)

print("Printing numpy array Attributes")
print("1> Array Shape is: ", firstArray.shape)
print("2>. Array dimensions are ", firstArray.ndim)
print("3>. Length of each element of array in bytes is ", firstArray.itemsize)

Printing Array
[[64757 19267]
 [45868 33230]
 [16280 34196]
 [32752     0]]
Printing numpy array Attributes
1> Array Shape is:  (4, 2)
2>. Array dimensions are  2
3>. Length of each element of array in bytes is  2


import numpy
a = numpy.zeros(shape=(4,2))

import numpy
print(numpy.empty([2, 2], dtype=int))
print(numpy.ones([2, 2], dtype=int))
print(numpy.zeros([2, 2], dtype=int))



#This should work:

from numpy import *

a = array([[1, 2, 3], [0, 3, NaN]])
where_are_NaNs = isnan(a)
a[where_are_NaNs] = 0


import numpy as geek 
  
in_arr = geek.array([[2, geek.inf, 2], [2, 2, geek.nan]]) 
   
print ("Input array : ", in_arr)  
    
out_arr = geek.nan_to_num(in_arr)  
print ("output array: ", out_arr) 


>>> a = numpy.empty((3,3,))
>>> a[:] = numpy.nan

#R  d[is.na(d)] <- 0
#python DataFrame.fillna(0)

import math
>>> train = [10, float('NaN'), 20, float('NaN'), 30]
>>> train = [3 if math.isnan(x) else x for x in train]

======== random =========
import random
import time

def getRandomDate(startDate, endDate ):
    print("Printing random date between", startDate, " and ", endDate)
    
    dateFormat = '%m/%d/%Y'

    startTime = time.mktime(time.strptime(startDate, dateFormat))
    endTime = time.mktime(time.strptime(endDate, dateFormat))

    randomGenerator = random.random()
    randomTime = startTime + randomGenerator * (endTime - startTime)
    randomDate = time.strftime(dateFormat, time.localtime(randomTime))
    return randomDate

print ("Random Date = ", getRandomDate("1/1/2016", "12/12/2018"))
Printing random date between 1/1/2016  and  12/12/2018
Random Date =  02/05/2017


import random

dice = [1, 2, 3, 4, 5, 6]
print("Randomly selecting same number of a dice")
for i in range(5):
    random.seed(25)
    print(random.choice(dice))


import secrets

print("Random secure Hexadecimal token is ", secrets.token_hex(64))
print("Random secure URL is ", secrets.token_urlsafe(64))
Random secure Hexadecimal token is  768b6a15dd04c9a407160b258a8f0f609c8e6911db360e2221f9a1508eeefec30f5dec1eb630dcc68ce4270a4c601c70d2a98e83a66c7f6164fe9c2233f486d6
Random secure URL is  PFBMuCd1kK8aBogzTNGXBgfb-szY2IqtE6NMKXuESu6dbq8Ag2fSiznqYZ52NcwMsKpyNAyy2R5z0bwlRHFQBg


import random

num1  = random.random()
print("First Random float is ", num1)
num2 = random.uniform(9.5, 99.5)
print("Second Random float is ", num1)

num3 = num1 * num2
print("Multiplication is ", num3)


import random
import string

def randomPassword():
    randomSource = string.ascii_letters + string.digits + string.punctuation
    password = random.sample(randomSource, 6)
    password += random.sample(string.ascii_uppercase, 2)
    password += random.choice(string.digits)
    password += random.choice(string.punctuation)
    print(type(password))
    print(password)

    #passwordList = list(password)
    passwordList = password
    random.SystemRandom().shuffle(passwordList)
    password = ''.join(passwordList)
    return password

print ("Password is ", randomPassword())
<class 'list'>
['1', '[', '&', '!', '|', '*', 'I', 'J', '8', ']']
Password is  *I]|1![8J&


import random
import string

def randomPassword():
    randomSource = string.ascii_letters + string.digits + string.punctuation
    password = random.sample(randomSource, 6)
    password += random.sample(string.ascii_uppercase, 2)
    password += random.choice(string.digits)
    password += random.choice(string.punctuation)

    passwordList = list(password)
    random.SystemRandom().shuffle(passwordList)
    password = ''.join(passwordList)
    return password

print ("Password is ", randomPassword())
Password is  ]IDO#C!I17


import random
import string

def randomString(stringLength):
    """Generate a random string of 5 charcters"""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))

print ("Random String is ", randomString(5) )
Random String is  ODmKV

import random

name = 'pynative'
char = random.choice(name)
print("random char is ", char)
random char is  v

import secrets

#Getting systemRandom class instance out of secrets module
secretsGenerator = secrets.SystemRandom()

print("Generating 6 digit random OTP")
otp = secretsGenerator.randrange(100000, 999999)

print("Secure random OTP is ", otp)
Generating 6 digit random OTP
Secure random OTP is  890495


import random

lottery_tickets_list = []
print("creating 100 random lottery tickets")
# to get 100 ticket
for i in range(100):
    # ticket number must be 10 digit (1000000000, 9999999999)
    lottery_tickets_list.append(random.randrange(1000000000, 9999999999))
# pick 2 luck tickets
winners = random.sample(lottery_tickets_list, 2)
print("Lucky 2 lottery tickets are", winners)

creating 100 random lottery tickets
Lucky 2 lottery tickets are [2224951518, 6391065163]


import random

print("Generating 3 random integer number between 100 and 999 divisible by 5")
for num in range(3):
    print(random.randrange(100, 999, 5), end=', ')
Generating 3 random integer number between 100 and 999 divisible by 5
210, 590, 915, 

======== jason =========

import json

sampleJson = """[ 
   { 
      "id":1,
      "name":"name1",
      "color":[ 
         "red",
         "green"
      ]
   },
   { 
      "id":2,
      "name":"name2",
      "color":[ 
         "pink",
         "yellow"
      ]
   }
]"""

data = []
try:
    data = json.loads(sampleJson)
except Exception as e:
    print(e)

dataList = [item.get('name') for item in data]
print(dataList)
['name1', 'name2']

import json

def validateJSON(jsonData):
    try:
        json.loads(jsonData)
    except ValueError as err:
        return False
    return True

InvalidJsonData = """{ "company":{ "employee":{ "name":"emma", "payble":{ "salary":7000 "bonus":800} } } }"""
isValid = validateJSON(InvalidJsonData)

print("Given JSON string is Valid", isValid)
Given JSON string is Valid False

#OR
echo "JSON DATA" | python -m json.tool

echo { "company":{ "employee":{ "name":"emma", "payble":{ "salary":7000 "bonus":800} } } } | python -m json.tool
#Expecting ',' delimiter: line 1 column 68 (char 67)


import json

class Vehicle:
    def __init__(self, name, engine, price):
        self.name = name
        self.engine = engine
        self.price = price

def vehicleDecoder(obj):
        return Vehicle(obj['name'], obj['engine'], obj['price'])

vehicleObj = json.loads('{ "name": "Toyota Rav4", "engine": "2.5L", "price": 32000 }',
           object_hook=vehicleDecoder)

print("Type of decoded object from JSON Data")
print(type(vehicleObj))
print("Vehicle Details")
print(vehicleObj.name, vehicleObj.engine, vehicleObj.price)




Type of decoded object from JSON Data
<class '__main__.Vehicle'>
Vehicle Details
Toyota Rav4 2.5L 32000

import json
from json import JSONEncoder

class Vehicle:
    def __init__(self, name, engine, price):
        self.name = name
        self.engine = engine
        self.price = price

class VehicleEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__

vehicle = Vehicle("Toyota Rav4", "2.5L", 32000)

print("Encode Vehicle Object into JSON")
vehicleJson = json.dumps(vehicle, indent=4, cls=VehicleEncoder)
print(vehicleJson)

Encode Vehicle Object into JSON
{
    "name": "Toyota Rav4",
    "engine": "2.5L",
    "price": 32000
}




import json

sampleJson = """{ 
   "company":{ 
      "employee":{ 
         "name":"emma",
         "payble":{ 
            "salary":7000,
            "bonus":800
         }
      }
   }
}"""

data = json.loads(sampleJson)
print(data['company']['employee']['payble']['salary'])

import json

sampleJson = {"id" : 1, "name" : "value2", "age" : 29}

print("Started writing JSON data into a file")
with open("sampleJson.json", "w") as write_file:
    json.dump(sampleJson, write_file, indent=4, sort_keys=True)
print("Done writing JSON data into a file")


import json

sampleJson = {"key1" : "value2", "key2" : "value2", "key3" : "value3"}
prettyPrintedJson  = json.dumps(sampleJson, indent=2, separators=(",", " = "))
print(prettyPrintedJson)


import json

sampleJson = """{"key1": "value1", "key2": "value2"}"""

data = json.loads(sampleJson)
print(data['key2'])


import json

data = {"key1" : "value1", "key2" : "value2"}

jsonData = json.dumps(data)
print(jsonData)



======== tuple =========
def check(sampleTuple):
    return all(i == sampleTuple[0] for i in sampleTuple)

tuple1 = (45, 45, 45, 45)
print(check(tuple1))

tuple1 = (50, 10, 60, 70, 50)
print(tuple1.count(50))

tuple1 = (('a', 23),('b', 37),('c', 11), ('d',29))
tuple1 = tuple(sorted(list(tuple1), key=lambda x: x[1]))
print(tuple1)

tuple1 = (11, 22)
tuple2 = (99, 88)
tuple1, tuple2 = tuple2, tuple1
print(tuple2)
print(tuple1)


aTuple = (10, 20, 30, 40)
a, b, c, d = aTuple

#If you want to create a single value tuple, you must indicate it by adding a comma just before the closing parentheses.
aTuple = (50, )
print(aTuple)
(50,)

======== set =========

set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

set1.intersection_update(set2)
print(set1)
{40, 50, 30}

set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

set1.symmetric_difference_update(set2)
print(set1)
{70, 10, 20, 60}

set1 = {10, 20, 30, 40, 50}
set2 = {60, 70, 80, 90, 10}

if set1.isdisjoint(set2):
  print("Two sets have no items in common")
else:
  print("Two sets have items in common")
  print(set1.intersection(set2))


set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

print(set1.symmetric_difference(set2))
{20, 70, 10, 60}


set1 = {10, 20, 30, 40, 50}
set1.difference_update({10, 20, 30})
print(set1)
{40, 50}


set1 = {10, 20, 30}
set2 = {20, 40, 50}

set1.difference_update(set2)
print(set1)
set1 = {10, 30}

set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

print(set1.union(set2))


set1 = {10, 20, 30, 40, 50}
set2 = {30, 40, 50, 60, 70}

print(set1.intersection(set2))


sampleSet = {"Yellow", "Orange", "Black"}
sampleList = ["Blue", "Green", "Red"]
sampleSet.update(sampleList)
print(sampleSet)



======== dict =========

sampleDict = {
  'Physics': 82,
  'Math': 65,
  'history': 75
}
print(min(sampleDict, key=sampleDict.get))
Math
print(min(sampleDict))
Math

#OR
print([k for k, v in sampleDict.items() if v == min(sampleDict.values())])
['Math']
print(min(sampleDict.values()))
65

sampleDict = {
  "name": "Kelly",
  "age":25,
  "salary": 8000,
  "city": "New york"
}

sampleDict['location'] = sampleDict.pop('city')
print(sampleDict)
{'name': 'Kelly', 'age': 25, 'salary': 8000, 'location': 'New york'}



sampleDict = {'a': 100, 'b': 200, 'c': 300}
print(200 in sampleDict.values())


sampleDict = {
  "name": "Kelly",
  "age":25,
  "salary": 8000,
  "city": "New york"
}
keysToRemove = ["name", "salary"]

sampleDict = {k: sampleDict[k] for k in sampleDict.keys() - keysToRemove}
print(sampleDict)
{'city': 'New york', 'age': 25}


sampleDict = { 
  "name": "Kelly",
  "age":25, 
  "salary": 8000, 
  "city": "New york" }

keys = ["name", "salary"]

newDict = {k: sampleDict[k] for k in keys}
print(newDict)
{'name': 'Kelly', 'salary': 8000}


employees = ['Kelly', 'Emma', 'John']
defaults = {"designation": 'Application Developer', "salary": 8000}

resDict = dict.fromkeys(employees, defaults)
print(resDict)

# Individual data
print(resDict["Kelly"])



sampleDict = { 
   "class":{ 
      "student":{ 
         "name":"Mike",
         "marks":{ 
            "physics":70,
            "history":80
         }
      }
   }
}
print(sampleDict['class']['student']['marks']['history'])



dict1 = {'Ten': 10, 'Twenty': 20, 'Thirty': 30}
dict2 = {'Thirty': 30, 'Fourty': 40, 'Fifty': 50}

dict3 = {**dict1, **dict2}
print(dict3)
print({**dict1})


dict1 = {'Ten': 10, 'Twenty': 20, 'Thirty': 30}
dict2 = {'Thirty': 30, 'Fourty': 40, 'Fifty': 50}

dict3 = dict1.copy()
dict3.update(dict2)
print(dict3)


keys = ['Ten', 'Twenty', 'Thirty']
values = [10, 20, 30]

sampleDict = dict(zip(keys, values))
print(sampleDict)


======== list =========

list1 = [5, 10, 15, 20, 25, 50, 20]

index = list1.index(20)
list1[index] = 200
print(list1)


list1 = ["a", "b", ["c", ["d", "e", ["f", "g"], "k"], "l"], "m", "n"]
subList = ["h", "i", "j"]

list1[2][1][2].extend(subList)
print(list1)


list1 = [10, 20, [300, 400, [5000, 6000], 500], 30, 40]

def add_new(n, alist):
    for a in alist:
        #print("current a=", a)
        if isinstance(a, list):
            #print("======")
            add_new(n, a)
        if a == 6000:
            #print("found a", a)
            alist.append(n)
            #print(alist)
            #return alist
    return alist


[10, 20, [300, 400, [5000, 6000, 7000], 500], 30, 40]


print(add_new(7000, list1))

list1 = [10, 20, [300, 400, [5000, 6000], 500], 30, 40]

def add_new2(n, alist):
    while(alist):
        for a in alist:
            if isinstance(a, list):
                alist = a
            if a == 6000:
                alist.append(n)
                #print(alist)
                return alist

print(add_new2(7000, list1))

[5000, 6000, 7000]




list1 = [10, 20, [300, 400, [5000, 6000], 500], 30, 40]
list1[2][2].append(7000)
print(list1)



# function that filters vowels 
def fun(variable): 
    letters = ['a', 'e', 'i', 'o', 'u'] 
    if (variable in letters): 
        return True
    else: 
        return False
  
  
# sequence 
sequence = ['g', 'e', 'e', 'j', 'k', 's', 'p', 'r'] 
  
# using filter function 
filtered = filter(fun, sequence) 
  
print('The filtered letters are:') 
for s in filtered: 
    print(s) 



# a list contains both even and odd numbers.  
seq = [0, 1, 2, 3, 5, 8, 13] 
  
# result contains odd numbers of the list 
result = filter(lambda x: x % 2 != 0, seq) 
print(list(result)) 
  
# result contains even numbers of the list 
result = filter(lambda x: x % 2 == 0, seq) 
print(list(result)) 



list1 = ["Mike", "", "Emma", "Kelly", "", "Brad"]
resList = list(filter(None, list1))
print(resList)


for x, y in zip(list1, list2[::-1]):
    print(x, y)


list1 = ["Hello ", "take "]
list2 = ["Dear", "Sir"]

resList = [x+y for x in list1 for y in list2]
print(resList)

['Hello Dear', 'Hello Sir', 'take Dear', 'take Sir']



list1 = ["M", "na", "i", "Ke"] 
list2 = ["y", "me", "s", "lly"]
list3 = [i + j for i, j in zip(list1, list2)]
print(list3)

['My', 'name', 'is', 'Kelly']

======== data structure =========

sampleList = [87, 52, 44, 53, 54, 87, 52, 53]

print("Original list", sampleList)

sampleList = list(set(sampleList))
print("unique list", sampleList)

tuple = tuple(sampleList)
print("tuple ", tuple)

print("Minimum number is: ", min(tuple))
print("Maximum number is: ", max(tuple))



print(list(set([s for s in speed.values()])))

#OR


speed  ={'jan':47, 'feb':52, 'march':47, 'April':44, 'May':52, 'June':53,
          'july':54, 'Aug':44, 'Sept':54} 

print("Dictionary's values - ", speed.values())

speedList = list()
for item in speed.values():
  if item not in speedList:
    speedList.append(item)
print("unique list", speedList)




rollNumber  = [47, 64, 69, 37, 76, 83, 95, 97]
sampleDict  ={'Jhon':47, 'Emma':69, 'Kelly':76, 'Jason':97} 

print("List -", rollNumber)
print("Dictionary - ", sampleDict)

rollNumber[:] = [item for item in rollNumber if item in sampleDict.values()]
print("after removing unwanted elemnts from list ", rollNumber)


firstSet  = {57, 83, 29}
secondSet = {57, 83, 29, 67, 73, 43, 48}

print("First Set ", firstSet)
print("Second Set ", secondSet)

print("First set is subset of second set -", firstSet.issubset(secondSet))
print("Second set is subset of First set - ", secondSet.issubset(firstSet))

print("First set is Super set of second set - ", firstSet.issuperset(secondSet))
print("Second set is Super set of First set - ", secondSet.issuperset(firstSet))

if(firstSet.issubset(secondSet)):
  firstSet.clear()
  
if(secondSet.issubset(firstSet)):
  secondSet.clear()

print("First Set ", firstSet)
print("Second Set ", secondSet)




firstSet  = {23, 42, 65, 57, 78, 83, 29}
secondSet = {57, 83, 29, 67, 73, 43, 48}

print("First Set ", firstSet)
print("Second Set ", secondSet)

intersection = firstSet.intersection(secondSet)
print("Intersection is ", intersection)
for item in intersection:
  firstSet.remove(item)

print("First Set after removing common element ", firstSet)



firstList = [2, 3, 4, 5, 6, 7, 8]
print("First List ", firstList)

secondList = [4, 9, 16, 25, 36, 49, 64]
print("Second List ", secondList)

result = zip(firstList, secondList)
resultSet = set(result)
print(resultSet)
print(list(zip(*resultSet)))

First List  [2, 3, 4, 5, 6, 7, 8]
Second List  [4, 9, 16, 25, 36, 49, 64]
{(7, 49), (2, 4), (4, 16), (8, 64), (6, 36), (3, 9), (5, 25)}
[(7, 2, 4, 8, 6, 3, 5), (49, 4, 16, 64, 36, 9, 25)]


sampleList = [11, 45, 8, 11, 23, 45, 23, 45, 89]
dic = dict()
for n in sampleList:
    dic[n] = dic.get(n,0)+1
print(dic)


sampleList = [34, 54, 67, 89, 11, 43, 94]

print("Original list ", sampleList)
element = sampleList.pop(4)
print("List After removing element at index 4 ", sampleList)

sampleList.insert(2, element)
print("List after Adding element at index 2 ", sampleList)

sampleList.append(element)
print("List after Adding element at last ", sampleList)



listOne = [3, 6, 9, 12, 15, 18, 21]
listTwo = [4, 8, 12, 16, 20, 24, 28]
listThree = list()

oddElements = listOne[1::2]
print("Element at odd-index positions from list one")
print(oddElements)

EvenElement = listTwo[0::2]
print("Element at even-index positions from list two")
print(EvenElement)

print("Printing Final third list")
listThree.extend(oddElements)
listThree.extend(EvenElement)
print(listThree)



======== str =========


inputStr = "pynativepynvepynative"
countDict = dict()
for char in inputStr:
  count = inputStr.count(char)
  countDict[char]=count
print(countDict)

inputStr = "pynativepynvepynative"
import collections

print(collections.Counter(inputStr))

import re

inputStr = "English = 78 Science = 83 Math = 68 History = 65"
markList = [int(num) for num in re.findall(r'\b\d+\b', inputStr)]
totalMarks = 0
for mark in markList:
  totalMarks+=mark

percentage = totalMarks/len(markList)  
print("Total Marks is:", totalMarks, "Percentage is ", percentage)




strs = "English = 78 Science = 83 Math = 68 History = 65"

tl =  [int(s) for s in strs.split() if s.isnumeric()]
print(tl)
import statistics 
print(sum(t for t in tl), statistics.mean(tl))
print(sum(t for t in tl), sum(t for t in tl)/len(tl))

#OR



inputString = "Welcome to USA. usa awesome, isn't it?"
substring = "USA"
tempString = inputString.lower()
count = tempString.count(substring.lower())
print("The USA count is:", count)



s1 = "yn"
s2 = "Pynative"

print(all((s in s2) for s in s1))


char.islower() or char.isupper()
char.isnumeric() char.isalpha()


input_String = "PyNaTive"
ini_list = [s for s in input_String] 
print(ini_list)
ls = sorted(ini_list, key = lambda s: s.casefold())
print(ls)
print(''.join(ls))

ls2 = sorted(ini_list, key = lambda s: s==s.upper())
print(ls2)
print(''.join(ls2))

#OR

ls2 = sorted(ini_list, key = lambda s: s.isupper())
print(ls2)
print(''.join(ls2))

======== def/function =========

def displayStudent(name, age):
    print(name, age)

displayStudent("Emma", 26)

showStudent = displayStudent
showStudent("Emma", 26)

def calculateSum(num):
    if num:
        return num + calculateSum(num-1)
    else:
        return 0

res = calculateSum(10)
print(res)


def func1(*args):
    for i in args:
        print(i)

func1(20, 40, 60)
func1(80, 100)

======== loop =========

for i in range(5):
    print(i)
else:
    print("Done!")


for i in range(5,0,-1):
    for j in range(i,0,-1):
        print(j, end=' ')
    print()


======== I/O =========

#read line #4:
with open("test.txt", "r") as fp:
    lines = fp.readlines()
    print(lines[3])


#check if the file is empty
import os
print(os.stat("test.txt").st_size == 0)

quantity = 3
totalMoney = 1000
price = 450
statement1 = "I have {1} dollars so I can buy {0} football for {2:.2f} dollars."
print(statement1.format(quantity, totalMoney, price))


str1, str2, str3 = input("Enter three string").split()
print(str1, str2, str3)

#read/write file line by line
import shutil
with open('test') as old, open('newtest', 'w') as new:
    for line in old:
        if line.rsplit('|', 1)[-1].strip() == 'number3':
            new.write('this is replacement|number7\n')
        else:
            new.write(line)
shutil.move('newtest', 'test')


with open(...) as f:
    for line in f:


#read in all, write selective lines
count = 0
with open("test.txt", "r") as fp:
    lines = fp.readlines()
    count = len(lines)
with open("newFile.txt", "w") as fp:
    for line in lines:
        if (count != 3):
            fp.write(line)
        count-=1


floatNumbers = []
n = int(input("Enter the list size : "))

print("\n")
for i in range(0, n):
    print("Enter number at location", i, ":")
    item = float(input())
    floatNumbers.append(item)
    
print("User List is ", floatNumbers)


Python String Interpolation with the Percent (%) Operator:

https://stackabuse.com/python-string-interpolation-with-the-percent-operator/

print('%.2f' % 458.541315)

print("The decimal value of", dec, "is:")
print(bin(dec), "in binary.")
print(oct(dec), "in octal.")
print(hex(dec), "in hexadecimal.")

print('%o,' % (8))
print('%x,' % (16))
print(bin(8)[2:], "in binary.")

print('My', 'Name', 'Is', 'James', sep='**')

num1 = int(input("Enter first number "))
num2 = int(input("Enter second number "))

print("\n")
res = num1 + num2
print("Sum is", res)

======== basic =========

def combine_list(A, B):
    res = []
    if A != None:
        res = [a for a in A if a%2]
    if B != None:
        res = res + [b for b in B if not b%2]

    return res

A, B = [10, 20, 23, 11, 17], [13, 43, 24, 36, 12]
print(combine_list(A,B))

def reverseNum(numb):
    org_numb, rev_numb = numb, 0
    while(numb):
        numb, rem = divmod(numb, 10)
        rev_numb = rev_numb*10 + rem
    print("rev_numb", rev_numb)
    
    return org_numb == rev_numb

print(reverseNum(121))

def reverseNum2(numb):
    print("rev_numb", int(str(numb)[::-1]))
    return numb == int(str(numb)[::-1])

reverseNum2(128)


Python tips - How to easily convert a list to a string for display:
https://www.decalage.info/en/python/print_list

from itertools import repeat

def rep_print(N):
    alist =  [list(repeat(i,i)) for i in range(1, N+1)]
    return '\n'.join( (lambda s: str(s).strip('[]')) (s) for s in alist)
    
    
print(rep_print(3))

ls = [[1], [2, 2], [3, 3, 3]]
print('\n'.join(map(str, ls)))


def rep_print2(N):
    return  [str(list(repeat(i,i))).strip('[]') for i in range(1, N+1)]

print('\n'.join(rep_print2(4)))


========= mysql ========

mysql> SHOW VARIABLES LIKE '%version%';
+--------------------------+-------------------------------+
| Variable_name            | Value                         |
+--------------------------+-------------------------------+
| immediate_server_version | 999999                        |
| innodb_version           | 8.0.20                        |
| original_server_version  | 999999                        |
| protocol_version         | 10                            |
| slave_type_conversions   |                               |
| tls_version              | TLSv1,TLSv1.1,TLSv1.2,TLSv1.3 |
| version                  | 8.0.20                        |
| version_comment          | MySQL Community Server - GPL  |
| version_compile_machine  | x86_64                        |
| version_compile_os       | macos10.15                    |
| version_compile_zlib     | 1.2.11                        |
+--------------------------+-------------------------------+
11 rows in set (0.01 sec)


mysql> select database()
    -> ;
+------------+
| database() |
+------------+
| employees  |
+------------+
1 row in set (0.01 sec)

mysql> show databases;
+--------------------+
| Database           |
+--------------------+
| employees          |
| information_schema |
| mysql              |
| performance_schema |
| sys                |
+--------------------+
5 rows in set (0.00 sec)

mysql> with recursive t(val) as (select 1 union all select val + 3 from t limit 5) select val from t;
+------+
| val  |
+------+
|    1 |
|    4 |
|    7 |
|   10 |
|   13 |
+------+
5 rows in set (0.00 sec)

mysql> with recursive t(val) as (select 1 union all select val + 1 from t limit 5) select val from t;
+------+
| val  |
+------+
|    1 |
|    2 |
|    3 |
|    4 |
|    5 |
+------+
5 rows in set (0.01 sec)

mysql> WITH
    ->     ASSIGN(ID, ASSIGN_AMT) AS (
    ->                   SELECT 1, 25150 FROM DUAL 
    ->         UNION ALL SELECT 2, 19800 FROM DUAL
    ->         UNION ALL SELECT 3, 27511 FROM DUAL
    ->     ),
    ->     VALS (ID, WORK_AMT) AS (
    ->                   SELECT 1 , 7120  FROM DUAL 
    ->         UNION ALL SELECT 2 , 8150  FROM DUAL
    ->         UNION ALL SELECT 3 , 8255  FROM DUAL
    ->         UNION ALL SELECT 4 , 9051  FROM DUAL
    ->         UNION ALL SELECT 5 , 1220  FROM DUAL
    ->         UNION ALL SELECT 6 , 12515 FROM DUAL
    ->         UNION ALL SELECT 7 , 13555 FROM DUAL
    ->         UNION ALL SELECT 8 , 5221  FROM DUAL
    ->         UNION ALL SELECT 9 , 812   FROM DUAL
    ->         UNION ALL SELECT 10, 6562  FROM DUAL
    ->     ) select vals.* from vals;
+----+----------+
| ID | WORK_AMT |
+----+----------+
|  1 |     7120 |
|  2 |     8150 |
|  3 |     8255 |
|  4 |     9051 |
|  5 |     1220 |
|  6 |    12515 |
|  7 |    13555 |
|  8 |     5221 |
|  9 |      812 |
| 10 |     6562 |
+----+----------+
10 rows in set (0.00 sec)


select vals.*, sum(work_amt) over (order by id) as subset_sum from vals;
+----+----------+------------+
| ID | WORK_AMT | subset_sum |
+----+----------+------------+
|  1 |     7120 |       7120 |
|  2 |     8150 |      15270 |
|  3 |     8255 |      23525 |
|  4 |     9051 |      32576 |
|  5 |     1220 |      33796 |
|  6 |    12515 |      46311 |
|  7 |    13555 |      59866 |
|  8 |     5221 |      65087 |
|  9 |      812 |      65899 |
| 10 |     6562 |      72461 |
+----+----------+------------+
10 rows in set (0.00 sec)


sums(id, work_amt, subset_sum) as (select vals.*, sum(work_amt) over (order by
id) from vals) select assign.id, assign.assign_amt, subset_sum from assign join sums on assign.id = sums.id;
+----+------------+------------+
| id | assign_amt | subset_sum |
+----+------------+------------+
|  1 |      25150 |       7120 |
|  2 |      19800 |      15270 |
|  3 |      27511 |      23525 |
+----+------------+------------+
3 rows in set (0.03 sec)


, sums(id, work_amt, subset_sum) as (select vals.*, sum(work_amt) over (order by id) from vals) select assign.id, assign.assign_amt, subset_sum from assign join sums on abs(assign_amt - subset_sum) <= ALL (select abs(assign_amt - subset_sum)  from sums);
+----+------------+------------+
| id | assign_amt | subset_sum |
+----+------------+------------+
|  1 |      25150 |      23525 |
|  2 |      19800 |      23525 |
|  3 |      27511 |      23525 |
+----+------------+------------+
3 rows in set (0.07 sec)


mysql> select * from (select 1 as a from dual union all select 2 as a from dual union all select 3 as a from dual) t;
+---+
| a |
+---+
| 1 |
| 2 |
| 3 |
+---+
3 rows in set (0.00 sec)

mysql> select substring('abcde',2,3) as a from dual;
+------+
| a    |
+------+
| bcd  |
+------+
1 row in set (0.00 sec)

mysql> with recursive t(v) as (select 1 union all select v + 1 from t where v < 3) select v from t;
+------+
| v    |
+------+
|    1 |
|    2 |
|    3 |
+------+
3 rows in set (0.01 sec)

mysql> with recursive t(v) as (select 1 union all select v + 1 from t) select v from t limit 1, 4;
ERROR 3636 (HY000): Recursive query aborted after 1001 iterations. Try increasing @@cte_max_recursion_depth to a larger value.
mysql> with recursive t(v) as (select 1 union all select v + 1 from t limit 5) select v from t;
+------+
| v    |
+------+
|    1 |
|    2 |
|    3 |
|    4 |
|    5 |
+------+
5 rows in set (0.00 sec)

mysql> with recursive t(v) as (select 1 union all select v + 1 from t limit 2, 5) select v from t;
+------+
| v    |
+------+
|    3 |
|    4 |
|    5 |
|    6 |
|    7 |
+------+
5 rows in set (0.00 sec)


mysql> select * from employees limit 10;
+--------+------------+------------+-----------+--------+------------+
| emp_no | birth_date | first_name | last_name | gender | hire_date  |
+--------+------------+------------+-----------+--------+------------+
|  10001 | 1953-09-02 | Georgi     | Facello   | M      | 1986-06-26 |
|  10002 | 1964-06-02 | Bezalel    | Simmel    | F      | 1985-11-21 |
|  10003 | 1959-12-03 | Parto      | Bamford   | M      | 1986-08-28 |
|  10004 | 1954-05-01 | Chirstian  | Koblick   | M      | 1986-12-01 |
|  10005 | 1955-01-21 | Kyoichi    | Maliniak  | M      | 1989-09-12 |
|  10006 | 1953-04-20 | Anneke     | Preusig   | F      | 1989-06-02 |
|  10007 | 1957-05-23 | Tzvetan    | Zielinski | F      | 1989-02-10 |
|  10008 | 1958-02-19 | Saniya     | Kalloufi  | M      | 1994-09-15 |
|  10009 | 1952-04-19 | Sumant     | Peac      | F      | 1985-02-18 |
|  10010 | 1963-06-01 | Duangkaew  | Piveteau  | F      | 1989-08-24 |
+--------+------------+------------+-----------+--------+------------+
10 rows in set (0.01 sec)


mysql> with d1 as (select distinct hire_date from employees), d2 as (select hire_date, row_number() over (order by hire_date) as grp from d1) select hire_date, grp from d2 limit 10;
+------------+-----+
| hire_date  | grp |
+------------+-----+
| 1985-01-01 |   1 |
| 1985-01-14 |   2 |
| 1985-02-01 |   3 |
| 1985-02-02 |   4 |
| 1985-02-03 |   5 |
| 1985-02-04 |   6 |
| 1985-02-05 |   7 |
| 1985-02-06 |   8 |
| 1985-02-07 |   9 |
| 1985-02-08 |  10 |
+------------+-----+
10 rows in set (0.11 sec)



mysql> with d1 as (select distinct hire_date from employees) select hire_date, DATE_SUB(hire_date, INTERVAL (row_number() over (order by hire_date)) DAY) as grp from d1 limit 10;
+------------+------------+
| hire_date  | grp        |
+------------+------------+
| 1985-01-01 | 1984-12-31 |
| 1985-01-14 | 1985-01-12 |
| 1985-02-01 | 1985-01-29 |
| 1985-02-02 | 1985-01-29 |
| 1985-02-03 | 1985-01-29 |
| 1985-02-04 | 1985-01-29 |
| 1985-02-05 | 1985-01-29 |
| 1985-02-06 | 1985-01-29 |
| 1985-02-07 | 1985-01-29 |
| 1985-02-08 | 1985-01-29 |
+------------+------------+
10 rows in set (0.10 sec)


mysql> with d1 as (select distinct hire_date from employees), d2 as ( select hire_date, DATE_SUB(hire_date, INTERVAL (row_number() over (order by hire_date)) DAY) as grp from d1) select min(hire_date), max(hire_date), max(hire_date) - min(hire_date) + 1 as length from d2 group by grp order by length DESC limit 10;
+----------------+----------------+--------+
| min(hire_date) | max(hire_date) | length |
+----------------+----------------+--------+
| 1985-02-01     | 1999-07-01     | 140501 |
| 1999-12-30     | 2000-01-04     |   8875 |
| 1999-07-12     | 1999-09-03     |    192 |
| 1999-09-05     | 1999-10-06     |    102 |
| 1999-11-29     | 1999-12-04     |     76 |
| 1999-11-12     | 1999-11-23     |     12 |
| 1999-10-08     | 1999-10-19     |     12 |
| 1999-07-03     | 1999-07-10     |      8 |
| 1999-10-26     | 1999-10-30     |      5 |
| 1999-11-25     | 1999-11-27     |      3 |
+----------------+----------------+--------+
10 rows in set (0.11 sec)




mysql> INSERT INTO h(val) VALUES(1),(0),(0),(1),(1),(0),(0),(0),(1),(1),(0);
Query OK, 11 rows affected (0.00 sec)
Records: 11  Duplicates: 0  Warnings: 0

mysql> select * from h
    -> ;
+------+
| val  |
+------+
|    1 |
|    0 |
|    0 |
|    1 |
|    1 |
|    0 |
|    0 |
|    0 |
|    1 |
|    1 |
|    0 |
+------+
11 rows in set (0.00 sec)


mysql> with d1 as (select val, row_number() over () as rn from h), d2 as (select val, case when coalesce(lag(val) over (),0) != val then rn end as lo, case when coalesce(lead(val) over (),0) !=
val then rn end as hi from d1) select * from d2;
+------+------+------+
| val  | lo   | hi   |
+------+------+------+
|    1 |    1 |    1 |
|    0 |    2 | NULL |
|    0 | NULL |    3 |
|    1 |    4 | NULL |
|    1 | NULL |    5 |
|    0 |    6 | NULL |
|    0 | NULL | NULL |
|    0 | NULL |    8 |
|    1 |    9 | NULL |
|    1 | NULL |   10 |
|    0 |   11 | NULL |
+------+------+------+
11 rows in set (0.00 sec)


mysql> set @cnt := 0; set @lag := 1; select val, @cnt := if(@lag = val, @cnt + 1, if(@lag := val, 1, 1)) as cnt, @lag as prev, lag(@cnt) over() from h where @lag =val;
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

+------+------+------+------------------+
| val  | cnt  | prev | lag(@cnt) over() |
+------+------+------+------------------+
|    1 |    1 |    1 |             NULL |
|    1 |    2 |    1 |                0 |
|    1 |    3 |    1 |                1 |
|    1 |    4 |    1 |                2 |
|    1 |    5 |    1 |                3 |
+------+------+------+------------------+
5 rows in set, 2 warnings (0.00 sec)




mysql> set @cnt := 0; set @lag := 1; select val, @cnt := if(@lag = val, @cnt + 1, 1) as cnt, @lag := if(@lag = val, @lag, val) as prev from h;
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

+------+------+------+
| val  | cnt  | prev |
+------+------+------+
|    1 |    1 |    1 |
|    0 |    1 |    0 |
|    0 |    2 |    0 |
|    1 |    1 |    1 |
|    1 |    2 |    1 |
|    0 |    1 |    0 |
|    0 |    2 |    0 |
|    0 |    3 |    0 |
|    1 |    1 |    1 |
|    1 |    2 |    1 |
|    0 |    1 |    0 |
+------+------+------+
11 rows in set, 2 warnings (0.01 sec)


mysql> set @cnt := 0; set @lag := 1; set @out := 100; select val, @cnt := if(@lag = val, @cnt + 1, 1) as cnt, @lag := if(@lag = val, @lag, val) as
 prev from h;
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.01 sec)

Query OK, 0 rows affected (0.00 sec)

+------+------+------+
| val  | cnt  | prev |
+------+------+------+
|    1 |    1 |    1 |
|    0 |    1 |    0 |
|    0 |    2 |    0 |
|    1 |    1 |    1 |
|    1 |    2 |    1 |
|    0 |    1 |    0 |
|    0 |    2 |    0 |
|    0 |    3 |    0 |
|    1 |    1 |    1 |
|    1 |    2 |    1 |
|    0 |    1 |    0 |
+------+------+------+
11 rows in set, 2 warnings (0.00 sec)







mysql> set @cnt := 0; set @lag := 1; set @ot := null; with d1 as (select val, @ot := if(@lag = val, null, @cnt) as ot1, @cnt := if(@lag = val, @cn
t + 1, 1) as cnt, @lag := if(@lag = val, @lag, val) as prev from h) select * from d1;
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

+------+------+------+------+
| val  | ot1  | cnt  | prev |
+------+------+------+------+
|    1 | NULL |    1 |    1 |
|    0 |    1 |    1 |    0 |
|    0 | NULL |    2 |    0 |
|    1 |    2 |    1 |    1 |
|    1 | NULL |    2 |    1 |
|    0 |    2 |    1 |    0 |
|    0 | NULL |    2 |    0 |
|    0 | NULL |    3 |    0 |
|    1 |    3 |    1 |    1 |
|    1 | NULL |    2 |    1 |
|    0 |    2 |    1 |    0 |
+------+------+------+------+
11 rows in set, 3 warnings (0.00 sec)



mysql> set @cnt := 0; set @lag := 1; set @ot := null; with d1 as (select val, @ot := if(@lag = val, null, @cnt) as ot1, @cnt := if(@lag = val, @cnt + 1, 1) as cnt, @lag := if(@lag = val, @lag, val) as prev from h) select ot1 from d1 where ot1 is not NULL;
Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

Query OK, 0 rows affected (0.00 sec)

+------+
| ot1  |
+------+
|    1 |
|    2 |
|    2 |
|    3 |
|    2 |
+------+
5 rows in set, 3 warnings (0.00 sec)




