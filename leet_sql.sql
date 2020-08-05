/* 
#####################################################
# Exchange Seats
# input:
+---------+---------+
|    id   | student |
+---------+---------+
|    1    | Abbot   |
|    2    | Doris   |
|    3    | Emerson |
|    4    | Green   |
|    5    | Jeames  |
+---------+---------+
# output:
+---------+---------+
|    id   | student |
+---------+---------+
|    1    | Doris   |
|    2    | Abbot   |
|    3    | Green   |
|    4    | Emerson |
|    5    | Jeames  |
+---------+---------+
*/

select 
    case
        when((select max(id) from seat)%2 = 1) and id = (select max(id) from seat) then id
        when id%2 = 1 then id + 1
        else id - 1
    end as id, student
from seat order by id;

/* 
#####################################################
# Swap salary
*/

update salary
    set sex = case sex
        when 'm' then 'f'
        else 'm'
    end;


/*
610. Triangle Judgement
*/

SELECT x,y,z,
    CASE
        WHEN x>0 and y>0 and z>0 and x + y > z AND x + z > y AND y + z > x THEN 'Yes'
        ELSE 'No'
    END AS 'is_triangle'
FROM triangle;

/* OR 
SELECT x, y, z, IIF(x+y>z AND x+z>y AND y+z>x, 'Yes', 'No') AS is_triangle FROM triangle; */

/* 176. Second Highest Salary  */

select max(a.Salary) as second_max from Employee a, (select max(Salary) from Employee) as first_max where a.Salary < first_max

select Id, max(Salary) as Salary from Employee where Salary<(select max(Salary) from Employee);

/* Nth salary */
Select salary from employee A where n-1 = (Select count(1) from employee B where B.salary>A.salary)

/* select salary from employee order by salary desc limit n-1, 1; */ /* Db2: fetch first n rows only */

select salary from (SELECT *, DENSE_RANK() OVER (ORDER BY Salary Desc) AS Rnk FROM Employees) as a where Rnk=n;



DENSE_RANK() RANK()




Recursion
A recursive routine is one that invokes itself. Recursive routines often offer elegant solutions to complex programming problems, but they also tend to consume large amounts of memory. They are also likely to be less efficient and less scalable than implementations based on iterative execution.

Most recursive algorithms can be reformulated using nonrecursive techniques involving iteration. Where possible, we should give preference to the more efficient iterative algorithm.

For example, in Example 22-18, the stored procedure uses recursion to calculate the Nth element of the Fibonacci sequence, in which each element in the sequence is the sum of the previous two numbers.

Example 22-18. Recursive implementation of the Fibonacci algorithm
CREATE PROCEDURE rec_fib(n INT,OUT out_fib INT)
BEGIN
  DECLARE n_1 INT;
  DECLARE n_2 INT;

  IF (n=0) THEN
    SET out_fib=0;
  ELSEIF (n=1) then
    SET out_fib=1;
  ELSE
    CALL rec_fib(n-1,n_1);
    CALL rec_fib(n-2,n_2);
    SET out_fib=(n_1 + n_2);
  END IF;
END
Example 22-19 shows a nonrecursive implementation that returns the Nth element of the Fibonacci sequence.

Example 22-19. Nonrecursive implementation of the Fibonacci sequence
CREATE PROCEDURE nonrec_fib(n INT,OUT out_fib INT)
BEGIN
  DECLARE m INT default 0;
  DECLARE k INT DEFAULT 1;
  DECLARE i INT;
  DECLARE tmp INT;

  SET m=0;
  SET k=1;
  SET i=1;

  WHILE (i<=n) DO
    SET tmp=m+k;
    SET m=k;
    SET k=tmp;
    SET i=i+1;
  END WHILE;
  SET out_fib=m;
 END

import mysql.connector
import datetime
from dateutil.relativedelta import relativedelta

def get_connection():
    connection = mysql.connector.connect(host='localhost',
                                         database='python_db',
                                         user='pynative',
                                         password='pynative@#29')
    return connection

def close_connection(connection):
    if connection:
        connection.close()

def update_doctor_experience(doctor_id):
    # Update Doctor Experience in Years
    try:
        # Get joining date
        connection = get_connection()
        cursor = connection.cursor()
        select_query = """select Joining_Date from Doctor where Doctor_Id = %s"""
        cursor.execute(select_query, (doctor_id,))
        joining_date = cursor.fetchone()

        # calculate Experience in years
        joining_date_1 = datetime.datetime.strptime(''.join(map(str, joining_date)), '%Y-%m-%d')
        today_date = datetime.datetime.now()
        experience = relativedelta(today_date, joining_date_1).years

        # Update doctor's Experience now
        connection = get_connection()
        cursor = connection.cursor()
        sql_select_query = """update Doctor set Experience = %s where Doctor_Id =%s"""
        cursor.execute(sql_select_query, (experience, doctor_id))
        connection.commit()
        print("Doctor Id:", doctor_id, " Experience updated to ", experience, " years")
        close_connection(connection)

    except (Exception, mysql.connector.Error) as error:
        print("Error while getting doctor's data", error)

print("Question 5: Calculate and Update experience of all doctors  \n")
update_doctor_experience(101)


mysql> CREATE TRIGGER ins_sum BEFORE INSERT ON account
       FOR EACH ROW SET @sum = @sum + NEW.amount;


SELECT OrderID, Quantity,
CASE
    WHEN Quantity > 30 THEN "The quantity is greater than 30"
    WHEN Quantity = 30 THEN "The quantity is 30"
    ELSE "The quantity is under 30"
END
FROM OrderDetails;

SELECT CustomerName, City, Country
FROM Customers
ORDER BY
(CASE
    WHEN City IS NULL THEN Country
    ELSE City
END);






with cte (promotion_id,first_day,last_day)
as (
select promotion_id, first_day= min(transaction_date)
last_day = max(transaction_date)
from sales
where promotion_id is not null
group by promotion_id
)
Select count(
case when transaction_date = cte.first_day or transaction_date = cte.last_day then 1 else 0 end )/count(*) *100
from sales s inner join cte on cte.promotion_id = s.promotion_id


 create table customer(cusID int, cusName VARCHAR2(10));

insert into customer values(1, 'Ali');
insert into customer values(2, 'Moh');
insert into customer values(3, 'Abdo');

create table item(cusID int, itemName VARCHAR2(10));

insert into item values(1, 'A');
insert into item values(1, 'B');
insert into item values(2, 'A');
insert into item values(2, 'B');
insert into item values(3, 'A');

select count(cusName) AS CountCus_having2
from customer
where cusID in
(
select cusID
  from item
  where itemName in ('A','B')
  group by cusID
  having count(distinct itemName) = 2
 );
Anonymous on Apr 9, 2019



SELECT
  DISTINCT i.id AS id,
  i.userid AS userid,
  i.itemname AS itemname,
  COALESCE(LEAD(i.id)        OVER (ORDER BY i.created DESC)
          ,FIRST_VALUE(i.id) OVER (ORDER BY i.created DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)) AS nextitemid,
  COALESCE(LAG(i.id)         OVER (ORDER BY i.created DESC)
          ,LAST_VALUE(i.id)  OVER (ORDER BY i.created DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)) AS previtemid,
  COALESCE(LEAD(i.id)        OVER (PARTITION BY i.userid ORDER BY i.created DESC)
          ,FIRST_VALUE(i.id) OVER (PARTITION BY i.userid ORDER BY i.created DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)) AS nextuseritemid,
  COALESCE(LAG(i.id)         OVER (PARTITION BY i.userid ORDER BY i.created DESC)
          ,LAST_VALUE(i.id)  OVER (PARTITION BY i.userid ORDER BY i.created DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)) AS prevuseritemid,
  i.created AS created
FROM items i
  LEFT JOIN users u
  ON i.userid = u.id
ORDER BY i.created DESC;



select t1.value - t2.value from table t1, table t2 
where t1.primaryKey = t2.primaryKey - 1

