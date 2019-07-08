from lexer import Lexer
from parse import Parser
from ast import Query
import mysql.connector

short_query = """
q() = WINNER(cand=w),deserts(id=w,contain_chocolate=False);
"""

disconnected_query = """ 
q() = deserts(id=w,contain_chocolate=False),bakeries(id=b,location="midtown"),sell(bakery=b,desert=w2),WINNER(cand=w),WINNER(cand=w2);
"""

connected_query_false = """
q() = deserts(id=w,contain_chocolate=False),deserts(id=w2,contain_chocolate=True),bakeries(id=b,location="uptown"),sell(bakery=b,desert=w2),sell(bakery=b,desert=w),WINNER(cand=w),WINNER(cand=w2);
"""
connected_query_true = """
q() = deserts(id=w,contain_chocolate=False),deserts(id=w2,contain_chocolate=True),bakeries(id=b,location="midtown"),sell(bakery=b,desert=w2),sell(bakery=b,desert=w),WINNER(cand=w),WINNER(cand=w2);
"""


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="Levrac54&@sql",
  database="dbcomsoc"
)


##
def text2query(text,m,db,ballots="ballots"):
    lexer = Lexer().get_lexer()
    tokens = lexer.lex(text)
    pg = Parser()
    pg.parse()
    parser = pg.get_parser()
    q = Query(parser.parse(tokens),db,m,ballots=ballots)
    return q
    
    
##
import time

t1 = time.time()
q = text2query(connected_query_false,8,mydb,ballots="fake_ballots")
t2 = time.time()
print("Prepross 1 :",t2-t1)
print(q.name)
print(q.cq_results)
print(q.m)

t1 = time.time()
print("N(q,plurality)",q.necessity())
t2 = time.time()
print(t2-t1)
t1 = time.time()
print("P(q,plurality)",q.possibility())
t2 = time.time()
print(t2-t1)
t1 = time.time()
print("N(q,veto)",q.necessity(rule="veto"))
t2 = time.time()
print(t2-t1)
t1 = time.time()
print("P(q,veto)",q.possibility(rule="veto"))
t2 = time.time()
print(t2-t1)
t1 = time.time()
print("N(q,k-approval)",q.necessity(rule="k_approval",k=3))
t2 = time.time()
print(t2-t1)
t1 = time.time()
print("P(q,k-approval)",q.possibility(rule="k_approval",k=3))
t2 = time.time()
print(t2-t1)
t1 = time.time()
print("N(q,borda)",q.necessity(rule="borda"))
t2 = time.time()
print(t2-t1)
t1 = time.time()
print("P(q,borda)",q.possibility(rule="borda"))
t2 = time.time()
print(t2-t1)
t1 = time.time()
##

q = text2query(disconnected_query,8,mydb)
print(q.name)
print(q.cq_results)
print(q.m)
print("N(q,plurality)",q.necessity())
print("P(q,plurality)",q.possibility())
print("N(q,veto)",q.necessity(rule="veto"))
print("P(q,veto)",q.possibility(rule="veto"))
print("N(q,k-approval)",q.necessity(rule="k_approval",k=3))
print("P(q,k-approval)",q.possibility(rule="k_approval",k=3))
print("N(q,borda)",q.necessity(rule="borda"))
print("P(q,borda)",q.possibility(rule="borda"))

q = text2query(connected_query_true,8,mydb)
print(q.name)
print(q.cq_results)
print(q.m)
print("P(q,plurality)",q.possibility())
print("N(q,plurality)",q.necessity())
print("P(q,veto)",q.possibility(rule="veto"))
print("N(q,veto)",q.necessity(rule="veto"))
print("P(q,k-approval)",q.possibility(rule="k_approval",k=3))
print("N(q,k-approval)",q.necessity(rule="k_approval",k=3))
print("P(q,borda)",q.possibility(rule="borda"))
print("N(q,borda)",q.necessity(rule="borda"))