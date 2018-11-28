import MySQLdb
db = MySQLdb.connect(host="dqn-db-instance.coib1qtynvtw.us-west-2.rds.amazonaws.com", user="", passwd="", db="dqn_results")

cur = db.cursor()









cur.execute("insert into experiments (label, x1, x2, x3, x4, y) values ('test', '0', '1', '2', '3', '4')")




db.commit()



query = ("select label, x1, x2, x3, x4, y from experiments")

cur.execute(query)

for (label, x1, x2, x3, x4, y) in cur:
  print(label)

cur.close()
db.close()




















