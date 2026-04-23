import sqlite3
con=sqlite3.connect(r'file:data/arxiv_papers.db?mode=ro', uri=True)
cur=con.cursor()
print('TABLES')
for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
    print(r[0])
print('SCHEMA')
for r in cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"):
    print('\nTABLE', r[0])
    print(r[1])
