import sqlite3
con=sqlite3.connect(r'file:data/arxiv_papers.db?mode=ro', uri=True)
cur=con.cursor()
query = '''
SELECT
  COUNT(*) AS total_rows,
  SUM(CASE WHEN categories LIKE '%cs.AI%' THEN 1 ELSE 0 END) AS cs_AI,
  SUM(CASE WHEN categories LIKE '%cs.LG%' THEN 1 ELSE 0 END) AS cs_LG,
  SUM(CASE WHEN categories LIKE '%cs.CV%' THEN 1 ELSE 0 END) AS cs_CV,
  SUM(CASE WHEN categories LIKE '%cs.CL%' THEN 1 ELSE 0 END) AS cs_CL,
  SUM(CASE WHEN categories LIKE '%stat.ML%' THEN 1 ELSE 0 END) AS stat_ML,
  SUM(CASE WHEN categories LIKE '%cs.RO%' THEN 1 ELSE 0 END) AS cs_RO,
  SUM(CASE WHEN categories LIKE '%cs.NE%' THEN 1 ELSE 0 END) AS cs_NE,
  SUM(CASE WHEN categories LIKE '%eess.SP%' THEN 1 ELSE 0 END) AS eess_SP
FROM papers;
'''
row = cur.execute(query).fetchone()
labels = ['total_rows','cs.AI','cs.LG','cs.CV','cs.CL','stat.ML','cs.RO','cs.NE','eess.SP']
for k, v in zip(labels, row):
    print(f'{k}\t{v}')
