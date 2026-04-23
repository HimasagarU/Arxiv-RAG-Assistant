import sqlite3, os
p="data/arxiv_papers.db"
print("db_exists", os.path.exists(p))
if os.path.exists(p):
    c=sqlite3.connect(p)
    cur=c.cursor()
    cur.execute("select count(*) from papers")
    print("rows", cur.fetchone()[0])
    try:
        cur.execute("select count(*) from papers where full_text is not null and length(trim(full_text))>0")
        print("full_text_nonempty", cur.fetchone()[0])
    except Exception as e:
        print("full_text_check_error", type(e).__name__, e)
    c.close()
