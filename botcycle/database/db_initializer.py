import sys
import os
import sqlite3

def init(sql_file_name, db_file_name):
    with open(sql_file_name, 'r') as sql_file:
        script = sql_file.read()
        conn = sqlite3.connect(db_file_name)
        cursor = conn.cursor()
        cursor.executescript(script)
        conn.commit()
        cursor.close()
        conn.close()

def execute_init_sql(db_file_name):
    sql_file_name = os.path.join(os.path.dirname(__file__), 'db_init.sql')
    return init(sql_file_name, db_file_name)

if __name__ == '__main__':
    init(sys.argv[1], sys.argv[2])
