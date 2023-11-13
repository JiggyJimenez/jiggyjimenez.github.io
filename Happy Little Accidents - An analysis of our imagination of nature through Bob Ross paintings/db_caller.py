import sqlite3

def connect(db_name, path_only=False):
    """Anonymously connect to the given database from the LT folder"""
    db_path = "/mnt/processed/private/msds2023/lt5/" + db_name
    if path_only == True:
        return db_path
    else:
        conn = sqlite3.connect(db_path)
        return conn