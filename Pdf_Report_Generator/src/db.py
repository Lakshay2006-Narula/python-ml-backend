import mysql.connector
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

# Basic validation
missing = [k for k, v in DB_CONFIG.items() if not v]
if missing:
    raise RuntimeError(f"Missing DB config values: {missing}")


def get_connection():
    """
    Create and return a MySQL connection
    """
    return mysql.connector.connect(
        **DB_CONFIG,
        autocommit=True,
        connection_timeout=600
    )


# =====================================================
# DEBUG / INSPECTION HELPERS
# =====================================================

def list_tables():
    cn = get_connection()
    cur = cn.cursor()
    cur.execute("SHOW TABLES;")
    tables = cur.fetchall()

    print("Tables in database:")
    for t in tables:
        print("-", t[0])

    cur.close()
    cn.close()


def describe_table(table_name: str):
    cn = get_connection()
    cur = cn.cursor()
    cur.execute(f"DESCRIBE {table_name};")
    cols = cur.fetchall()

    print(f"\nColumns in {table_name}:")
    for col in cols:
        print(col)

    cur.close()
    cn.close()


# =====================================================
# CORE DATA ACCESS FUNCTIONS (SHARED CONNECTION SAFE)
# =====================================================

def get_project_by_id(project_id: int, cn):
    cur = cn.cursor(dictionary=True)

    query = """
    SELECT *
    FROM tbl_project
    WHERE id = %s
    """

    cur.execute(query, (project_id,))
    row = cur.fetchone()

    cur.close()
    return row


def get_network_logs_for_sessions(session_ids: list[int], cn) -> pd.DataFrame:
    """
    Fetch network log rows for given session IDs
    Returns pandas DataFrame
    """

    if not session_ids:
        return pd.DataFrame()

    placeholders = ",".join(["%s"] * len(session_ids))

    query = f"""
    SELECT *
    FROM tbl_network_log
    WHERE session_id IN ({placeholders})
    """

    df = pd.read_sql(query, cn, params=session_ids)
    return df


def get_project_regions(project_id: int, cn) -> list[dict]:
    """
    Fetch project polygons as WKT (MySQL-safe)
    """

    cur = cn.cursor(dictionary=True)

    query = """
    SELECT
        id,
        name,
        ST_AsText(region) AS region_wkt
    FROM map_regions
    WHERE tbl_project_id = %s
      AND status = 1
    """

    cur.execute(query, (project_id,))
    rows = cur.fetchall()

    cur.close()
    return rows


def get_user_thresholds(user_id: int, debug: bool = False) -> dict | None:
    cn = get_connection()
    cur = cn.cursor(dictionary=True)

    query = """
        SELECT *
        FROM thresholds
        WHERE user_id = %s
        LIMIT 1
    """
    cur.execute(query, (user_id,))
    row = cur.fetchone()

    cur.close()
    cn.close()

    # Only print if debug mode is enabled
    if debug:
        print("\n================ DB THRESHOLD ROW =================")
        print(f"user_id = {user_id}")
        if not row:
            print("NO ROW RETURNED FROM DB")
            return None

        for k, v in row.items():
            print(f"{k}: {repr(v)}")

        print("===================================================\n")
    
    return row


def get_user_by_id(user_id: int) -> dict | None:
    cn = get_connection()
    cur = cn.cursor(dictionary=True)

    query = """
        SELECT *
        FROM tbl_user
        WHERE id = %s
        LIMIT 1
    """
    cur.execute(query, (user_id,))
    row = cur.fetchone()

    cur.close()
    cn.close()
    return row
