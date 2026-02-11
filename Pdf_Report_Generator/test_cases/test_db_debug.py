from src.db import list_tables, describe_table, get_project_by_id

if __name__ == "__main__":
    list_tables()
    describe_table("tbl_project")
    get_project_by_id(149)
