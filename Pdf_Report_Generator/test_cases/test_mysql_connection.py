import mysql.connector

try:
    cn = mysql.connector.connect(
        host="database-1.cvog840g81fo.ap-south-1.rds.amazonaws.com",
        port=3306,
        user="admin",
        password="stracer12345",
        database="defaultdb",
        connection_timeout=10
    )

    print("✅ MySQL connection succeeded")
    cn.close()

except mysql.connector.Error as e:
    print("❌ MySQL connection failed")
    print(e)
