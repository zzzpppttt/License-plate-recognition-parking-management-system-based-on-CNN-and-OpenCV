import sqlite3
from datetime import datetime
# 连接到SQLite数据库（如果不存在，则会创建新的数据库文件）
conn = sqlite3.connect('parking_management.db')


# 创建车辆信息表格
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE vehicles (
        id INT IDENTITY(1,1) PRIMARY KEY,
        license_plate VARCHAR(10),
        entry_time DATETIME
    )
''')
conn.commit()


# 插入车辆信息记录
def insert_vehicle_info(license_plate):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO vehicles (license_plate, entry_time)
        VALUES (?, ?)
    ''', (license_plate, current_time))
    conn.commit()


# 查询车辆信息记录
def get_vehicle_info(license_plate):
    cursor.execute('''
        SELECT * FROM vehicles WHERE license_plate = ?
    ''', (license_plate,))
    result = cursor.fetchone()
    if result:
        vehicle_id, license_plate, entry_time = result
        print(f"License Plate: {license_plate}")
        print(f"Entry Time: {entry_time}")
        return entry_time
    else:
        print("Vehicle not found.")


# 查询车牌信息记录
def get_license_plate_info():
    cursor.execute('''
        SELECT license_plate FROM vehicles
    ''')
    result = cursor.fetchone()
    if result:
        print(result)
        return result
    else:
        print("Vehicle not found.")


#删除车辆信息
def delete_vehicle_info(license_plate):
    cursor.execute('''
        DELETE FROM vehicles WHERE license_plate = ?
    ''', (license_plate,))
    print('delete successfullly')
    conn.commit()
# 示例插入和查询车辆信息
# insert_vehicle_info("ABC123")
# get_vehicle_info("ABC123")
# delete_vehicle_info("ABC123")


# 关闭数据库连接
conn.close()
