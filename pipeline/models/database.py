import psycopg2

def connect_db():
    return psycopg2.connect(dbname="tsdb", user="tsdbadmin", password="twbnwtsaanybxcbd", host="efyjw8a4w9.wu5ui8n1r3.tsdb.cloud.timescale.com", port="36784")

def insert_person(id, features, height, stride_length, gender, age, glasses, hairline_id):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO Person (id, features, height, stride_length, gender, age, glasses, hairline_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);"
    cursor.execute(query, (id, features, height, stride_length, gender, age, glasses, hairline_id))
    conn.commit()
    cursor.close()
    conn.close()

def insert_hairline(id, type):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO Hairline (id, type) VALUES (%s, %s);"
    cursor.execute(query, (id, type))
    conn.commit()
    cursor.close()
    conn.close()

def insert_area(id, name):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO Area (id, name) VALUES (%s, %s);"
    cursor.execute(query, (id, name))
    conn.commit()
    cursor.close()
    conn.close()

def insert_event(person_id, area_id, time):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO Event (person_id, area_id, time) VALUES (%s, %s, %s);"
    cursor.execute(query, (person_id, area_id, time))
    conn.commit()
    cursor.close()
    conn.close()

def insert_apparel(person_id, shirt_colour, pant_colour, shoe_colour, time):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO Apparel (person_id, shirt_colour, pant_colour, shoe_colour, time) VALUES (%s, %s, %s, %s, %s);"
    cursor.execute(query, (person_id, shirt_colour, pant_colour, shoe_colour, time))
    conn.commit()
    cursor.close()
    conn.close()

def insert_track(person_id, time, action_id, duration, x, y, velocity):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO Track (person_id, time, action_id, duration, x, y, velocity) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    cursor.execute(query, (person_id, time, action_id, duration, x, y, velocity))
    conn.commit()
    cursor.close()
    conn.close()

def insert_action(id, type):
    conn = connect_db()
    cursor = conn.cursor()
    query = "INSERT INTO Action (id, type) VALUES (%s, %s);"
    cursor.execute(query, (id, type))
    conn.commit()
    cursor.close()
    conn.close()
