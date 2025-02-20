import psycopg2

def connect_db():
    return psycopg2.connect(dbname="your_database", user="your_username", password="your_password", host="your_host", port="5432")

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
