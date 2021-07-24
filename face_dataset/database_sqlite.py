# encoding: utf-8
import sqlite3, base64
import os, io
import numpy as np
import cv2


def _adapt_array(arr):
    """
    convert numpy array to Binary data into sqlite server.
    :param arr: a numpy array
    :return: Binary in sqlite
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_array(text):
    """
    convert text into numpy array
    :param text: Binary in sqlite
    :return: numpy array
    """
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def create_table(database_path: str):
    """
    create a sqlite table
    usage:
        create_table(./face_database.db)
    2021-07-19 Jie Y.
    :param database_path: str, the path and filename of sqlite database
    :return: None
    """
    if not os.path.exists(database_path):
        conn = sqlite3.connect(database_path, detect_types=sqlite3.PARSE_DECLTYPES)
        c = conn.cursor()
        c.execute("""CREATE TABLE FACE 
                    (
                     NAME   TEXT                  NOT NULL,
                     AGE    INT                   ,
                     IMAGE  TEXT                  NOT NULL,
                     ENCODING   ARRAY              NOT NULL
                    )""")
        conn.commit()
        print("Create a new table")
        conn.close()
        del conn


def insert_record(database_path: str, record: dict):
    """
    insert a record into sqlite database
    2021-07-19  Jie Y.
    usage:
        insert(./face_database.db,
                {'NAME': 'Defang F.', 'IMAGE': 'base64imagestr',
                'ENCODING': 'encoding list'})
    :param database_path: str, the path and filename of sqlite database
    :param record: dict, the dict of record, the keys should be [NAME, (AGE), IMGAE, ENCODING]
    :return:
    """

    keys = ','.join(list(record.keys()))
    placeholder = ','.join('?' for _ in range(len(record.keys())))
    sqlstr = f"INSERT INTO FACE ({keys}) VALUES ({placeholder})"

    conn = sqlite3.connect(database_path, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(sqlstr, tuple(record.values()))
    conn.commit()
    conn.close()


def select_record(database_path):
    """
    select record from sqlite database
    2021-07-22  Jie Y.  Init
    :param database_path: str, the path and filename of sqlite database
    :return: cursor: sqlite.Cursor object, there will be four element in each row.
        0:  str, NAME
        1:  int or NoneType, AGE
        2:  bytes, IMAGE. Should be decode from base64 algorithm.
            pic = base64.b64decode(row[2])
            img = cv2.imdecode(np.frombuffer(pic, np.uint8), cv2.IMREAD_COLOR)
        3:  ndarray, ENCODING
    """
    conn = sqlite3.connect(database_path, detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    sqlstr = "SELECT * FROM FACE"
    cursor = c.execute(sqlstr)
    return cursor


sqlite3.register_adapter(np.ndarray, _adapt_array)
sqlite3.register_converter("ARRAY", _convert_array)

if __name__ == '__main__':

    create_table('./face_database.db')
    """
    img_res = base64.b64encode(open('images/fang.png', 'rb').read())

    coding = np.array([i for i in range(128)])
    insert_record('./face_database.db',
                  {'NAME': 'Defang F.', 'AGE': 28, 'IMAGE': img_res,
                   'ENCODING': coding})

    select_record('./face_database.db')
    """
