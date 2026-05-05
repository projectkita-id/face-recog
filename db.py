import pickle
import sqlite3


class UserDatabase:
    """Handle user data storage: gestures + face encoding."""

    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                gesture_1 TEXT NOT NULL,
                gesture_2 TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

    def add_user(self, username, gesture_1, gesture_2, face_encoding):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            face_encoding_bytes = pickle.dumps(face_encoding)
            cursor.execute(
                """
                INSERT INTO users (username, gesture_1, gesture_2, face_encoding)
                VALUES (?, ?, ?, ?)
            """,
                (username, gesture_1, gesture_2, face_encoding_bytes),
            )
            conn.commit()
            conn.close()
            return True, f"User '{username}' berhasil ditambahkan"
        except sqlite3.IntegrityError:
            return False, f"Username '{username}' sudah terdaftar"
        except Exception as exc:
            return False, f"Error: {str(exc)}"

    def get_user(self, username):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT username, gesture_1, gesture_2, face_encoding
                FROM users WHERE username = ?
            """,
                (username,),
            )
            result = cursor.fetchone()
            conn.close()
            if result:
                username, gesture_1, gesture_2, face_encoding_bytes = result
                face_encoding = pickle.loads(face_encoding_bytes)
                return {
                    "username": username,
                    "gesture_1": gesture_1,
                    "gesture_2": gesture_2,
                    "face_encoding": face_encoding,
                }
            return None
        except Exception as exc:
            print(f"Error getting user: {str(exc)}")
            return None

    def get_all_users(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT username, gesture_1, gesture_2 FROM users")
            results = cursor.fetchall()
            conn.close()
            return [
                {"username": r[0], "gesture_1": r[1], "gesture_2": r[2]}
                for r in results
            ]
        except Exception as exc:
            print(f"Error getting users: {str(exc)}")
            return []

    def get_all_users_with_encoding(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT username, gesture_1, gesture_2, face_encoding
                FROM users
            """
            )
            results = cursor.fetchall()
            conn.close()
            users = []
            for username, gesture_1, gesture_2, face_encoding_bytes in results:
                face_encoding = pickle.loads(face_encoding_bytes)
                users.append(
                    {
                        "username": username,
                        "gesture_1": gesture_1,
                        "gesture_2": gesture_2,
                        "face_encoding": face_encoding,
                    }
                )
            return users
        except Exception as exc:
            print(f"Error getting users: {str(exc)}")
            return []

    def delete_user(self, username):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            conn.commit()
            conn.close()
            return True, f"User '{username}' berhasil dihapus"
        except Exception as exc:
            return False, f"Error: {str(exc)}"

    def user_exists(self, username):
        return self.get_user(username) is not None
