import sqlite3

db = sqlite3.connect("domino.db")

db.execute("""
CREATE TABLE teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    points INTEGER
)
""")

db.execute("""
CREATE TABLE matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER,
    points INTEGER
)
""")

db.commit()
db.close()

print("Base de datos creada correctamente")
