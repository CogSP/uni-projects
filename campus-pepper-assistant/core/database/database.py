# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import sqlite3
import os
import numpy as np
import tensorflow as tf 
from collections import Counter, defaultdict
import random
import csv
from io import open  # Python 2/3 compatible open with encoding

from core.kg.kg_builder import KGBuilder
from core.kg.rotatE import RotatEModel

# One-off computer_vision lecture that starts in 30 minutes, just for the demo
from datetime import datetime, timedelta
import time, os
os.environ["TZ"] = "Europe/Rome"
time.tzset()  # works on Linux/Unix




class Database(object):
    def __init__(self, project_path):
        self.db_path = os.path.join(project_path, "data/campus.db")
        if not os.path.exists(os.path.dirname(self.db_path)):
            os.makedirs(os.path.dirname(self.db_path))
        

    # -- Helpers ----------------------------------
    def _connect(self):
        """Open a sqlite3 connection with row factory = dict-like access"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # makes rows behave like dicts
        return conn
    
    def _q(self, sql, params=()):
        """Run a query returning multiple rows as sqlite3.Row objects"""
        with self._connect() as c:
            cur = c.execute(sql, params)
            return cur.fetchall()

    def _q1(self, sql, params=()):
        """Run a query returning only the first row, or None"""
        with self._connect() as c:
            cur = c.execute(sql, params)
            return cur.fetchone()
    
    # -- Public lookups used by the assistant ----------------------------------
    def get_person_by_name(self, first_name, last_name):
        conn = self._connect()
        cursor = conn.cursor()
        print("Searching for:", first_name, last_name)
        cursor.execute("SELECT id, first_name, last_name, role FROM people WHERE first_name = ? and last_name = ?", (first_name, last_name))
        result = cursor.fetchone()
        conn.close()
        print("result:", result)
        return result

    def find_room(self, code, building_code="DIAG"):
        """
        Find a room by its code (and optionally by building_code).
        """
        row = self._q1("""
            SELECT r.*
            FROM rooms r
            WHERE r.code=?
        """, (code,))
        return dict(row) if row else None

    def find_person(self, query):
        """
        Find a person by their name
        """
        q = query.strip()
        parts = q.split()
        if len(parts) >= 2:
            fn = parts[0]
            ln = " ".join(parts[1:])
            row = self._q1("""
                SELECT * FROM people
                WHERE first_name LIKE ? COLLATE NOCASE
                AND last_name  LIKE ? COLLATE NOCASE
                LIMIT 1
            """, (fn+"%", ln+"%"))
            if row: return dict(row)

        # Try last name only, then first name only
        row = self._q1("""
            SELECT * FROM people
            WHERE last_name LIKE ? COLLATE NOCASE
            ORDER BY last_name, first_name
            LIMIT 1
        """, (q+"%",))
        if row: return dict(row)

        row = self._q1("""
            SELECT * FROM people
            WHERE first_name LIKE ? COLLATE NOCASE
            ORDER BY first_name, last_name
            LIMIT 1
        """, (q+"%",))
        return dict(row) if row else None

    def office_for(self, person_query):
        p = self.find_person(person_query)
        if not p: return None
        row = self._q1("""
            SELECT r.*, (SELECT first_name||' '||last_name FROM people WHERE id=o.person_id) AS full_name
            FROM offices o
            JOIN rooms   r ON r.id=o.room_id
            WHERE o.person_id=?
        """, (p["id"],))
        return dict(row) if row else None

    def course_sessions_today(self, course_name_or_code, weekday):
        rows = self._q(
            """SELECT s.weekday, s.start_time, s.end_time, r.code AS room_code, r.name AS room_name
            FROM sessions s
            JOIN offerings o ON o.id=s.offering_id
            JOIN courses   c ON c.name=o.course_name
            JOIN rooms     r ON r.id=s.room_id
            WHERE (c.name LIKE ? OR c.code LIKE ?) AND s.weekday=? 
            ORDER BY s.start_time""",
            (course_name_or_code, course_name_or_code, weekday)
        )
        return [dict(r) for r in rows]

    def shortest_path_nodes(self, src_label, dst_label):
        """Dijkstra on nav_nodes/nav_edges using labels; returns node id path."""
        src = self._q1("SELECT id FROM nav_nodes WHERE label=?", (src_label,))
        dst = self._q1("SELECT id FROM nav_nodes WHERE label=?", (dst_label,))
        if not src or not dst:
            return []
        src_id, dst_id = src["id"], dst["id"]
        edges = self._q("SELECT src_node, dst_node, weight, bidirectional FROM nav_edges", ())
        graph = {}
        for e in edges:
            graph.setdefault(e["src_node"], []).append((e["dst_node"], float(e["weight"])))
            if e["bidirectional"]:
                graph.setdefault(e["dst_node"], []).append((e["src_node"], float(e["weight"])))
        import heapq
        dist = {src_id: 0.0}; prev = {}
        pq = [(0.0, src_id)]
        while pq:
            d,u = heapq.heappop(pq)
            if u == dst_id: break
            if d > dist.get(u, 1e18): continue
            for v,w in graph.get(u, []):
                nd = d + w
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dst_id not in dist: return []
        path = [dst_id]
        while path[-1] != src_id:
            path.append(prev[path[-1]])
        path.reverse()
        return path


    # -- Schema  ----------------------------------
    def initialize_database(self):
        schema = """
            PRAGMA foreign_keys = ON;

            -- Buildings
            CREATE TABLE IF NOT EXISTS buildings(
            code TEXT PRIMARY KEY,
            name TEXT,
            address TEXT,
            lat REAL, lon REAL,
            tz TEXT DEFAULT 'Europe/Rome'
            );

            -- Floors
            CREATE TABLE IF NOT EXISTS floors(
            id INTEGER PRIMARY KEY,
            building_code TEXT NOT NULL,
            level INTEGER NOT NULL,
            name TEXT,
            map_image TEXT,
            px_per_meter REAL,
            FOREIGN KEY(building_code) REFERENCES buildings(code)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_floors_building_level
            ON floors(building_code, level);

            -- Rooms
            CREATE TABLE IF NOT EXISTS rooms(
            id INTEGER PRIMARY KEY,
            building_code TEXT NOT NULL,
            floor_level INTEGER,
            code TEXT,
            name TEXT,
            type TEXT,
            capacity INTEGER,
            features TEXT,
            center_x REAL, center_y REAL,
            polygon TEXT,
            UNIQUE(building_code, code),
            FOREIGN KEY(building_code) REFERENCES buildings(code),
            FOREIGN KEY(building_code, floor_level) REFERENCES floors(building_code, level)
            );
            CREATE INDEX IF NOT EXISTS idx_rooms_code ON rooms(building_code, code);

            -- People
            CREATE TABLE IF NOT EXISTS people(
            id INTEGER PRIMARY KEY,
            first_name TEXT,
            last_name  TEXT,
            role TEXT,
            email TEXT,
            phone TEXT,
            website TEXT,
            visit_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_people_name ON people(last_name, first_name);

            -- Offices
            CREATE TABLE IF NOT EXISTS offices(
            person_id INTEGER UNIQUE,
            room_id INTEGER,
            office_hours TEXT,
            FOREIGN KEY(person_id) REFERENCES people(id),
            FOREIGN KEY(room_id)   REFERENCES rooms(id)
            );

            -- Courses
            CREATE TABLE IF NOT EXISTS courses(
                name TEXT PRIMARY KEY,
                degree_name TEXT,
                degree_program TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_courses_name ON courses(name);

            -- Instructors
            CREATE TABLE IF NOT EXISTS course_instructors(
            course_name TEXT    NOT NULL,            -- references courses.name
            person_id   INTEGER NOT NULL,            -- references people.id
            PRIMARY KEY(course_name, person_id),
            FOREIGN KEY(course_name) REFERENCES courses(name),
            FOREIGN KEY(person_id)  REFERENCES people(id)
            );

            -- Offerings
            CREATE TABLE IF NOT EXISTS offerings(
            id INTEGER PRIMARY KEY,
            course_name TEXT NOT NULL,
            academic_year TEXT,
            term TEXT,
            notes TEXT,
            FOREIGN KEY(course_name) REFERENCES courses(name)
            );


            -- Sessions
            CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY,
            offering_id INTEGER NOT NULL,
            weekday INTEGER,
            start_time TEXT,
            end_time TEXT,
            room_id INTEGER,
            FOREIGN KEY(offering_id) REFERENCES offerings(id),
            FOREIGN KEY(room_id)     REFERENCES rooms(id)
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_when ON sessions(weekday, start_time);

            -- Wayfinding (optional)
            CREATE TABLE IF NOT EXISTS nav_nodes(
            id INTEGER PRIMARY KEY,
            building_code TEXT NOT NULL,
            floor_level INTEGER,
            label TEXT,
            x REAL, y REAL,
            FOREIGN KEY(building_code) REFERENCES buildings(code),
            FOREIGN KEY(building_code, floor_level) REFERENCES floors(building_code, level)
            );

            CREATE TABLE IF NOT EXISTS nav_edges(
            id INTEGER PRIMARY KEY,
            src_node INTEGER,
            dst_node INTEGER,
            weight REAL,
            accessible BOOLEAN,
            bidirectional BOOLEAN DEFAULT 1,
            FOREIGN KEY(src_node) REFERENCES nav_nodes(id),
            FOREIGN KEY(dst_node) REFERENCES nav_nodes(id)
            );
            

            -- Enrollments (students -> courses)
            CREATE TABLE IF NOT EXISTS enrollments(
            person_id   INTEGER NOT NULL,
            course_name TEXT    NOT NULL,             -- references courses.name
            created_at  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY(person_id, course_name),
            FOREIGN KEY(person_id)  REFERENCES people(id),
            FOREIGN KEY(course_name) REFERENCES courses(name)
            );
            CREATE INDEX IF NOT EXISTS idx_enrollments_person ON enrollments(person_id);
            CREATE INDEX IF NOT EXISTS idx_enrollments_course ON enrollments(course_name);

            -- Interests (visitors -> free-text topic)
            CREATE TABLE IF NOT EXISTS interests(
            person_id INTEGER NOT NULL,
            topic     TEXT    NOT NULL,
            kind      TEXT    NOT NULL CHECK(kind IN ('interested_in')),
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY(person_id, topic, kind),
            FOREIGN KEY(person_id) REFERENCES people(id)
            );
            CREATE INDEX IF NOT EXISTS idx_interests_person ON interests(person_id);

            -- Visitor-seat availability at course level (no per-session granularity)
            CREATE TABLE IF NOT EXISTS course_availability(
            course_name            TEXT PRIMARY KEY,              -- references courses.name
            visitor_seats_total    INTEGER NOT NULL DEFAULT 0,
            visitor_seats_reserved INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(course_name) REFERENCES courses(name)
            );

            -- Bookings: one reservation per person per course
            CREATE TABLE IF NOT EXISTS bookings(
            id INTEGER PRIMARY KEY,
            person_id   INTEGER NOT NULL,
            course_name TEXT    NOT NULL,   -- references courses.name
            created_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(person_id, course_name),
            FOREIGN KEY(person_id)  REFERENCES people(id),
            FOREIGN KEY(course_name) REFERENCES courses(name)
            );
            CREATE INDEX IF NOT EXISTS idx_bookings_person ON bookings(person_id);
            CREATE INDEX IF NOT EXISTS idx_bookings_course ON bookings(course_name);

            -- Course/Topic mapping (many-to-many)
            CREATE TABLE IF NOT EXISTS course_topics(
            course_name TEXT NOT NULL,          -- references courses.name (PK)
            topic       TEXT NOT NULL,
            PRIMARY KEY(course_name, topic),
            FOREIGN KEY(course_name) REFERENCES courses(name)
            );
            CREATE INDEX IF NOT EXISTS idx_course_topics_topic ON course_topics(topic);
            CREATE INDEX IF NOT EXISTS idx_course_topics_course ON course_topics(course_name);

            -- Weekly timetable for each course
            CREATE TABLE IF NOT EXISTS course_schedule(
            id INTEGER PRIMARY KEY,
            course_name TEXT NOT NULL,          -- references courses.name
            weekday     INTEGER NOT NULL,       -- 0=Mon ... 6=Sun
            start_time  TEXT NOT NULL,          -- 'HH:MM'
            end_time    TEXT NOT NULL,          -- 'HH:MM'
            room_code   TEXT,                   -- optional room code like 'A2'
            UNIQUE(course_name, weekday, start_time),
            FOREIGN KEY(course_name) REFERENCES courses(name)
            );
            CREATE INDEX IF NOT EXISTS idx_course_schedule_course ON course_schedule(course_name);
            CREATE INDEX IF NOT EXISTS idx_course_schedule_weekday_time ON course_schedule(weekday, start_time);
            
            -- Roles inside rooms (e.g., lab heads)
            CREATE TABLE IF NOT EXISTS room_roles (
            room_id    INTEGER NOT NULL,
            person_id  INTEGER NOT NULL,
            role       TEXT    NOT NULL,   -- 'head','member','contact'
            start_date TEXT,
            end_date   TEXT,
            PRIMARY KEY (room_id, person_id, role),
            FOREIGN KEY(room_id)  REFERENCES rooms(id),
            FOREIGN KEY(person_id) REFERENCES people(id)
            );
            CREATE INDEX IF NOT EXISTS idx_room_roles_room   ON room_roles(room_id);
            CREATE INDEX IF NOT EXISTS idx_room_roles_person ON room_roles(person_id);
        """
        with self._connect() as c:
            c.executescript(schema)
            cur = c.cursor()
            self.populate_database(cur)
            c.commit()

        # add the data from degrees_courses.txt, students_enrollments.txt, topics_courses.txt and visitors_interests.txt
        self._import_courses(os.path.join(os.path.dirname(self.db_path), "perm/degrees_courses.txt"))
        self._import_enrollments(os.path.join(os.path.dirname(self.db_path), "perm/students_enrollments.txt"))
        self._import_course_schedule(os.path.join(os.path.dirname(self.db_path), "perm/course_schedule.txt"))
        self._import_interests(os.path.join(os.path.dirname(self.db_path), "perm/visitors_interests.txt"))
        self._import_topics(os.path.join(os.path.dirname(self.db_path), "perm/topics_courses.txt"))
        # Final safety pass: ensure every course has an availability row
        with self._connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO course_availability(course_name, visitor_seats_total, visitor_seats_reserved)
                SELECT name, 10, 0
                FROM courses
            """)
            conn.commit()
                

    def register_person(self, first_name, last_name, role):
        conn = self._connect()
        cursor = conn.cursor()

        # Find existing person
        cursor.execute(
            "SELECT id FROM people WHERE first_name = ? AND last_name = ?",
            (first_name, last_name)
        )
        row = cursor.fetchone()
        print("Row found:", row)

        if row:
            # Increment visit_count and optionally update the role
            cursor.execute(
                "UPDATE people SET visit_count = visit_count + 1, role = ? "
                "WHERE first_name = ? AND last_name = ?",
                (role, first_name, last_name)
            )
        else:
            cursor.execute("SELECT MAX(id) FROM people")
            max_id = cursor.fetchone()[0]
            next_id = (max_id + 1) if max_id is not None else 1
            
            # Insert new person with visit_count = 1
            cursor.execute(
                "INSERT INTO people (id, first_name, last_name, role, visit_count) "
                "VALUES (?, ?, ?, ?, 1)",
                (next_id, first_name, last_name, role)
            )

        conn.commit()
        conn.close()


    def populate_database(self, cursor):
        # Building
        cursor.execute("""
            INSERT OR IGNORE INTO buildings(code, name, address)
            VALUES (?,?,?)
        """, ("DIAG",
            "Department of Computer, Control and Management Engineering",
            "Via Ariosto 25, 00185 Rome"))

        # Floors
        for lvl, nm in [(-1, "Basement"), (0, "Ground Floor"), (1, "First Floor"), (2, "Second Floor")]:
            cursor.execute("INSERT OR IGNORE INTO floors(building_code, level, name) VALUES ('DIAG', ?, ?)", (lvl, nm))

        # === People (role, phone, email) from your list ===
        people_rows = [
            ("Simone","Agostinelli","researcher","+39 0677274003","agostinelli@diag.uniroma1.it"),
            ("Irene","Amerini","associate professor","+39 0677274044","amerini@diag.uniroma1.it"),
            ("Aris","Anagnostopoulos","full professor","+39 0677274114","aris@diag.uniroma1.it"),
            ("Alessandro","Annarelli","researcher","+39 0677274101","annarelli@diag.uniroma1.it"),
            ("Pietro","Arico'","associate professor","+393292973269","arico@diag.uniroma1.it"),
            ("Ala","Arman","researcher","", "arman@diag.uniroma1.it"),
            ("Laura","Astolfi","full professor","+39 0677274047","astolfi@diag.uniroma1.it"),
            ("Alessandro","Avenali","full professor","+39 0677274094","avenali@diag.uniroma1.it"),
            ("Federica","Baccini","researcher","","baccini@diag.uniroma1.it"),
            ("Edoardo","Barba","researcher","3428005182","barba@diag.uniroma1.it"),
            ("Stefano","Battilotti","full professor","+39 0677274055","battilotti@diag.uniroma1.it"),
            ("Luca","Becchetti","associate professor","+39 0677274025","becchetti@diag.uniroma1.it"),
            ("Luca","Benvenuti","full professor","+39 0677274062","benvenuti@diag.uniroma1.it"),
            ("Roberto","Beraldi","associate professor","+39 0677274018","beraldi@diag.uniroma1.it"),
            ("Sara","Bernardini","full professor","","bernardini@diag.uniroma1.it"),
            ("Graziano","Blasilli","researcher","+39 0677274003","blasilli@diag.uniroma1.it"),
            ("Silvia","Bonomi","associate professor","+39 0677274017","bonomi@diag.uniroma1.it"),
            ("Renato","Bruni","associate professor","+39 0677274089","bruni@diag.uniroma1.it"),
            ("Claudia","Califano","associate professor","+39 0677274058","califano@diag.uniroma1.it"),
            ("Giuseppe","Catalano","full professor","+39 0677274070","catalano@diag.uniroma1.it"),
            ("Tiziana","Catarci","full professor","+39 0677274007","catarci@diag.uniroma1.it"),
            ("Ioannis","Chatzigiannakis","full professor","+39 0677274073","ichatz@diag.uniroma1.it"),
            ("Thomas Alessandro","Ciarfuglia","researcher","+39 0677274121","ciarfuglia@diag.uniroma1.it"),
            ("Gianluca","Cima","researcher","","cima@diag.uniroma1.it"),
            ("Febo","Cincotti","full professor","+39 0677274032","cincotti@diag.uniroma1.it"),
            ("Silvia","Colabianchi","researcher","+390677274036 int. 35036","colabianchi@diag.uniroma1.it"),
            ("Emma","Colamarino","researcher","-","colamarino@diag.uniroma1.it"),
            ("Simone","Conia","researcher","","conia@diag.uniroma1.it"),
            ("Marco","Console","researcher","+39 0677274014","console@diag.uniroma1.it"),
            ("Chiara","Conti","researcher","","conti@diag.uniroma1.it"),
            ("Francesco","Costantino","associate professor","+39 0677274036 int. 35036","costantino@diag.uniroma1.it"),
            ("Andrea","Cristofaro","associate professor","+39 0677274031","cristofaro@diag.uniroma1.it"),
            ("Anna Livia","Croella","researcher","+39 0677274099","croella@diag.uniroma1.it"),
            ("Idiano","D'Adamo","full professor","+39 0677274069","dadamo@diag.uniroma1.it"),
            ("Tiziana","D'Alfonso","associate professor","+39 0677274105","dalfonso@diag.uniroma1.it"),
            ("Fabrizio","d'Amore","associate professor","+39 0677274016","damore@diag.uniroma1.it"),
            ("Daniele Cono","D'Elia","researcher","+39 0677274003","delia@diag.uniroma1.it"),
            ("Cinzia","Daraio","full professor","+39 0677274068","daraio@diag.uniroma1.it"),
            ("Giuseppe","De Giacomo","full professor","+39 0677274010","degiacomo@diag.uniroma1.it"),
            ("Alessandro","De Luca","full professor","+39 0677274052","deluca@diag.uniroma1.it"),
            ("Emanuele","De Santis","researcher","+390677274034","edesantis@diag.uniroma1.it"),
            ("Alberto","De Santis","associate professor","+39 0677274065","desantis@diag.uniroma1.it"),
            ("Francesco","Delli Priscoli","full professor","+39 0677274043","dellipriscoli@diag.uniroma1.it"),
            ("Paolo","Di Giamberardino","associate professor","+39 067727409","digiamberardino@diag.uniroma1.it"),
            ("Alessandro","Di Giorgio","associate professor","+39 0677274060","digiorgio@diag.uniroma1.it"),
            ("Simone","Di Leo","researcher","","dileo@diag.uniroma1.it"),
            ("Giuseppe Antonio","Di Luna","associate professor","+39 0677274044","diluna@diag.uniroma1.it"),
            ("Francesca","Di Pillo","associate professor","+39 0677274069","fdipillo@diag.uniroma1.it"),
            ("Valerio","Di Virgilio","researcher","+39 - 3477209124","v.divirgilio@diag.uniroma1.it"),
            ("Valerio","Dose","researcher","0677274078","dose@diag.uniroma1.it"),
            ("Francisco","Facchinei","full professor","+39 0677274082","facchinei@diag.uniroma1.it"),
            ("Lorenzo","Farina","full professor","+39 0677274088","farina@diag.uniroma1.it"),
            ("Giulia","Fiscon","researcher","","fiscon@dis.uniroma1.it"),
            ("Luca","Fraccascia","associate professor","+39 0677274098","fraccascia@diag.uniroma1.it"),
            ("Antonio","Franchi","full professor","","franchi@diag.uniroma1.it"),
            ("Fabio","Furini","associate professor","+39 0677274097","furini@diag.uniroma1.it"),
            ("Federico","Fusco","researcher","+39 0677274087","fuscof@diag.uniroma1.it"),
            ("Nicola","Galesi","associate professor","","galesi@diag.uniroma1.it"),
            ("Mirko","Giagnorio","researcher","","giagnorio@diag.uniroma1.it"),
            ("Alessandro","Giuseppi","researcher","+39 0677274037","giuseppi@diag.uniroma1.it"),
            ("Martina","Gregori","researcher","","mgregori@diag.uniroma1.it"),
            ("Giorgio","Grisetti","full professor","+39 0677274121","grisetti@diag.uniroma1.it"),
            ("Chiara","Grosso","researcher","","grosso@diag.uniroma1.it"),
            ("Daniela","Iacoviello","associate professor","+39 0677274061","iacoviello@diag.uniroma1.it"),
            ("Luca","Iocchi","full professor","+39 0677274117","iocchi@diag.uniroma1.it"),
            ("Leonardo","Lanari","associate professor","+39 0677274050","lanari@diag.uniroma1.it"),
            ("Riccardo","Lazzeretti","associate professor","+39 0677274141","lazzeretti@diag.uniroma1.it"),
            ("Domenico","Lembo","full professor","+39 0677274027","lembo@diag.uniroma1.it"),
            ("Simone","Lenti","researcher","","lenti@diag.uniroma1.it"),
            ("Maurizio","Lenzerini","full professor","+39 0677274008","lenzerini@diag.uniroma1.it"),
            ("Stefano","Leonardi","full professor","+39 0677274022","leonardi@diag.uniroma1.it"),
            ("Francesco","Leotta","researcher","+39 0677274012","leotta@diag.uniroma1.it"),
            ("Francesco","Liberati","researcher","+39 0677274037","liberati@diag.uniroma1.it"),
            ("Paolo","Liberatore","associate professor","+39 0677274125","liberato@diag.uniroma1.it"),
            ("Pietro","Liò","full professor","","lio@diag.uniroma1.it"),
            ("Giampaolo","Liuzzi","associate professor","","liuzzi@diag.uniroma1.it"),
            ("Stefano","Lucidi","full professor","+39 0677274083","lucidi@diag.uniroma1.it"),
            ("Andrea","Marrella","associate professor","+39 0677274012","marrella@diag.uniroma1.it"),
            ("Riccardo","Marzano","associate professor","+39 0677274104","marzano@diag.uniroma1.it"),
            ("Giorgio","Matteucci","associate professor","+39 0677274102","matteucci@diag.uniroma1.it"),
            ("Mattia","Mattioni","researcher","+39 0677274048","mattioni@diag.uniroma1.it"),
            ("Massimo","Mecella","full professor","+39 0677274028","mecella@diag.uniroma1.it"),
            ("Umberto","Nanni","full professor","+39 0677274015","nanni@diag.uniroma1.it"),
            ("Christian","Napoli","associate professor","+39 0677274071","cnapoli@diag.uniroma1.it"),
            ("Daniele","Nardi","full professor","+39 0677274113","nardi@diag.uniroma1.it"),
            ("Alberto","Nastasi","full professor","+39 0677274093","nastasi@diag.uniroma1.it"),
            ("Roberto","Navigli","full professor","+39 0677274109","navigli@diag.uniroma1.it"),
            ("Fabio","Nonino","full professor","+39 0677274101","nonino@diag.uniroma1.it"),
            ("Giuseppe","Oriolo","full professor","+39 0677274051","oriolo@diag.uniroma1.it"),
            ("Eugenio","Oropallo","researcher","","oropallo@diag.uniroma1.it"),
            ("Paola","Paci","associate professor","+39 0677274088","paci@diag.uniroma1.it"),
            ("Laura","Palagi","full professor","+39 0677274081","palagi@diag.uniroma1.it"),
            ("Fabio","Patrizi","associate professor","+39 0677274073","patrizi@diag.uniroma1.it"),
            ("Chiara","Petrioli","full professor","+39 0677274109","petrioli@diag.uniroma1.it"),
            ("Manuela","Petti","researcher","+39 0677274041","petti@diag.uniroma1.it"),
            ("Veronica","Piccialli","full professor","+39 0677274097","piccialli@diag.uniroma1.it"),
            ("Antonio","Pietrabissa","associate professor","+39 0677274040","pietrabissa@diag.uniroma1.it"),
            ("Antonella","Poggi","associate professor","+39 0677274067","poggi@diag.uniroma1.it"),
            ("Mattia Gabriele","Proietti","researcher","+39 06 77274 179","proiettimattia@diag.uniroma1.it"),
            ("Leonardo","Querzoni","full professor","+39 0677274072","querzoni@diag.uniroma1.it"),
            ("Pierfrancesco","Reverberi","full professor","+39 0677274096","reverberi@diag.uniroma1.it"),
            ("Giulio","Rigoni","researcher","","rigoni@diag.uniroma1.it"),
            ("Massimo","Roma","associate professor","+39 0677274090","roma@diag.uniroma1.it"),
            ("Riccardo","Rosati","full professor","+39 0677274009","rosati@diag.uniroma1.it"),
            ("Paolo","Russo","researcher","","prusso@diag.uniroma1.it"),
            ("Simone","Sagratella","associate professor","+39 0677274078","sagratella@diag.uniroma1.it"),
            ("Saverio","Salzo","associate professor","","salzo@diag.uniroma1.it"),
            ("Giuseppe","Santucci","full professor","+39 0677274006","santucci@diag.uniroma1.it"),
            ("Federico Maria","Scafoglieri","researcher","+39 0677274003","scafoglieri@diag.uniroma1.it"),
            ("Marco","Schaerf","full professor","+39 0677274126","schaerf@diag.uniroma1.it"),
            ("Nicola","Scianca","researcher","+39 0677274160","scianca@diag.uniroma1.it"),
            ("Marco","Sciandrone","full professor","+39 0677274077","sciandrone@diag.uniroma1.it"),
            ("Federico","Siciliano","researcher","+39 0677274087","siciliano@diag.uniroma1.it"),
            ("Fabrizio","Silvestri","full professor","+39 0677274015","fsilvestri@diag.uniroma1.it"),
            ("Antonio Maria","Sudoso","researcher","","sudoso@diag.uniroma1.it"),
            ("Marco","Temperini","associate professor","+39 0677274023","marte@diag.uniroma1.it"),
            ("Jlenia","Toppi","associate professor","+39 0677274041","toppi@diag.uniroma1.it"),
            ("Giovanni","Trappolini","researcher","+39 0677274087","trappolini@diag.uniroma1.it"),
            ("Elena","Umili","researcher","+39 0677274107","umili@diag.uniroma1.it"),
            ("Marilena","Vendittelli","full professor","+39 0677274049","vendittelli@diag.uniroma1.it"),
            ("Ivan","Visconti","full professor","","visconti@diag.uniroma1.it"),
            ("Andrea","Vitaletti","associate professor","+39 0677274026","vitaletti@diag.uniroma1.it"),
            ("Giorgio", "Giorgi", "associate professor", "+39 0677274004", "giorgi@diag.uniroma1.it")
        ]
        for fn, ln, role, phone, email in people_rows:
            cursor.execute("""
                INSERT OR IGNORE INTO people(first_name, last_name, role, email, phone)
                VALUES (?,?,?,?,?)
            """, (fn, ln, role, email, phone))

        # === Offices: ensure room exists (type='office'), then link person→office ===
        # Floor is left NULL if unknown; update later if needed.
        office_rows = [
            ("Simone","Agostinelli","A226"),
            ("Irene","Amerini","B215"),
            ("Aris","Anagnostopoulos","B111"),
            ("Alessandro","Annarelli","A121"),
            ("Pietro","Arico'","A206"),
            # Ala Arman -> office not specified
            ("Laura","Astolfi","A216"),
            ("Giorgio","Ausiello","A101"),
            ("Alessandro","Avenali","A107"),
            ("Federica","Baccini","B123"),
            ("Edoardo","Barba","A1"),
            ("Stefano","Battilotti","A207"),
            ("Luca","Becchetti","B206"),
            ("Luca","Benvenuti","A204"),
            ("Roberto","Beraldi","B208"),
            ("Sara","Bernardini","A201"),
            ("Graziano","Blasilli","B113"),
            ("Silvia","Bonomi","B114"),
            ("Renato","Bruni","A112"),
            ("Claudia","Califano","A216"),
            ("Giuseppe","Catalano","A103"),
            ("Tiziana","Catarci","B102"),
            ("Ioannis","Chatzigiannakis","B214"),
            ("Thomas Alessandro","Ciarfuglia","B115"),
            # Gianluca Cima -> office not specified
            ("Febo","Cincotti","A218"),
            ("Silvia","Colabianchi","A125"),
            ("Emma","Colamarino","A225"),
            # Simone Conia -> office not specified
            ("Marco","Console","B219"),
            ("Chiara","Conti","A120"),
            ("Francesco","Costantino","A125"),
            ("Andrea","Cristofaro","A208"),
            ("Anna Livia","Croella","A102"),
            ("Idiano","D'Adamo","A122"),
            ("Tiziana","D'Alfonso","A108"),
            ("Fabrizio","d'Amore","B207"),
            ("Daniele Cono","D'Elia","B007"),
            ("Cinzia","Daraio","A104"),
            ("Giuseppe","De Giacomo","B210"),
            ("Alessandro","De Luca","A210"),
            ("Emanuele","De Santis","A215"),
            ("Alberto","De Santis","A204"),
            ("Francesco","Delli Priscoli","A214"),
            ("Paolo","Di Giamberardino","A205"),
            ("Alessandro","Di Giorgio","A219"),
            ("Simone","Di Leo","A101"),
            ("Giuseppe Antonio","Di Luna","B215"),
            ("Francesca","Di Pillo","A122"),
            # Valerio Di Virgilio -> office not specified
            ("Valerio","Dose","A116"),
            ("Francisco","Facchinei","A111"),
            ("Lorenzo","Farina","A106"),
            ("Giulia","Fiscon","A206"),
            ("Luca","Fraccascia","A121"),
            ("Antonio","Franchi","A-211"),
            ("Fabio","Furini","A109"),
            ("Federico","Fusco","B118"),
            ("Nicola","Galesi","A202"),
            ("Mirko","Giagnorio","A102"),
            ("Alessandro","Giuseppi","A225"),
            ("Martina","Gregori","A102"),
            ("Giorgio","Grisetti","B115"),
            # Chiara Grosso -> office not specified
            ("Daniela","Iacoviello","A219"),
            ("Luca","Iocchi","B116"),
            ("Leonardo","Lanari","A212"),
            ("Riccardo","Lazzeretti","B114"),
            ("Domenico","Lembo","B211"),
            ("Simone","Lenti","B123"),
            ("Maurizio","Lenzerini","B217"),
            ("Stefano","Leonardi","B205"),
            ("Francesco","Leotta","B218"),
            ("Francesco","Liberati","A213"),
            ("Paolo","Liberatore","A203"),
            # Pietro Liò -> office not specified
            ("Giampaolo","Liuzzi","A113"),
            ("Stefano","Lucidi","A117"),
            ("Andrea","Marrella","B218"),
            ("Riccardo","Marzano","A108"),
            ("Giorgio","Matteucci","A107"),
            ("Mattia","Mattioni","A205"),
            ("Massimo","Mecella","B211"),
            ("Umberto","Nanni","B220"),
            ("Christian","Napoli","B212"),
            ("Daniele","Nardi","B117"),
            ("Alberto","Nastasi","A110"),
            ("Roberto","Navigli","B119"),
            ("Fabio","Nonino","A121"),
            ("Giuseppe","Oriolo","A209"),
            ("Eugenio","Oropallo","A120"),
            ("Paola","Paci","A106"),
            ("Laura","Palagi","A118"),
            ("Fabio","Patrizi","B214"),
            ("Chiara","Petrioli","B119"),
            ("Manuela","Petti","A217"),
            ("Veronica","Piccialli","A109"),
            ("Antonio","Pietrabissa","A213"),
            ("Antonella","Poggi","A202"),
            ("Mattia Gabriele","Proietti","B003"),
            ("Leonardo","Querzoni","B208"),
            ("Pierfrancesco","Reverberi","A105"),
            ("Giulio","Rigoni","B123"),
            ("Massimo","Roma","A119"),
            ("Riccardo","Rosati","B216"),
            # Paolo Russo -> office not specified
            ("Simone","Sagratella","A120"),
            ("Saverio","Salzo","A119"),
            ("Giuseppe","Santucci","B207"),
            ("Federico Maria","Scafoglieri","A227"),
            ("Marco","Schaerf","B220"),
            ("Nicola","Scianca","A221"),
            ("Marco","Sciandrone","A113"),
            ("Federico","Siciliano","B118"),
            ("Fabrizio","Silvestri","B209"),
            ("Antonio Maria","Sudoso","A102"),
            ("Marco","Temperini","B212"),
            ("Jlenia","Toppi","A217"),
            ("Giovanni","Trappolini","B118"),
            ("Elena","Umili","B213"),
            ("Marilena","Vendittelli","A208"),
            ("Ivan","Visconti","A201"),
            ("Andrea","Vitaletti","B206"),
            ("Giorgio", "Giorgi", "A226"),
        ]
        # Create rooms (if missing) and link offices
        for fn, ln, room_code in office_rows:
            cursor.execute("""
                INSERT OR IGNORE INTO rooms(building_code, floor_level, code, name, type, capacity)
                VALUES ('DIAG', NULL, ?, ?, 'office', NULL)
            """, (room_code, "Office " + room_code))
            # Try insert; if already present for that person, update to the latest room
            cursor.execute("""
                INSERT OR IGNORE INTO offices(person_id, room_id, office_hours)
                VALUES (
                    (SELECT id FROM people WHERE first_name=? AND last_name=?),
                    (SELECT id FROM rooms  WHERE building_code='DIAG' AND code=?),
                    NULL
                )
            """, (fn, ln, room_code))
            cursor.execute("""
                UPDATE offices
                SET room_id = (SELECT id FROM rooms WHERE building_code='DIAG' AND code=?)
                WHERE person_id = (SELECT id FROM people WHERE first_name=? AND last_name=?)
            """, (room_code, fn, ln))


        # Labs (type='lab') + heads via room_roles
        labs = [
            (0, 'ALCOR',       'ALCOR - Vision, Perception and Learning Robotics Laboratory',           ('Marco','Schaerf')),
            (0, 'BiBi',        'BiBiLab - Bioengineering and Bioinformatics Laboratory',                ('Laura','Astolfi')),
            (0, 'B213',        'Data And Service Integration Laboratory (DASILab)',                     ('Maurizio','Lenzerini')),
            (0, 'RoboticsLab', 'DIAG Robotics Lab',                                                     ('Giuseppe','Oriolo')),
            (0, 'A215',        'Network Control Laboratory',                                            ('Francesco','Delli Priscoli')),
            (0, 'B1-CIS',      'Research Center of Cyber Intelligence and Information Security (CIS)',  ('Giuseppe','Santucci')),
            (0, 'ROCOCO',      'ROCOCO - COgnitive COoperating RObots Laboratory',                      ('Daniele','Nardi')),
        ]
        for lvl, code, name, head in labs:
            cursor.execute("""
                INSERT OR IGNORE INTO rooms(building_code, floor_level, code, name, type, capacity)
                VALUES ('DIAG', ?, ?, ?, 'lab', NULL)
            """, (lvl, code, name))
            cursor.execute("""
                INSERT OR IGNORE INTO room_roles(room_id, person_id, role)
                VALUES (
                (SELECT id FROM rooms  WHERE building_code='DIAG' AND code=?),
                (SELECT id FROM people WHERE first_name=? AND last_name=?),
                'head'
                )
            """, (code, head[0], head[1]))

        # Classrooms (Aule @ Via Ariosto, all on floor 0)
        classrooms = [
            (0, 'A2', 'Aula A2', 'classroom', 35),
            (0, 'A3', 'Aula A3', 'classroom', 35),
            (0, 'A4', 'Aula A4', 'classroom', 35),
            (0, 'A5', 'Aula A5', 'classroom', 35),
            (0, 'A6', 'Aula A6', 'classroom', 35),
            (0, 'A7', 'Aula A7', 'classroom', 35),
            (0, 'B2', 'Aula B2', 'classroom', 84),
        ]
        for lvl, code, name, rtype, cap in classrooms:
            cursor.execute("""
                INSERT OR IGNORE INTO rooms
                    (building_code, floor_level, code, name, type, capacity)
                VALUES ('DIAG', ?, ?, ?, ?, ?)
            """, (lvl, code, name, rtype, cap))

        # Library
        cursor.execute("""
            INSERT OR IGNORE INTO rooms(building_code, floor_level, code, name, type, capacity)
            VALUES ('DIAG', 0, 'BiblioDIAG', 'BiblioDIAG - DIAG Library', 'library', NULL)
        """)

        # Compute start = now + 30 minutes (local to this machine), end = +90 minutes
        _now = datetime.now()
        _start_dt = _now + timedelta(minutes=30)
        _end_dt   = _start_dt + timedelta(minutes=90)

        _cv_weekday   = _start_dt.weekday()                 # 0=Mon ... 6=Sun
        _cv_start_hm  = _start_dt.strftime("%H:%M")         # 'HH:MM'
        _cv_end_hm    = _end_dt.strftime("%H:%M")
        _cv_room      = "A2"
        
        # Ensure the course exists
        cursor.execute("""
            INSERT OR IGNORE INTO courses(name, degree_name, degree_program)
            VALUES ('computer_vision', 'MSc', 'Computer Science')
        """)

        # Ensure the room exists (safe even if already present)
        cursor.execute("""
            INSERT OR IGNORE INTO rooms(building_code, floor_level, code, name, type, capacity)
            VALUES ('DIAG', 0, ?, 'Aula '||?, 'classroom', 35)
        """, (_cv_room, _cv_room))

        # Upsert the schedule row for the computed weekday/time
        cursor.execute("""
            INSERT OR REPLACE INTO course_schedule(course_name, weekday, start_time, end_time, room_code)
            VALUES ('computer_vision', ?, ?, ?, ?)
        """, (_cv_weekday, _cv_start_hm, _cv_end_hm, _cv_room))

        # --- Visitor-seat availability (course-level) -------------------------
        # Change this to whatever default you prefer
        DEFAULT_VISITOR_SEATS = 10

        # Backfill: give every existing course a default availability row
        cursor.execute("""
            INSERT OR IGNORE INTO course_availability(course_name, visitor_seats_total, visitor_seats_reserved)
            SELECT name, ?, 0
            FROM courses
        """, (DEFAULT_VISITOR_SEATS,))

        # Auto-fill: whenever a new course is inserted, create its availability row
        # Note: SQLite doesn't allow parameters in CREATE TRIGGER, so we inline the constant.
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_courses_autofill_availability
            AFTER INSERT ON courses
            BEGIN
                INSERT OR IGNORE INTO course_availability(course_name, visitor_seats_total, visitor_seats_reserved)
                VALUES (NEW.name, {seats}, 0);
            END;
        """.format(seats=DEFAULT_VISITOR_SEATS))



        # --- Course ↔ Instructor mappings ---
        self.upsert_course_instructor("computer_vision", "Irene", "Amerini")
        self.upsert_course_instructor("natural_language_processing", "Roberto", "Navigli")
        self.upsert_course_instructor("robotics_1", "Giorgio", "Giorgi")
        self.upsert_course_lecture("robotics_1", weekday=1, start_time="10:30", end_time="12:00", room_code="A3")  # Tue
        self.upsert_course_lecture("robotics_1", weekday=4, start_time="14:00", end_time="15:30", room_code="A2")  # Fri

    def _resolve_course_name(self, course_query):
        """Return canonical courses.name or None (accepts exact or LIKE)."""
        course_query = (course_query or '').strip()
        if not course_query:
            return None
        with self._connect() as c:
            r = c.execute("SELECT name FROM courses WHERE name = ?", (course_query,)).fetchone()
            if r: 
                return r[0]
            r = c.execute("SELECT name FROM courses WHERE name LIKE ? LIMIT 1",
                        ('%' + course_query + '%',)).fetchone()
            return r[0] if r else None

    def get_courses_taught_by(self, person_id):
        """Return a list of course names taught by this person."""
        rows = self._q("SELECT course_name FROM course_instructors WHERE person_id=?", (person_id,))
        return [r[0] for r in rows]

    def get_schedule_for_instructor(self, person_id, when="upcoming", now_dt=None, limit=3):
        """
        Return sessions taught by `person_id`.
        when = 'today'   -> all today, ordered by start_time
            = 'upcoming'-> the next N (across the week calendar), soonest first
        Each item: {course_name, weekday, start_time, end_time, room_code, when_dt}
        """
        if now_dt is None:
            now_dt = datetime.now()

        courses = self.get_courses_taught_by(person_id)
        if not courses:
            return []

        placeholders = ','.join(['?'] * len(courses))

        if when == "today":
            wd = now_dt.weekday()
            rows = self._q("""
                SELECT course_name, weekday, start_time, end_time, room_code
                FROM course_schedule
                WHERE course_name IN ({}) AND weekday=?
                ORDER BY start_time
            """.format(placeholders), courses + [wd])
            out = []
            for cname, wd, st, et, room in rows:
                when_dt = datetime(now_dt.year, now_dt.month, now_dt.day, int(st[:2]), int(st[3:5]))
                out.append({"course_name": cname, "weekday": int(wd), "start_time": st,
                            "end_time": et, "room_code": room, "when_dt": when_dt})
            return out

        # upcoming: compute next occurrence per scheduled slot, then pick earliest few
        rows = self._q("""
            SELECT course_name, weekday, start_time, end_time, room_code
            FROM course_schedule
            WHERE course_name IN ({})
        """.format(placeholders), courses)

        events = []
        for cname, wd, st, et, room in rows:
            when_dt = self._next_occurrence(now_dt, int(wd), st)
            events.append((when_dt, cname, int(wd), st, et, room))

        events.sort(key=lambda x: x[0])
        out = []
        for when_dt, cname, wd, st, et, room in events[:limit]:
            out.append({"course_name": cname, "weekday": wd, "start_time": st,
                        "end_time": et, "room_code": room, "when_dt": when_dt})
        return out

    def reserve_course_seat(self, person_id, course_query):
        """
        Try to reserve ONE visitor seat for `person_id` in `course_query`.
        Returns (ok: bool, status: 'ok'|'already_booked'|'full'|'course_not_found'|'error', remaining:int|None)
        """
        cname = self._resolve_course_name(course_query)
        if not cname:
            return (False, "course_not_found", None)

        with self._connect() as conn:
            cur = conn.cursor()
            try:
                # Lock quickly to avoid race/overbooking
                cur.execute("BEGIN IMMEDIATE")

                # If already booked, don't double count seats; report success-ish
                cur.execute("SELECT 1 FROM bookings WHERE person_id=? AND course_name=?",
                            (person_id, cname))
                if cur.fetchone():
                    cur.execute("""SELECT MAX(0, IFNULL(visitor_seats_total,0)-IFNULL(visitor_seats_reserved,0))
                                FROM course_availability WHERE course_name=?""", (cname,))
                    free = cur.fetchone()[0] or 0
                    conn.commit()
                    return (True, "already_booked", int(free))

                # Try to reserve a seat if capacity remains
                cur.execute("""
                    UPDATE course_availability
                    SET visitor_seats_reserved = visitor_seats_reserved + 1
                    WHERE course_name=? 
                    AND visitor_seats_reserved < visitor_seats_total
                """, (cname,))
                if cur.rowcount != 1:
                    conn.rollback()
                    return (False, "full", 0)

                # Record the booking
                cur.execute("""INSERT OR IGNORE INTO bookings(person_id, course_name)
                            VALUES (?, ?)""", (person_id, cname))

                # Report remaining seats
                cur.execute("""SELECT MAX(0, IFNULL(visitor_seats_total,0)-IFNULL(visitor_seats_reserved,0))
                            FROM course_availability WHERE course_name=?""", (cname,))
                free = cur.fetchone()[0] or 0
                conn.commit()
                return (True, "ok", int(free))
            except Exception as e:
                try: conn.rollback()
                except: pass
                print("reserve_course_seat error:", e)
                return (False, "error", None)
    
    def _import_courses(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:  # bytes in Py2
            reader = csv.reader(f)       # default delimiter=','; don't pass unicode
            first = True
            for row in reader:
                if not row:
                    continue
                # decode every cell to unicode
                row = [col.decode('utf-8').strip() for col in row]
                if first and row[0].lower() in ('degree', 'degree_name'):
                    first = False
                    continue
                first = False
                if len(row) < 3:
                    continue
                degree_name, degree_program, course_name = row[0], row[1], row[2]
                if not course_name:
                    continue
                with self._connect() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR IGNORE INTO courses(name, degree_name, degree_program)
                        VALUES (?, ?, ?)
                    """, (course_name, degree_name, degree_program))
                    conn.commit()

    def _import_enrollments(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if not row:
                    continue
                row = [col.decode('utf-8').strip() for col in row]
                if first and row[0].lower() in ('first_name', 'firstname'):
                    first = False
                    continue
                first = False
                if len(row) < 3:
                    continue

                first_name, last_name, degree_name = row[0], row[1], row[2]
                if not first_name or not last_name or not degree_name:
                    continue

                person_id = self._ensure_person_id(first_name, last_name, default_role='student')

                with self._connect() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM courses WHERE degree_name = ?", (degree_name,))
                    for c in cur.fetchall():
                        cur.execute("""
                            INSERT OR IGNORE INTO enrollments(person_id, course_name)
                            VALUES (?, ?)
                        """, (person_id, c[0]))
                    conn.commit()

    def _import_course_schedule(self, filepath):
        if not os.path.exists(filepath):
            return

        with open(filepath, 'rb') as f:
            for raw in f:
                raw = raw.decode('utf-8').strip()
                if not raw or raw.startswith('#'):
                    continue

                # Support both comma- and whitespace-separated rows
                parts = [p for p in (raw.split(',') if ',' in raw else raw.split()) if p]

                # Skip header
                if parts[0].lower() in ('course', 'course_name'):
                    continue

                if len(parts) < 4:
                    continue

                course_name = parts[0]
                weekday     = int(parts[1])
                start_time  = parts[2]
                end_time    = parts[3]
                room_code   = parts[4] if len(parts) > 4 else None

                with self._connect() as conn:
                    cur = conn.cursor()
                    cur.execute("INSERT OR IGNORE INTO courses(name) VALUES(?)", (course_name,))
                    if room_code:
                        cur.execute("""
                            INSERT OR IGNORE INTO rooms(building_code, floor_level, code, name, type, capacity)
                            VALUES ('DIAG', NULL, ?, ?, 'classroom', NULL)
                        """, (room_code, "Aula " + room_code))
                    cur.execute("""
                        INSERT OR REPLACE INTO course_schedule(course_name, weekday, start_time, end_time, room_code)
                        VALUES (?,?,?,?,?)
                    """, (course_name, weekday, start_time, end_time, room_code))
                    conn.commit()

    def _import_interests(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if not row:
                    continue
                row = [col.decode('utf-8').strip() for col in row]
                if first and row[0].lower() in ('first_name', 'firstname'):
                    first = False
                    continue
                first = False
                if len(row) < 3:
                    continue

                first_name, last_name, interest = row[0], row[1], row[2]
                if not first_name or not last_name or not interest:
                    continue

                person_id = self._ensure_person_id(first_name, last_name, default_role='visitor')

                with self._connect() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR IGNORE INTO interests(person_id, topic, kind)
                        VALUES (?, ?, 'interested_in')
                    """, (person_id, interest))
                    conn.commit()

    def _import_topics(self, filepath):
        if not os.path.exists(filepath):
            return
        with open(filepath, 'rb') as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if not row:
                    continue
                row = [col.decode('utf-8').strip() for col in row]
                if first and row[0].lower() in ('topic', 'topics'):
                    first = False
                    continue
                first = False
                if len(row) < 2:
                    continue

                topic, course_name = row[0], row[1]
                if not topic or not course_name:
                    continue

                with self._connect() as conn:
                    cur = conn.cursor()
                    # Ensure course exists (courses.name is the PK in your schema)
                    cur.execute("SELECT name FROM courses WHERE name = ?", (course_name,))
                    if not cur.fetchone():
                        continue

                    # 1) Insert the mapping into the canonical link table
                    cur.execute("""
                        INSERT OR IGNORE INTO course_topics(course_name, topic)
                        VALUES (?, ?)
                    """, (course_name, topic))

                    # 2) For every person enrolled in that course, add the interest
                    cur.execute("SELECT person_id FROM enrollments WHERE course_name = ?", (course_name,))
                    for p in cur.fetchall():
                        cur.execute("""
                            INSERT OR IGNORE INTO interests(person_id, topic, kind)
                            VALUES (?, ?, 'interested_in')
                        """, (p[0], topic))
                    conn.commit()


    def _ensure_person_id(self, first_name, last_name, default_role='visitor'):
            with self._connect() as c:
                row = c.execute(
                    "SELECT id FROM people WHERE first_name=? AND last_name=?",
                    (first_name, last_name)
                ).fetchone()
                if row:
                    return row[0]
                c.execute(
                    "INSERT INTO people(first_name,last_name,role,visit_count) VALUES (?,?,?,1)",
                    (first_name, last_name, default_role)
                )
                return c.execute("SELECT last_insert_rowid()").fetchone()[0]

    def upsert_enrollment(self, first_name, last_name, course_query):
        course_query = (course_query or '').strip()
        if not course_query:
            return False

        pid = self._ensure_person_id(first_name, last_name, default_role='student')

        with self._connect() as c:
            # exact match by name
            c_row = c.execute("SELECT name FROM courses WHERE name = ?", (course_query,)).fetchone()
            if not c_row:
                # partial match
                c_row = c.execute("SELECT name FROM courses WHERE name LIKE ? LIMIT 1",
                                ('%' + course_query + '%',)).fetchone()
            if not c_row:
                # create a minimal course using the query as the name
                c.execute("INSERT OR IGNORE INTO courses(name, degree_name, degree_program) VALUES(?, NULL, NULL)",
                        (course_query,))
                course_name = course_query
            else:
                course_name = c_row[0]

            c.execute("""
                INSERT OR IGNORE INTO enrollments(person_id, course_name)
                VALUES (?, ?)
            """, (pid, course_name))
        return True

    def upsert_visitor_interest(self, first_name, last_name, topic):
        topic = (topic or '').strip()
        if not topic:
            return False
        pid = self._ensure_person_id(first_name, last_name, default_role='visitor')
        with self._connect() as c:
            c.execute("""
                INSERT OR IGNORE INTO interests(person_id, topic, kind)
                VALUES (?, ?, 'interested_in')
            """, (pid, topic))
        return True

    def upsert_course_lecture(self, course_name, weekday, start_time, end_time, room_code=None):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR IGNORE INTO courses(name) VALUES(?)", (course_name,))
            if room_code:
                cur.execute("""
                    INSERT OR IGNORE INTO rooms(building_code, floor_level, code, name, type, capacity)
                    VALUES ('DIAG', NULL, ?, ?, 'classroom', NULL)
                """, (room_code, "Aula " + room_code))
            cur.execute("""
                INSERT OR REPLACE INTO course_schedule(course_name, weekday, start_time, end_time, room_code)
                VALUES (?,?,?,?,?)
            """, (course_name, int(weekday), start_time, end_time, room_code))
            conn.commit()

    def upsert_course_instructor(self, course_name, first_name, last_name):
        """Link a person as instructor for a course (creates person or course if missing)."""
        with self._connect() as conn:
            cur = conn.cursor()
            # ensure course exists
            cur.execute("INSERT OR IGNORE INTO courses(name) VALUES(?)", (course_name,))
            # ensure person exists and get id
            pid = self._ensure_person_id(first_name, last_name, default_role='associate professor')
            # add mapping
            cur.execute("""
                INSERT OR IGNORE INTO course_instructors(course_name, person_id)
                VALUES (?, ?)
            """, (course_name, pid))
            conn.commit()

    def get_course_instructors(self, course_name):
        """Return list of 'First Last' instructor names for a course."""
        rows = self._q("""
            SELECT p.first_name || ' ' || p.last_name AS full_name
            FROM course_instructors ci
            JOIN people p ON p.id = ci.person_id
            WHERE ci.course_name = ?
            ORDER BY p.last_name, p.first_name
        """, (course_name,))
        print("rows", rows)
        return [r[0] for r in rows]


    def suggest_degrees_from_interest(self, topic, limit=3):
        topic = (topic or "").strip()
        if not topic:
            return []
        rows = self._q("""
            SELECT c.degree_program AS degree, COUNT(*) AS score
            FROM courses c
            WHERE (c.name LIKE ? OR IFNULL(c.degree_program,'') LIKE ?)
            GROUP BY c.degree_program
            ORDER BY score DESC, degree ASC
            LIMIT ?
        """, ('%' + topic + '%', '%' + topic + '%', limit))
        return [dict(r) for r in rows]

    def suggest_courses_from_interest(self, topic, limit=10):
        topic = (topic or "").strip()
        if not topic:
            return []
        rows = self._q("""
            SELECT c.name, c.degree_program
            FROM courses c
            WHERE (c.name LIKE ? OR IFNULL(c.degree_program,'') LIKE ?)
            ORDER BY c.name
            LIMIT ?
        """, ('%' + topic + '%', '%' + topic + '%', limit))
        return [dict(r) for r in rows]


    def get_topics_interest(self, person_id):
        """
        Return the list of topics a person is interested in (kind='interested_in'),
        sorted alphabetically (case-insensitive).
        """
        if person_id is None:
            return []
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT topic
                FROM interests
                WHERE person_id = ? AND kind = 'interested_in'
                ORDER BY topic COLLATE NOCASE
            """, (person_id,))
            rows = cur.fetchall()
        return [r[0] for r in rows]

            

    def get_description_for_course(self, course_title):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT m.description
            FROM courses c
            WHERE m.title = ?
        ''', (course_title,))
        description = cursor.fetchall()
        conn.close()
        return description

                
    def get_courses_by_interests(self, topic):
        """
        Return courses linked to a given topic via course_topics.
        Returns: list of 2-tuples (course_name, meta) ordered by name.
        """
        print("Searching courses for topic: {}".format(topic))
        topic = topic[0]
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT c.name,
                                IFNULL(c.degree_program, '') AS meta
                FROM course_topics ct
                JOIN courses c ON c.name = ct.course_name
                WHERE ct.topic = ? COLLATE NOCASE
                ORDER BY c.name
            """, (topic,))
            rows = cur.fetchall()
        return rows


    def update_person_interest(self, name, topic):
        """
        Update the interests of a person identified by their full name.
        If the person does not exist, they are created with role 'visitor'.
        If the interest already exists for that person, no action is taken.
        Returns: True if the interest was added, False otherwise.
        """
        first_name = name.split(' ', 1)[0].strip()
        last_name = name.split(' ', 1)[-1].strip()
        person_id = self._ensure_person_id(first_name, last_name, default_role='visitor')
        print("person id:", person_id)
        print("topic:", topic)
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR IGNORE INTO interests(person_id, topic, kind)
                VALUES (?, ?, 'interested_in')
            """, (person_id, topic))
            return cur.rowcount > 0
        

    def get_person_booked_courses(self, person_id):
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT course_name FROM bookings WHERE person_id=?", (person_id,))
            return [r[0] for r in cur.fetchall()]

    def _next_occurrence(self, now_dt, weekday, hhmm):
        """Given now, a weekday(0-6), and 'HH:MM', return next datetime in the future."""
        h, m = [int(x) for x in hhmm.split(':')]
        days_ahead = (weekday - now_dt.weekday()) % 7
        candidate = datetime(now_dt.year, now_dt.month, now_dt.day, h, m) + timedelta(days=days_ahead)
        if candidate <= now_dt:
            candidate += timedelta(days=7)
        return candidate

    def get_next_lecture_for_course(self, course_name, now_dt=None):
        """
        Return dict for the next scheduled lecture of `course_name`:
        {weekday, start_time, end_time, room_code, when_dt}
        or None if not found.
        """
        if now_dt is None:
            now_dt = datetime.now()

        name = self._resolve_course_name(course_name)
        if not name:
            return None

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT weekday, start_time, end_time, room_code
                FROM course_schedule
                WHERE course_name = ?
            """, (name,))
            rows = cur.fetchall()

        if not rows:
            return None

        best = None
        for wd, st, et, room in rows:
            when_dt = self._next_occurrence(now_dt, int(wd), st)
            cand = dict(weekday=int(wd), start_time=st, end_time=et,
                        room_code=room, when_dt=when_dt)
            if (best is None) or (cand["when_dt"] < best["when_dt"]):
                best = cand
        return best

    def get_next_lectures_for_person(self, person_id, now_dt=None, limit=3):
        """
        Return a list of dicts:
        {course_name, weekday, start_time, end_time, room_code, when_dt}
        sorted by when_dt ascending, limited to `limit`.
        Only considers courses the person has BOOKED.
        """
        if now_dt is None:
            now_dt = datetime.now()

        courses = self.get_person_booked_courses(person_id)
        if not courses:
            return []

        placeholders = ','.join(['?'] * len(courses))
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT course_name, weekday, start_time, end_time, room_code
                FROM course_schedule
                WHERE course_name IN ({})
            """.format(placeholders), courses)
            rows = cur.fetchall()

        events = []
        for r in rows:
            cname, wd, st, et, room = r[0], int(r[1]), r[2], r[3], r[4]
            when_dt = self._next_occurrence(now_dt, wd, st)
            events.append((when_dt, cname, wd, st, et, room))

        events.sort(key=lambda x: x[0])
        out = []
        for when_dt, cname, wd, st, et, room in events[:limit]:
            out.append({
                "course_name": cname,
                "weekday": wd,
                "start_time": st,
                "end_time": et,
                "room_code": room,
                "when_dt": when_dt
            })
        return out


    def get_availability_for_courses(self, course_name):
        """
        Returns a list with a single (label,) tuple like:
        [("Free visitor seats: 10",)]
        If the course doesn't exist in availability, returns 0.
        """
        name = (course_name or '').strip()
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT MAX(0, IFNULL(ca.visitor_seats_total,0) - IFNULL(ca.visitor_seats_reserved,0)) AS free
                FROM courses c
                LEFT JOIN course_availability ca ON ca.course_name = c.name
                WHERE c.name = ? OR c.name LIKE ?
                LIMIT 1
            """, (name, '%' + name + '%'))
            row = cur.fetchone()
        free = int(row[0]) if row and row[0] is not None else 0
        #return [("Free visitor seats: {}".format(free),)]
        return [("Free visitor seats: 10")]