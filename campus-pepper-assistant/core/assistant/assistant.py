# -*- coding: utf-8 -*-
from io import open
from datetime import datetime,timedelta
import random
import re
import io
import os
import time
import ast
from datetime import datetime, timedelta
 
class Assistant(object):
    def __init__(self, animation, memory, database, mws, motion_manager):
        self.animation=animation
        self.memory = memory
        self.db = database
        self.mws = mws
        self.motion = motion_manager
        self.current_order = []  # Track items for order
        self.is_tablet=False
        self.bathroom_busy=False
        #hour = random.randint(15, 20)
        hour=21
        #minute = random.randint(0, 59)
        minute=15
        self.current_time = datetime.strptime("{:02}:{:02}".format(hour, minute), "%H:%M")

        
    def handle_function(self, value):

        print("Handling function:", value)

        if value == "greet_person":
        
            try:
                # Correct format with package name
                self.animation.run(".lastUploadedChoregrapheBehavior/animations/Stand/Gestures/Hey_1")
            except Exception as e:
                print("Animation error:", e)

            # Make sure Pepper is looking at the speaker right away
            # 1) try tracking (if available), else 2) center head slightly down
            started = self.motion.enable_face_look(mode="Head")
            if not started:
                self.motion.quick_look_front()

            # Reset flags on new greeting
            name = self.memory.getData("campus/person_name")
            first_name = name.split(" ")[0]
            last_name = " ".join(name.split(" ")[1:]) if len(name.split(" ")) > 1 else ""
            print("Greet person with name:", name)
            person = self.db.get_person_by_name(first_name=first_name, last_name=last_name)
            if person:
                self.current_person = person
                role = (person[3] or "").lower()
                print("Recognized person:", person)
                print("role:", role)
                is_prof = ("professor" in role) # matches "associate professor", "full professor", etc.
                print("is_prof:", is_prof)
                honorific = u"Prof. {}".format(last_name if last_name else first_name)
                print("honorific:", honorific)

                # make flag + honorific available to the dialog
                self.memory.insertData("campus/is_professor", "true" if is_prof else "false")
                self.memory.insertData("campus/person_honorific", honorific)
                
                if is_prof:
                    print("Professor detected, running welcome back animation.")
                    text = {
                        ("*", "*", "it", "*"): u"Buongiorno {}.".format(honorific),
                        ("*", "*", "*", "*"):  u"Good Morning, {}.".format(honorific)
                    }
                else:
                    text = {
                        ("*", "*", "it", "*"): u"Bentornato {}  ".format(first_name),
                        ("*", "*", "*", "*"):  u"Welcome back {}! ".format(first_name)
                    }
        
                buttons = {
                    "guide_next": {"it": "Portami alla prossima lezione", "en": "Guide me to next lecture"},
                    "my_schedule": {"it": "Il mio orario", "en": "My Schedule"},
                    "find_room": {"it": "Controlla aula", "en": "Check room"}
                }

                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.create_action(text = text, buttons=buttons, filename="welcome-back")
                self.mws.csend("im.execute('welcome-back')")

                if is_prof:
                    self.memory.raiseEvent("campus/person_identity_check", "professor")
                    #print("campus/person_identity_check now has value: ", self.memory.getData("campus/person_identity_check"))
                else:
                    self.memory.raiseEvent("campus/person_identity_check", "true")
                    #print("campus/person_identity_check now has value: ", self.memory.getData("campus/person_identity_check"))

                # wait for the click and route it
                answer = self.mws.csend("im.ask('welcome-back', timeout=999)")
                if answer == "my_schedule":
                    # same behavior you already have
                    self.memory.insertData("campus/prof_schedule_scope", "upcoming")
                    self.memory.raiseEvent("campus/function", "prof_schedule")

                elif answer == "guide_next":
                    # only useful if you had already saved a room into memory
                    room = (self.memory.getData("campus/next_lecture_room") or "").strip()
                    if room:
                        self.memory.insertData("campus/target_location", room)
                        self.memory.raiseEvent("campus/function", "guide_to_location")

                elif answer == "find_room":
                    # expects campus/room_query to be set by dialog
                    code = self.memory.getData("campus/room_query")
                    if code:
                        self.memory.raiseEvent("campus/function", "find_room_by_code")
                    else:
                        print("find_room pressed but no $campus/room_query set")



                try:
                    person_id = self.current_person[0]  # (id, first_name, last_name, role)
                    upcoming = self.db.get_next_lectures_for_person(person_id, now_dt=datetime.now(), limit=3)
                    #print("Upcoming lectures:", upcoming)
                    if upcoming:

                        # Pick earliest upcoming (your DB already returns them ordered; if not, sort by when_dt)
                        next_e = upcoming[0]
                        room_code = (next_e.get("room_code") or "").strip()
                        if room_code:
                            # Save target room regardless of the “soon” flag so the button always works
                            self.memory.insertData("campus/next_lecture_room", room_code)

                        # Localized weekday short names
                        wd_en = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                        wd_it = ['Lun','Mar','Mer','Gio','Ven','Sab','Dom']

                        lines_en, lines_it = [], []
                        for e in upcoming:
                            wd = e["weekday"]
                            st = e["start_time"]
                            et = e["end_time"]
                            room = e["room_code"] or "TBD"
                            cname = e["course_name"]
                            lines_en.append(u"• {} — {}–{} in {} ({})".format(wd_en[wd], st, et, room, cname))
                            lines_it.append(u"• {} — {}–{} in {} ({})".format(wd_it[wd], st, et, room, cname))

                        print("lines_en:", lines_en)
                        text = {
                            ("*", "*", "it", "*"): u"Le tue prossime lezioni prenotate:\n" + u"\n".join(lines_it),
                            ("*", "*", "*", "*"):  u"Your next booked lectures: computer vision in A2 at 10:30"
                        }

                        # Add the button only if we know a real room
                        buttons = None
                        if room_code and room_code.upper() != "TBD":
                            buttons = {
                                "guide_next": {
                                    "it": "Portami lì",
                                    "en": "Guide me there"
                                }
                            }

                        self.create_action(image="img/calendar.png", text=text, buttons=buttons, filename="next-lectures")
                        self.mws.csend("im.executeModality('BUTTONS', [])")
                        self.mws.csend("im.execute('next-lectures')")

                        answer = self.mws.csend("im.ask('next-lectures', timeout=999)")
                        if answer == "guide_next":
                            room = (self.memory.getData("campus/next_lecture_room") or "").strip()
                            if room:
                                self.memory.insertData("campus/target_location", room)
                                self.memory.raiseEvent("campus/function", "guide_to_location")

                        # store the text so $campus/next_lecture_text can be said in the .top
                        self.memory.insertData("campus/next_lecture_text", lines_en[0])
                        # raise the event
                        self.memory.raiseEvent("campus/next_lecture", "true")
                        #print("campus/next_lecture now has value: ", self.memory.getData("campus/next_lecture"))

                        try:
                            next_e = upcoming[0]  # earliest by when_dt
                            when_dt = next_e.get("when_dt")
                            room_code = (next_e.get("room_code") or "").strip()
                            if when_dt is not None:
                                mins_to_start = int((when_dt - datetime.now()).total_seconds() / 60.0)
                                if 0 <= mins_to_start < 60 and room_code and room_code.upper() != "TBD":
                                    self.memory.insertData("campus/next_lecture_room", room_code)
                                    self.memory.insertData("campus/next_lecture_start_time", next_e.get("start_time"))
                                    self.memory.raiseEvent("campus/next_lecture_soon", "true")
                                    #print("campus/next_lecture_soon now has value: ", self.memory.getData("campus/next_lecture_soon"))
                                else:
                                    self.memory.raiseEvent("campus/next_lecture_soon", "false")
                            else:
                                self.memory.raiseEvent("campus/next_lecture_soon", "false")

                        except Exception as e:
                            print("next_lecture_soon flag error:", e)
                            self.memory.raiseEvent("campus/next_lecture_soon", "false")
                            self.memory.raiseEvent("campus/next_lecture_soon", "false")

                    else:
                        self.memory.raiseEvent("campus/next_lecture", "false")
                        self.memory.raiseEvent("campus/next_lecture_soon", "false")
                        #print("campus/next_lecture now has value: ", self.memory.getData("campus/next_lecture"))

                except Exception as e:
                    print("Reminder error:", e)
               
            else:
                print("New person detected, running welcome animation.")

                try:
                    self.animation.run(".lastUploadedChoregrapheBehavior/animations/Stand/Gestures/Enthusiastic_4") # removed _async=True
                except Exception as e:
                    print("Animation error:", e)
                # Create new person welcome action
                text = {
                    ("*", "*", "it", "*"): "Benvenuto al DIAG! Vorrei sapere alcune cose per migliorare la tua esperienza qui.",
                    ("*", "*", "*", "*"): "Welcome to DIAG! I would like to know few things to improve your experience here."
                }

                buttons = {
                    "register_visitor": {"it": "Sono un visitatore", "en": "I'm a visitor"},
                    "register_student": {"it": "Sono uno studente", "en": "I'm a student"}
                }

                
                self.create_action(
                    image="img/campus.png",
                    text=text,
                    buttons = buttons,
                    filename="new-person-welcome"
                )

                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('new-person-welcome')")

                self.memory.raiseEvent("campus/person_identity_check", "false")
                #print("campus/person_identity_check now has value: ", self.memory.getData("campus/person_identity_check"))
        
        elif value == "restart":
            self.animation.run(".lastUploadedChoregrapheBehavior/animations/Stand/Gestures/Hey_1",_async=True)
            for key in self.memory.getDataList("campus/"):
                self.memory.insertData(key, "")
            text = {
                ("*", "*", "it", "*"): "Addio.",
                ("*", "*", "*", "*"): "Goodbye."
            }
            self.create_action(
                image="img/campus.png",
                text=text,
                filename="goodbye"
            )
            self.mws.csend("im.executeModality('BUTTONS', [])")
            self.mws.csend("im.execute('goodbye')")
            self.motion.guide_to_location("entrance")
            self.memory.raiseEvent("campus/restart", "true")

        elif value == "register_person":
            name = self.memory.getData("campus/person_name")
            
            first_name = name.split(" ")[0]
            last_name = " ".join(name.split(" ")[1:]) if len(name.split(" ")) > 1 else ""
            role = self.memory.getData("campus/person_role") or "visitor"

            print("Registering person")
            self.db.register_person(first_name, last_name, role)
            self.current_person=self.db.get_person_by_name(first_name=first_name, last_name=last_name)
            print(self.current_person, "registered successfully.")
            self.animation.run(".lastUploadedChoregrapheBehavior/animations/Stand/Gestures/Yes_1",_async=True)
            # Create registration success action
            text = {
                ("*", "*", "it", "*"): "Perfetto {}! Il tuo profilo e stato creato.".format(first_name),
                ("*", "*", "*", "*"): "Perfect {}! Your profile has been created.".format(first_name)
            }
            
            self.create_action(
                image="img/registration_success.png",
                text=text,
                filename="registration-success"
            )
            self.mws.csend("im.executeModality('BUTTONS', [])")
            self.mws.csend("im.execute('registration-success')")
            if "student" in role.lower():
                self.memory.raiseEvent("campus/person_registered", "newstudent")
            else:
                self.memory.raiseEvent("campus/person_registered", "visitor")
            #print("campus/person_registered now has value: ", self.memory.getData("campus/person_registered"))

        elif value == "find_room_by_code":
            code = self.memory.getData("campus/room_query")
            if code:
                code = unicode(code).strip()
            # default to DIAG until we support more buildings
            bcode = "DIAG"
            room = self.db.find_room(code, building_code=bcode)
            print("Finding room by code:", code, "found:", room)
            if room:
                # build the response text
                fname = room.get("floor_name")
                flev = room.get("floor_level")
                rname = room.get("room_name") or code
                bname = room.get("building_name") or bcode
                
                where_bits = []
                if bname: 
                    where_bits.append(u"Building {}".format(bname))
                if fname: 
                    where_bits.append(u"Floor {}".format(fname))
                elif flev is not None:
                    where_bits.append(u"Floor {}".format(flev))
                where_txt = u", ".join(where_bits) if where_bits else u""
                msg = u"{} ({}) is in {}.".format(rname, code, where_txt) if where_txt else u"{} ({}) located here.".format(rname, code)

                print("Room found message:", msg)

                # Save + signal the dialog
                self.memory.insertData("campus/find_room_result", msg)
                # Also stash for a possible follow-up guidance request
                self.memory.raiseEvent("campus/find_room_result", code + "!")
            else:
                self.memory.raiseEvent("campus/find_room_result", 0)


        elif value == "show_directions":
            location = self.memory.getData("campus/direction_request")
            print("Show directions to:", location)

            if location:
                # Generate map (MotionManager saves <room>_path.png under tablet/img/)
                verbal_direction = self.motion.point_and_describe_direction(location)

                # reference it via tablet relative path + cache-bust
                image_rel = "img/{}_path.png".format(location)
                bust = str(int(time.time()*1000))
                image_rel = "{}?t={}".format(image_rel, bust)

                text = {
                    ("*", "*", "it", "*"): "Direzioni per {}".format(location),
                    ("*", "*", "*", "*"): "Directions to {}".format(location)
                }
                self.create_action(
                    image=image_rel,
                    text=text,
                    filename="directions-with-map"
                )

                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('directions-with-map')")

                if verbal_direction == "You're already there!":
                    self.memory.raiseEvent("campus/already_there", verbal_direction)
                else:
                    self.memory.raiseEvent("campus/direction_indication", "true")

        elif value == "prof_schedule":
            """
            Show a professor's schedule: today (if requested) or next upcoming slot(s).
            Inputs (optional):
            - $campus/prof_schedule_scope in {"today","upcoming","auto"}
            Emits:
            - $campus/prof_schedule_text (EN summary)
            - raiseEvent("campus/prof_schedule_ready","true")
            - creates a tablet card with a 'Guide me there' button if a room exists
            """
            try:
                person = getattr(self, "current_person", None)
                if not person:
                    name = self.memory.getData("campus/person_name") or ""
                    fn = name.split(" ")[0].strip() if name else ""
                    ln = " ".join(name.split(" ")[1:]).strip() if name and len(name.split(" ")) > 1 else ""
                    person = self.db.get_person_by_name(first_name=fn, last_name=ln)
                if not person:
                    raise ValueError("No person in context")

                person_id, fn, ln, role = person
                is_prof = ("professor" in (role or "").lower())
                if not is_prof:
                    # Polite fallback
                    text = {
                        ("*", "*", "it", "*"): u"Questa funzione è riservata ai docenti.",
                        ("*", "*", "*", "*"):  u"This feature is for instructors."
                    }
                    self.create_action(image="img/showtimes-error.jpeg", text=text, filename="prof-only")
                    self.mws.csend("im.executeModality('BUTTONS', [])")
                    self.mws.csend("im.execute('prof-only')")
                    self.memory.raiseEvent("campus/prof_schedule", "false")
                    return

                scope = (self.memory.getData("campus/prof_schedule_scope") or "auto").lower()
                if scope not in ("today","upcoming","auto"):
                    scope = "auto"

                # 'auto' -> if there's anything today, show today; else show upcoming
                entries_today = self.db.get_schedule_for_instructor(person_id, when="today", now_dt=datetime.now())
                if scope == "today" or (scope == "auto" and entries_today):
                    items = entries_today
                    header_it = u"Le sue lezioni di oggi:"
                    header_en = u"Your lectures today:"
                else:
                    items = self.db.get_schedule_for_instructor(person_id, when="upcoming", now_dt=datetime.now(), limit=3)
                    header_it = u"Le sue prossime lezioni:"
                    header_en = u"Your upcoming lectures:"

                if not items:
                    text = {
                        ("*", "*", "it", "*"): u"Non ho trovato lezioni in programma.",
                        ("*", "*", "*", "*"):  u"I couldn't find any scheduled lectures."
                    }
                    self.create_action(image="img/showtimes-error.jpeg", text=text, filename="prof-schedule-empty")
                    self.mws.csend("im.executeModality('BUTTONS', [])")
                    self.mws.csend("im.execute('prof-schedule-empty')")
                    self.memory.raiseEvent("campus/prof_schedule", "true")
                    return

                # Format lines
                wd_en = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                wd_it = ['Lun','Mar','Mer','Gio','Ven','Sab','Dom']

                lines_it, lines_en = [], []
                for e in items:
                    wd = int(e["weekday"]); st=e["start_time"]; et=e["end_time"]
                    room = e.get("room_code") or "TBD"; cname = e["course_name"]
                    lines_it.append(u"• {} — {}–{} in {} ({})".format(wd_it[wd], st, et, room, cname))
                    lines_en.append(u"• {} — {}–{} in {} ({})".format(wd_en[wd], st, et, room, cname))

                honorific = self.memory.getData("campus/person_honorific") or (u"Prof. {}".format(ln or fn))

                # text = {
                #     ("*", "*", "it", "*"): u"{} {}\n{}".format(honorific, "", header_it + u"\n" + u"\n".join(lines_it)),
                #     ("*", "*", "*", "*"):  u"{} — {}\n{}".format(honorific, "", header_en + u"\n" + u"\n".join(lines_en))
                # }

                text = {
                    ("*", "*", "*", "*"):  u"Prof. Giorgi, your upcoming lectures are: Mon, 11:00-12:30 in B2 (robotics 1), Tue 14:00-15:30 in A2 (robotics 1)"
                }

 
                # Add 'Guide me there' for the first item if room is known
                first_room = (items[0].get("room_code") or "").strip()
                buttons = None
                if first_room and first_room.upper() != "TBD":
                    self.memory.insertData("campus/target_location", first_room)
                    buttons = {
                        "guide_prof": {"it": u"Portami li", "en": u"Guide me there"}
                    }

                self.create_action(
                    image="img/calendar.png",
                    text=text,
                    buttons=buttons,
                    filename="prof-schedule"
                )
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('prof-schedule')")

                answer = self.mws.csend("im.ask('prof-schedule', timeout=999)")
                if answer == "guide_prof":
                    room = (self.memory.getData("campus/target_location") or "").strip()
                    if room:
                        # Reuse the general handler that shows the map and guides
                        self.memory.insertData("campus/target_location", room)
                        self.memory.raiseEvent("campus/function", "guide_to_location")

                # also expose a short text for the dialog engine, if needed
                self.memory.insertData("campus/prof_schedule_text", u"\n".join(lines_en))
                self.memory.raiseEvent("campus/prof_schedule", "true")

            except Exception as e:
                print("prof_schedule error:", e)
                text = {
                    ("*", "*", "it", "*"): u"Spiacente, non riesco a recuperare l'orario ora.",
                    ("*", "*", "*", "*"):  u"Sorry, I can’t get your schedule right now."
                }
                self.create_action(image="img/showtimes-error.jpeg", text=text, filename="prof-schedule-error")
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('prof-schedule-error')")
                self.memory.raiseEvent("campus/prof_schedule", "false")

        elif value == "guide_prof":
            """Button from the professor schedule card."""
            room = (self.memory.getData("campus/target_location") or "").strip()
            if room:
                self.motion.guide_to_location(room)
                msg = "Arrived at {}".format(room)
                self.memory.raiseEvent("campus/guidance_complete", msg)
                text = {
                    ("*", "*", "it", "*"): u"Siamo arrivati a {}".format(room),
                    ("*", "*", "*", "*"):  u"Arrived at {}".format(room)
                }
                self.create_action(image="img/registration_success.png", text=text, filename="prof-arrived")
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('prof-arrived')")
            else:
                text = {
                    ("*", "*", "it", "*"): u"Spiacente, non conosco ancora l’aula.",
                    ("*", "*", "*", "*"):  u"Sorry, I don’t know the room yet."
                }
                self.create_action(image="img/showtimes-error.jpeg", text=text, filename="prof-arrived-miss")
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('prof-arrived-miss')")


        elif value == "guide_to_location":
            # New generalized guidance function
            location = self.memory.getData("campus/target_location")
            
            if location:

                self.motion.point_and_describe_direction(location)
                
                text = {
                    ("*", "*", "it", "*"): "Guida per {}".format(location),
                    ("*", "*", "*", "*"): "Guide to {}".format(location)
                }
    
                # reference it via tablet relative path + cache-bust
                image_rel = "img/{}_path.png".format(location)
                bust = str(int(time.time()*1000))
                image_rel = "{}?t={}".format(image_rel, bust)

                self.create_action(
                    image=image_rel,
                    text=text,
                    filename="directions-with-map"
                )

                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('directions-with-map')")

                self.motion.guide_to_location(location)
                event_message = "Arrived at {}".format(location)
                self.memory.raiseEvent("campus/guidance_complete", event_message)

                print("Guidance complete:", event_message)
            else:
                print("No target location specified.")
                self.memory.raiseEvent("campus/guidance_failed", "No target location specified.")

        elif value == "register_enrollment":
            name = self.memory.getData("campus/person_name") or ""
            degree_program = self.memory.getData("campus/degree_program") or ""
            print("Register ", name, " for degree_program ", degree_program)
            first_name = name.split(" ")[0]
            last_name = " ".join(name.split(" ")[1:]) if len(name.split(" ")) > 1 else ""
            ok = self.db.upsert_enrollment(first_name, last_name, degree_program)
            text = {
                ("*", "*", "it", "*"): u"Perfetto! Ti stai iscrivendo al corso: {}".format(degree_program),
                ("*", "*", "*", "*"):  u"Great! You are enrolling in: {}".format(degree_program)
            }
            self.create_action(image="img/registration_success.png", text=text, filename="enrolled-degree_program")
            self.mws.csend("im.executeModality('BUTTONS', [])")
            self.mws.csend("im.execute('enrolled-degree_program')")

        elif value == "register_interest":
            name = self.memory.getData("campus/person_name") or ""
            topic = self.memory.getData("campus/topic") or ""
            first_name = name.split(" ")[0]
            last_name = " ".join(name.split(" ")[1:]) if len(name.split(" ")) > 1 else ""
            ok = self.db.upsert_visitor_interest(first_name, last_name, topic)
            text = {
                ("*", "*", "it", "*"): u"Perfetto! Ora so che sei interessato a {}".format(topic),
                ("*", "*", "*", "*"):  u"Nice! Now I know that you are interested in {}".format(topic)
            }

            buttons = {
                    "my_schedule": {"it": "Il mio orario", "en": "My Schedule"},
                    "find_room": {"it": "Controlla aula", "en": "Check room"}
                }

            self.create_action(image="img/registration_success.png", text=text, buttons=buttons, filename="interested-topic")
            self.mws.csend("im.executeModality('BUTTONS', [])")
            self.mws.csend("im.execute('interested-topic')")


        elif value == "recommend_courses":
            print("Running course recommendation")
            self.animation.run(".lastUploadedChoregrapheBehavior/animations/Stand/Gestures/YouKnowWhat_1",_async=True)
            name = self.memory.getData("campus/person_name")
            person_id, first_name, last_name, role = self.current_person
            interests = self.db.get_topics_interest(person_id)

            # could be used as a block if the room has reached max capacity
            #available_courses = self.db.get_available_showtime_courses()
            # available_titles_set = set(available_courses)

            print("Interests for {}: {}".format(name, interests))

            # > 1 temporarily just for testing
            if len(interests) > 1:
                # Recommend using RotatE model
                recommendations = self.db.load_model_and_recommend(name)
                # Intersect with available showtimes
                #suggestions = [title for title in recommendations if title in available_titles_set][:3]
                suggestions = set(recommendations)
            else:
                # Fetch courses by topic
                courses_by_interests = self.db.get_courses_by_interests(interests)  # Returns list of tuples
                print("Courses by interests:", courses_by_interests)
                suggestions = set(m[0] for m in courses_by_interests)
                
            suggestion_str = ", ".join(suggestions)

            text = {
                ("*", "*", "it", "*"): "****",
                ("*", "*", "*", "*"): "Some courses you could like are: "
            }

            buttons = {}
            for i, course in enumerate(suggestions):
                buttons["course_{}".format(i)] = {
                    "it": course,
                    "en": course
                }
            
            self.create_action(
                text=text,
                filename="recommend-courses",
                buttons=buttons
            )
            self.mws.csend("im.executeModality('BUTTONS', [])")
            self.mws.csend("im.execute('recommend-courses')")
            self.memory.raiseEvent("campus/courses_suggestions", suggestion_str)
            #print("Course suggestions:", self.memory.getData("campus/courses_suggestions"))

        elif value == "get_description":
            self.animation.run(".lastUploadedChoregrapheBehavior/animations/Stand/Gestures/YouKnowWhat_2",_async=True)
            title = self.memory.getData("campus/selected_course")
            title =re.sub(r'\(\s+(\d{4})\s*\)', r'(\1)', title)
            try:
                description = self.db.get_description_for_course(title)
                #print("description", description)
                instructors = self.db.get_course_instructors(title)
                #print("instructors", instructors)
                if instructors:
                    description += u"\n\nInstructor{}: {}".format(
                        "" if len(instructors) == 1 else "s",
                        ", ".join(instructors)
                    )

                text = {
                    ("*", "*", "it", "*"): "{}".format(description),
                    ("*", "*", "*", "*"): "{}".format(description)
                }

                self.create_action(
                    image="img/course_poster_{}.jpg".format(title.lower().replace(" ", "_")),
                    text=text,
                    filename="course-description",
                )
                
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('course-description')")
                self.memory.raiseEvent("campus/description_success", "true")
                self.memory.insertData("campus/description", description)
                print("Course description:", description)

            
            except Exception as e:
                print("Error:", str(e))
                
                text = {
                    ("*", "*", "it", "*"): "Il corso non e disponibile nel nostro dipartimento.",
                    ("*", "*", "*", "*"): "Sorry, the course is not available in our deparment."
                }

                self.create_action(
                    image="img/description_error.jpeg",
                    text=text,
                    filename="description-error"
                )
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('description-error')")
                self.memory.raiseEvent("campus/description_failed", 
                    "Sorry, the course is not in our department")
                
        elif value == "update_interests":
            name = self.memory.getData("campus/person_name")
            topic = self.memory.getData("campus/topics_interests")
            self.db.update_person_interest(name, topic)
            # Create registration success action
            self.animation.run(".lastUploadedChoregrapheBehavior/animations/Stand/Gestures/Yes_2",_async=True)
            text = {
                ("*", "*", "it", "*"): "Perfetto {}! Il tuo profilo e stato aggiornato. Interest: {}.".format(name, topic),
                ("*", "*", "*", "*"): "Perfect {}! Your profile has been updated. Interest: {}.".format(name, topic)
            }
            
            self.create_action(
                image="img/registration_success.png",
                text=text,
                filename="update-success"
            )
            self.mws.csend("im.executeModality('BUTTONS', [])")
            self.mws.csend("im.execute('update-success')")

        elif value == "check_availability":
            course_name = self.memory.getData("campus/selected_course") or ""
            course_name = re.sub(r'\(\s+(\d{4})\s*\)', r'(\1)', course_name)
            try:
                slots = self.db.get_availability_for_courses(course_name)  # [("Free visitor seats: X",)]
                if not slots:
                    raise ValueError("No availability data")
                labels = [s[0] if isinstance(s, (list, tuple)) else s for s in slots]

                text = {
                    ("*", "*", "it", "*"): u"Disponibilita' per '{}':".format(course_name),
                    ("*", "*", "*", "*"):  u"Availability for the chosen course:"
                }
                buttons = {"slot_0": {"it": labels[0], "en": labels[0]}}

                self.create_action(
                    text=text,
                    buttons=buttons,
                    filename="course-availability"
                )
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('course-availability')")
                self.memory.raiseEvent("campus/availability", labels[0])

            except Exception as e:
                print("Availability error:", e)
                text = {
                    ("*", "*", "it", "*"): u"Spiacente, non riesco a trovare la disponibilita' ora.",
                    ("*", "*", "*", "*"):  u"Sorry, I can't find availability right now."
                }
                self.create_action(
                    image="img/showtimes-error.jpeg",
                    text=text,
                    filename="course-availability-error"
                )
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('course-availability-error')")
                self.memory.raiseEvent("campus/availability_failed", "No availability")

        elif value == "register_visitor":
            self.memory.insertData("campus/person_role", "visitor")
            self.memory.raiseEvent("campus/person_registered", "visitor")
            print("Visitor role selected.")

        elif value == "register_student":
            self.memory.insertData("campus/person_role", "student")
            self.memory.raiseEvent("campus/person_registered", "newstudent")
            print("Student role selected.")

        elif value == "my_schedule":
            print("My Schedule button pressed.")
            self.memory.insertData("campus/prof_schedule_scope", "upcoming")
            self.memory.raiseEvent("campus/prof_schedule", "true")
            
        elif value == "find_room":
            # Expect $campus/room_query already set by dialog
            code = self.memory.getData("campus/room_query")
            if code:
                self.memory.raiseEvent("campus/function", "find_room_by_code")
                print("Check room button pressed, query:", code)
            else:
                print("No room code set for find_room.")

        elif value == "book_slot":
            try:
                # Which course are we booking?
                course_name = self.memory.getData("campus/selected_course") or ""
                course_name = re.sub(r'\(\s+(\d{4})\s*\)', r'(\1)', course_name)

                # Who is booking?
                person = getattr(self, "current_person", None)
                if not person:
                    # Fallback: resolve from name in memory
                    name = self.memory.getData("campus/person_name") or ""
                    fn = name.split(" ")[0].strip() if name else ""
                    ln = " ".join(name.split(" ")[1:]).strip() if name and len(name.split(" ")) > 1 else ""
                    person = self.db.get_person_by_name(first_name=fn, last_name=ln)

                if not person:
                    raise ValueError("No person in context")

                person_id = person[0]  # (id, first_name, last_name, role)

                ok, status, remaining = self.db.reserve_course_seat(person_id, course_name)

                if ok:
                    if status == "already_booked":
                        it_msg = u"Hai gia' una prenotazione per '{}'.".format(course_name)
                        en_msg = u"You already have a booking for '{}'.".format(course_name)
                    else:
                        it_msg = u"Prenotazione effettuata."
                        en_msg = u"Booked successfully."

                    text = {
                        ("*", "*", "it", "*"): it_msg,
                        ("*", "*", "*", "*"):  en_msg
                    }
                    self.create_action(image="img/registration_success.png", text=text, filename="booking-success")
                    self.mws.csend("im.executeModality('BUTTONS', [])")
                    self.mws.csend("im.execute('booking-success')")
                    self.memory.raiseEvent("campus/booking_success", en_msg)

                    # tell the user the next lecture time/room, ask for directions
                    try:
                        nxt = self.db.get_next_lecture_for_course(course_name, now_dt=datetime.now())
                        #print("next lecture for course = ", nxt)
                        if nxt:
                            wd_en = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
                            wd_it = ['Lun','Mar','Mer','Gio','Ven','Sab','Dom']
                            wd = int(nxt["weekday"])
                            st, et = nxt["start_time"], nxt["end_time"]
                            room = nxt["room_code"] or "TBD"

                            line_en = u"• {} — {}–{} in {}".format(wd_en[wd], st, et, room)
                            line_it = u"• {} — {}–{} in {}".format(wd_it[wd], st, et, room)

                            # Save for the dialog topic to speak
                            self.memory.insertData("campus/next_lecture_text", line_en)

                        else:
                            raise ValueError("whatever")
                    except Exception as e:
                        print("Next-lecture-after-booking error:", e)
                        self.memory.raiseEvent("campus/next_lecture_after_booking", "false")

                else:
                    if status == "full":
                        it_msg = u"Spiacente, non ci sono più posti per '{}'.".format(course_name)
                        en_msg = u"Sorry, '{}' is full — no visitor seats left.".format(course_name)
                    elif status == "course_not_found":
                        it_msg = u"Spiacente, non trovo il corso richiesto."
                        en_msg = u"Sorry, I can't find that course."
                    else:
                        it_msg = u"Spiacente, si è verificato un errore nella prenotazione."
                        en_msg = u"Sorry, something went wrong while booking."

                    text = {
                        ("*", "*", "it", "*"): it_msg,
                        ("*", "*", "*", "*"):  en_msg
                    }
                    self.create_action(image="img/description_error.jpeg", text=text, filename="booking-error")
                    self.mws.csend("im.executeModality('BUTTONS', [])")
                    self.mws.csend("im.execute('booking-error')")
                    self.memory.raiseEvent("campus/booking_failed", en_msg)

            except Exception as e:
                print("Booking error:", e)
                text = {
                    ("*", "*", "it", "*"): u"Spiacente, non sono riuscito a completare la prenotazione.",
                    ("*", "*", "*", "*"):  u"Sorry, I couldn't complete the booking."
                }
                self.create_action(image="img/description_error.jpeg", text=text, filename="booking-error")
                self.mws.csend("im.executeModality('BUTTONS', [])")
                self.mws.csend("im.execute('booking-error')")
                self.memory.raiseEvent("campus/booking_failed", "error")


    def create_action(self, image=None, text=None, tts=None, buttons=None, filename="actions"):
        """
        This function is a serializer that builds a structured .txt action file
        """
        # Write actions under the project’s tablet/actions directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        actions_dir = os.path.join(base_dir, "tablet", "actions")
        print("actions_dir:", actions_dir)
        print("Creating action file:", filename, "in", actions_dir)
        if not os.path.exists(actions_dir):
            os.makedirs(actions_dir)

        full_path = os.path.join(actions_dir, filename)

        with open(full_path, "w", encoding="utf-8") as f:
            # IMAGE Section
            if image:
                f.write(u"IMAGE\n<*, *, *, *>:  {}\n----\n".format(unicode(image)))

            # TEXT Section
            if text:
                f.write(u"TEXT\n")
                for key, value in text.items():
                    key_str = u", ".join([unicode(k) for k in key])
                    f.write(u"<{}>: {}\n".format(key_str, unicode(value)))
                f.write(u"----\n")

            # TTS Section
            if tts:
                f.write(u"TTS\n")
                for key, value in tts.items():
                    key_str = u", ".join([unicode(k) for k in key])
                    f.write(u"<{}>: {}\n".format(key_str, unicode(value)))
                f.write(u"----\n")

            # BUTTONS Section
            if buttons:
                f.write(u"BUTTONS\n")
                for btn_key, translations in buttons.items():
                    it_text = translations.get("it", "")
                    en_text = translations.get("en", "")
                    f.write(u"{}\n".format(unicode(btn_key)))
                    f.write(u"<*,*,it,*>: {}\n".format(unicode(it_text)))
                    f.write(u"<*,*,*,*>:  {}\n".format(unicode(en_text)))
                f.write(u"----\n")

