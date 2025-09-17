#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Campus Assistant — entrypoint script for a Pepper/NAOqi + MODIM project.


What this script does (high level):
1) Parses robot connection params (IP/port) and connects to NAOqi.
2) Grabs core NAOqi services (ALMemory, ALMotion, ALAnimationPlayer, ALDialog).
3) Initializes local project components:
- SQLite DB for data
- Tablet / MODIM web socket client (ModimWSClient)
- Motion manager wrapper around ALMotion (MotionManager)
- Main business logic / dialog handlers (CampusAssistant)
4) Registers the CampusAssistant as a NAOqi service so other modules can call it.
5) Loads and activates the main dialog topic (.top) for ALDialog.
6) Subscribes to ALMemory events to handle function and tablet commands.
7) Simulates a sonar reading and optionally moves Pepper closer to a detected human.
8) Builds a knowledge graph / dataset (KGBuilder)
9) Keeps the app alive until CTRL+C, then performs a clean shutdown and DB cleanup.


Notes:
- Requires environment variable MODIM_HOME to import the ws_client GUI client.
- Expects a NAOqi instance reachable at the given IP/port.
- Expects data files in ./data and topic/main.top in the project root.
"""

import qi
import argparse
import sys
import os
import time
import signal
import math


# Project-specific modules (must exist in ./core)
from core.motion.motion_manager import MotionManager
from core.database.database import Database
from core.assistant.assistant import Assistant
from core.kg.kg_builder import KGBuilder


# ──────────────────────────────────────────────────────────────────────────────
# MODIM GUI client import
# The GUI/ws client is shipped as part of the MODIM toolkit. The code expects
# that the environment variable MODIM_HOME points to the MODIM repository root.
# If MODIM_HOME is not set or invalid, we exit with a helpful message.
# ──────────────────────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, os.getenv('MODIM_HOME')+'/src/GUI')
except Exception as e:
    print("Please set MODIM_HOME environment variable to MODIM folder.")
    sys.exit(1)

# MODIM GUI client import
from ws_client import *


def init_client():
    """Initialize the MODIM interaction manager.    
    Called by ModimWSClient.run_interaction(). The global `im` is provided by
    the MODIM ws_client module and is used to manage interactions/scenes.
    """
    im.init()

def delete_db():
    """Delete the local SQLite database file if it exists.
    Useful during development to reset state on shutdown. Adjust path as needed
    if you relocate the DB.
    """
    DB_PATH = "data/campus.db"

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("[INFO] File deleted.")
    else:
        print("[INFO] The file does not exist or is already deleted.")



def main():
    """Program entrypoint.
        - Parses CLI args `--pip` and mandatory `--pport`.
        - Establishes a qi.Application connection to NAOqi.
        - Initializes services, DB, tablet GUI, motion manager, and assistant logic.
        - Registers the assistant as a NAOqi service and wires ALDialog topics.
        - Subscribes to ALMemory events exposed by the project.
        - Demonstrates a simple sonar-based approach to human detection/motion.
        - Builds a recommender dataset / knowledge graph (offline step).
        - Keeps running until KeyboardInterrupt, then performs graceful teardown.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pip", type=str, default=os.environ.get('PEPPER_IP', '127.0.0.1'), help="Robot IP")
    parser.add_argument("--pport", type=int, required=True, help="Naoqi port")
    args = parser.parse_args()

    project_path = os.path.dirname(os.path.abspath(__file__))

    try:
        connection_url = "tcp://{}:{}".format(args.pip, args.pport)
        app = qi.Application(["Campus Assistant", "--qi-url=" + connection_url])
    except RuntimeError:
        print("Could not connect to NAOqi.")
        sys.exit(1)

    app.start()
    session = app.session
    print("Successfully connected to the robot at {}:{}".format(args.pip, args.pport))

    ALMemory = session.service("ALMemory")
    ALMotion = session.service("ALMotion")

    # TODO: to check
    # Turn stiffness ON for the whole body
    ALMotion.setStiffnesses("Body", 1.0)

    # Optional: face tracker (best-effort)
    try:
            ALTracker = session.service("ALTracker")
    except Exception as _:
        ALTracker = None
 

    ALAnimation = session.service("ALAnimationPlayer")
    

    db = Database(project_path)
    db.initialize_database()

    mws = ModimWSClient()
    path = os.path.join(project_path,"tablet/placeholder/another")
    
    mws.setDemoPathAuto(path)
    mws.run_interaction(init_client)
    mws.cconnect()
    
    # try:
    #     mws.csend("im.setDemoPath('{}')".format(tablet_dir.replace("\\", "\\\\")))
    #     mws.csend("console.log('DEMO_PATH runtime:', im.demo_path)")
    # except Exception:
    #     raise RuntimeError("Failed to set MODIM demo path. Check MODIM_HOME environment variable.")

    mws.cconnect()
    mws.run_interaction(init_client)


    # Tell MotionManager where to drop dynamic images (tablet/img)
    motion = MotionManager(
        ALMotion,
        tracker=ALTracker,
        asset_dir=os.path.join(path, "img")
    )
    try:
        campus_assistant = Assistant(ALAnimation, ALMemory, db, mws, motion)
    except Exception as e:
        print("Failed to initialize Campus Assistant:", e)
        sys.exit(1)

    try:
        service_id = session.registerService("CampusAssistantApp", campus_assistant)
        print("Campus Assistant service registration, with ID:", service_id)
    except Exception as e:
        print("Failed Campus Assistant service registration:", e)
        sys.exit(1)


    # Dialog Topic
    ALDialog = session.service("ALDialog") # get the dialog service
    try: 
        topic_path = os.path.join(project_path, "topic", "main.top")
        topic_name = ALDialog.loadTopic(topic_path.encode('utf-8')) # load the topic file, telling NAOqi to parse and register it
        ALDialog.activateTopic(topic_name)
        ALDialog.subscribe("campus_assistant") # subscribe to the dialog system. From now on, if the user says something that matches the topic, it will be handled.
    except Exception as e:
        print("Failed to load or activate dialog topic:", e)
        sys.exit(1)


    # Subscribe to ALMemory events
    function_sub = ALMemory.subscriber("campus/function")
    function_sub.signal.connect(campus_assistant.handle_function)

    # tablet_sub = ALMemory.subscriber("campus/tablet")
    # tablet_sub.signal.connect(campus_assistant.handle_tablet)    
    

    # Simulated human position (relative to robot)
    human_position = (1.0, 0.0)  # 1 meter in front
    distance = math.sqrt(human_position[0]**2 + human_position[1]**2)
    sonar_key = 'Device/SubDeviceList/Platform/Front/Sonar/Sensor/Value'
    ALMemory.insertData(sonar_key, distance)
    
    # Read back the value
    sonar_value = ALMemory.getData(sonar_key)
    print("Simulated sonar value: {}".format(sonar_value))
    if sonar_value < 2.0:
        print("Human detected.")
        try:
            ALMotion.moveTo(human_position[0], human_position[1], 0.0)
            # Immediately try to look at the person
            motion.enable_face_look(mode="Head")   # head-only face tracking
            ALDialog.forceInput("hi")
            print("Robot moved towards human.")
        except Exception as e:
            print("Motion error:", e)
    else:
        print("No human detected.")

    builder = KGBuilder(
        out_train="data/train.txt",
        out_test="data/test.txt",
        out_kg="data/kg.txt"
        )
    builder.build_kg()
    
    print("Campus Assistant is running...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        # Stop tracking cleanly
        try:
            motion.disable_face_look()
        except Exception:
            pass
        ALDialog.unsubscribe("campus_assistant")
        ALDialog.deactivateTopic(topic_name)
        ALDialog.unloadTopic(topic_name)
        print("Dialog topic stopped and unloaded.")
        session.unregisterService(service_id)
        print("CampusAssistantApp service unregistered.")
        
        app.stop()
        print("Application stopped.")
        

if __name__ == "__main__":
    main()
