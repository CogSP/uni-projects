# -*- coding: utf-8 -*-
from core.map.map import Map
import math
import os


class MotionManager(object):
    def __init__(self, motion_proxy, tracker=None, asset_dir=None):
        self.motion = motion_proxy
        self.tracker = tracker
        self.campus_map = Map()
        self.current_orientation=0
        self.bathroom_busy=False
        self.asset_dir = asset_dir  # e.g., <project>/tablet/img

    # ───────── Face look / tracking helpers ─────────
    def enable_face_look(self, mode="Head"):
        """Start face tracking with ALTracker if available; otherwise center head."""
        try:
            if self.tracker is not None:
                # Only track with the head so navigation is unaffected.
                self.tracker.setMode(mode)           # "Head" or "WholeBody"
                self.tracker.setEffector("None")     # ensure only head is used
                # Face target requires an approximate face size (meters)
                if "Face" not in self.tracker.getRegisteredTargets():
                    self.tracker.registerTarget("Face", 0.12)
                self.tracker.track("Face")
                return True
        except Exception as e:
            print("[MotionManager] enable_face_look fallback:", e)
        # Fallback: gently tilt head toward a human in front
        self.quick_look_front()
        return False

    def disable_face_look(self):
        """Stop face tracking if it was enabled."""
        try:
            if self.tracker is not None:
                self.tracker.stopTracker()
                self.tracker.unregisterAllTargets()
        except Exception as e:
            print("[MotionManager] disable_face_look:", e)

    def quick_look_front(self, yaw=0.0, pitch=-0.15, speed=0.2):
        """Simple head pose to 'look at' a person presumed in front of Pepper."""
        try:
            self.motion.setAngles(["HeadYaw","HeadPitch"], [yaw, pitch], speed)
        except Exception as e:
            print("[MotionManager] quick_look_front:", e)


    def point_and_describe_direction(self, target_location):
        """Point to a location and give verbal directions"""
        try:
            # Determine target node
            target_node = target_location
                
            if not target_node or target_node not in self.campus_map.nodes:
                print("Unknown location: {}".format(target_location))
                return "I'm not sure where that is."
            
            filename = "{}_path.png".format(target_node)
            save_path = filename
            path=self.campus_map.draw_map_with_path(target_node, bathroom_busy=self.bathroom_busy, save_path=save_path)
            
            if not path or len(path) < 2:
                return "You're already there!"
            
            # Get next node in path for pointing direction
            next_node = path[1]
            
            # Calculate pointing angle for robot
            current_pos = self.campus_map.nodes[self.campus_map.current_position]
            next_pos = self.campus_map.nodes[next_node]
            
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            target_angle = math.atan2(dy, dx)
            if hasattr(self, 'current_orientation'):
                    relative_angle = target_angle - self.current_orientation
                    # Normalize angle to [-pi, pi]
                    relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
            else:
                # If no orientation tracking, use absolute angle
                relative_angle = target_angle
            # Point with robot
            self.motion.moveTo(0.0, 0.0, relative_angle)
            names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
            
            initial_angles = self.motion.getAngles(names, True)

            # initial movement for the greeting
            angles = [-0.5, -0.3, 1.0, 0.5]
            self.motion.angleInterpolation(names, angles, [1.0]*4, True)

            # go back to initial position
            self.motion.angleInterpolation(names, initial_angles, [1.0]*4, True)
            
            
            # Add landmarks if available
            landmarks = self._get_landmarks_on_path(path)
            if landmarks:
                verbal_direction = "{}".format(landmarks)
                
            # Hold pointing position briefly
            import time
            time.sleep(2)
            
            # Return to neutral and face customer
            self.motion.moveTo(0.0, 0.0, -relative_angle)
            
            return verbal_direction
            
        except Exception as e:
            print("Pointing failed:", e)
            return "I had trouble pointing in that direction."
    
    def _get_landmarks_on_path(self, path):
        """Generate landmark descriptions for the path"""
        landmarks = []
        
        for i, node in enumerate(path):
            if 'lobby' in node and node != path[-1]:
                landmarks.append("you'll pass the lobby")

            # could add other landmarks here

        return ", ".join(landmarks) if landmarks else ""

    def guide_to_location(self, location):
        """Guide to location using graph-based pathfinding"""
        try:
            target_node = location
            if not target_node or target_node not in self.campus_map.nodes:
                print("Unknown location: {}".format(location))
                return
            
            print("[Motion] Guiding to {}".format(location))
                
            # Find shortest path
            path = self.campus_map.find_shortest_path(target_node)
            
            if len(path) < 2:
                print("Already there")
                return
            
            if not path:
                print("No path found to {}".format(target_node))
                return
                
            print("Path found: {}".format(" -> ".join(path)))
            
            # Execute movement along path
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                
                # Calculate movement vector
                current_pos = self.campus_map.nodes[current_node]
                next_pos = self.campus_map.nodes[next_node]
                
                dx = next_pos[0] - current_pos[0]
                dy = next_pos[1] - current_pos[1]
                distance = math.sqrt(dx**2 + dy**2)
                target_angle = math.atan2(dy, dx)
                
                if hasattr(self, 'current_orientation'):
                    relative_angle = target_angle - self.current_orientation
                    # Normalize angle to [-pi, pi]
                    relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
                else:
                    # If no orientation tracking, use absolute angle
                    relative_angle = target_angle
            
                print("Moving from {} to {} (distance: {:.2f}m, target_angle: {:.2f} rad, relative_angle: {:.2f} rad)".format(
                    current_node, next_node, distance, target_angle, relative_angle))
                
                # Use relative movements
                # First rotate to face direction (relative turn)
                if abs(relative_angle) > 0.01:  # Only rotate if significant angle difference
                    self.motion.moveTo(0.0, 0.0, relative_angle)
                    # Update current orientation after rotation
                    if hasattr(self, 'current_orientation'):
                        self.current_orientation = target_angle

                # Then move forward
                self.motion.moveTo(distance, 0.0, 0.0)
                
                # Update current position
                self.campus_map.current_position = next_node

            # Face user at destination
            self.face_person()

        except Exception as e:
            print("Guidance failed:", e)
            
    def face_person(self):
        """Turn to face the person (assumes person is behind robot after guidance)."""
        try:
            # 1) Pause tracking so it doesn't fight the turn
            self.disable_face_look()
        except Exception as e:
            print("face_person: disable_face_look:", e)

        try:
            # 2) Relative 180° turn to face the follower
            self.motion.moveTo(0.0, 0.0, -0.40)

            # Keep our internal orientation consistent
            if hasattr(self, 'current_orientation'):
                self.current_orientation = math.atan2(
                    math.sin(self.current_orientation + math.pi),
                    math.cos(self.current_orientation + math.pi)
                )

            # 3) Neutral, friendly posture (slight head down helps eye contact with standing user)
            self.motion.angleInterpolation(["HeadYaw", "HeadPitch"], [0.0, -0.15], [0.8, 0.8], True)

            # Right arm back to neutral (optional)
            neutral_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
            neutral_angles = [1.5, -0.15, 0.0, 0.0]
            self.motion.angleInterpolation(neutral_names, neutral_angles, [1.0]*4, True)

        except Exception as e:
            print("Failed to face customer:", e)
        finally:
            # 4) Re-enable face tracking to keep looking at the user
            try:
                self.enable_face_look(mode="Head")
            except Exception as e:
                print("face_person: enable_face_look:", e)

