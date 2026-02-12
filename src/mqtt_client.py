"""
MQTT client for publishing face tracking data.
Communicates with backend via MQTT broker.
"""

from __future__ import annotations
import json
import time
from typing import Optional
import paho.mqtt.client as mqtt


class FaceTrackingMQTTClient:
    """
    MQTT client for publishing face tracking movement data.
    Publishes locked face movements and heartbeat status.
    """
    
    def __init__(self, broker_host: str = "157.173.101.159", broker_port: int = 1883, team_id: str = "team1"):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.team_id = team_id
        self.client = mqtt.Client()
        
        # MQTT topics
        self.movement_topic = f"vision/{team_id}/movement"
        self.heartbeat_topic = f"vision/{team_id}/heartbeat"
        
        # Client state
        self.connected = False
        self.last_heartbeat = 0
        self.heartbeat_interval = 30  # seconds
        
        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            print(f"[MQTT] Connecting to broker {self.broker_host}:{self.broker_port}")
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = time.time() + 10
            while not self.connected and time.time() < timeout:
                time.sleep(0.1)
            
            if self.connected:
                print(f"[MQTT] Connected successfully")
                self._send_heartbeat()
                return True
            else:
                print(f"[MQTT] Connection timeout")
                return False
                
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            print(f"[MQTT] Disconnected")
    
    def publish_movement(self, status: str, confidence: float = 0.0) -> bool:
        """
        Publish movement status to MQTT.
        
        Args:
            status: Movement status (MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN, CENTERED, NO_FACE_LOCKED)
            confidence: Confidence score (0.0-1.0)
        """
        if not self.connected:
            print(f"[MQTT] Not connected, cannot publish movement")
            return False
        
        payload = {
            "status": status,
            "confidence": confidence,
            "timestamp": int(time.time())
        }
        
        try:
            result = self.client.publish(self.movement_topic, json.dumps(payload), qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT] Published movement: {status} (confidence: {confidence:.2f})")
                return True
            else:
                print(f"[MQTT] Failed to publish movement: {result.rc}")
                return False
                
        except Exception as e:
            print(f"[MQTT] Error publishing movement: {e}")
            return False
    
    def update_heartbeat(self):
        """Send heartbeat if needed."""
        current_time = time.time()
        if current_time - self.last_heartbeat > self.heartbeat_interval:
            self._send_heartbeat()
    
    def _send_heartbeat(self):
        """Send heartbeat message."""
        payload = {
            "node": "pc",
            "status": "ONLINE",
            "timestamp": int(time.time())
        }
        
        try:
            result = self.client.publish(self.heartbeat_topic, json.dumps(payload), qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"[MQTT] Sent heartbeat")
                self.last_heartbeat = time.time()
            else:
                print(f"[MQTT] Failed to send heartbeat: {result.rc}")
                
        except Exception as e:
            print(f"[MQTT] Error sending heartbeat: {e}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.connected = True
            print(f"[MQTT] Connected to broker")
        else:
            self.connected = False
            print(f"[MQTT] Connection failed with code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        self.connected = False
        print(f"[MQTT] Disconnected with code {rc}")


# Movement status constants
class MovementStatus:
    NO_FACE_LOCKED = "NO_FACE_LOCKED"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    CENTERED = "CENTERED"
