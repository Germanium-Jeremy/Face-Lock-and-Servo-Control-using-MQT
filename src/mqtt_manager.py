import json
import time
import socket
from typing import Optional
try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None
    print("WARNING: paho-mqtt not installed. Run: pip install paho-mqtt")

class MQTTManager:
    def __init__(self, broker: str = "157.173.101.159", port: int = 1883, team_id: str = "default_team"):
        self.broker = broker
        self.port = port
        self.team_id = team_id
        self.client: Optional[mqtt.Client] = None
        
        if mqtt:
            self.client = mqtt.Client()
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            
            try:
                self.client.connect(self.broker, self.port, 60)
                self.client.loop_start()
            except Exception as e:
                print(f"[MQTT] Failed to connect: {e}")
        else:
            print("[MQTT] Client not initialized due to missing dependency")

    def _on_connect(self, client, userdata, flags, rc):
        print(f"[MQTT] Connected with result code {rc}")
        if rc == 0:
            self.publish_heartbeat()

    def _on_disconnect(self, client, userdata, rc):
        print(f"[MQTT] Disconnected with result code {rc}")

    def publish_movement(self, status: str, confidence: float = 1.0):
        if not self.client:
            return
        
        topic = f"vision/{self.team_id}/movement"
        payload = {
            "status": status,
            "confidence": confidence,
            "timestamp": int(time.time())
        }
        try:
            self.client.publish(topic, json.dumps(payload))
            # print(f"[MQTT] Published to {topic}: {payload}")
        except Exception as e:
            print(f"[MQTT] Failed to publish movement: {e}")

    def publish_heartbeat(self):
        if not self.client:
            return

        topic = f"vision/{self.team_id}/heartbeat"
        payload = {
            "node": "pc",
            "status": "ONLINE",
            "timestamp": int(time.time())
        }
        try:
            self.client.publish(topic, json.dumps(payload))
            print(f"[MQTT] Published heartbeat to {topic}")
        except Exception as e:
            print(f"[MQTT] Failed to publish heartbeat: {e}")

    def stop(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
