"""
WebSocket server for real-time face tracking dashboard.
Receives MQTT messages and forwards them to connected web clients.
"""

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import json
import time
import threading
import paho.mqtt.client as mqtt
from pathlib import Path


class DashboardServer:
    """
    WebSocket server that bridges MQTT messages to web dashboard.
    """
    
    def __init__(self, team_id: str = "team1", mqtt_host: str = "157.173.101.159", 
                 mqtt_port: int = 1883, web_port: int = 9002):
        self.team_id = team_id
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.web_port = web_port
        
        # Flask and SocketIO setup
        self.app = Flask(__name__, 
                       template_folder=str(Path(__file__).parent.parent / 'web'),
                       static_folder=str(Path(__file__).parent.parent / 'web'))
        self.app.config['SECRET_KEY'] = 'face_tracking_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_connected = False
        
        # MQTT topics
        self.movement_topic = f"vision/{team_id}/movement"
        self.heartbeat_topic = f"vision/{team_id}/heartbeat"
        
        # Setup routes and callbacks
        self._setup_routes()
        self._setup_socketio_events()
        self._setup_mqtt_callbacks()
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return send_from_directory(Path(__file__).parent.parent / 'web', 'dashboard.html')
        
        @self.app.route('/dashboard')
        def dashboard():
            return send_from_directory(Path(__file__).parent.parent / 'web', 'dashboard.html')
    
    def _setup_socketio_events(self):
        """Setup SocketIO event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"[WebSocket] Client connected")
            emit('status', {'message': 'Connected to face tracking server'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"[WebSocket] Client disconnected")
        
        @self.socketio.on('request_status')
        def handle_status_request():
            # Send current status to newly connected client
            emit('status', {'message': 'Server running'})
    
    def _setup_mqtt_callbacks(self):
        """Setup MQTT callbacks."""
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                self.mqtt_connected = True
                print(f"[MQTT] Connected to broker")
                # Subscribe to topics
                client.subscribe(self.movement_topic)
                client.subscribe(self.heartbeat_topic)
                print(f"[MQTT] Subscribed to topics: {self.movement_topic}, {self.heartbeat_topic}")
            else:
                self.mqtt_connected = False
                print(f"[MQTT] Connection failed with code {rc}")
        
        def on_disconnect(client, userdata, rc):
            self.mqtt_connected = False
            print(f"[MQTT] Disconnected with code {rc}")
        
        def on_message(client, userdata, msg):
            """Handle incoming MQTT messages."""
            try:
                payload = json.loads(msg.payload.decode())
                
                if msg.topic == self.movement_topic:
                    # Forward movement data to web clients
                    self.socketio.emit('movement_update', payload)
                    print(f"[MQTT] Movement update: {payload}")
                
                elif msg.topic == self.heartbeat_topic:
                    # Forward heartbeat data to web clients
                    self.socketio.emit('heartbeat_update', payload)
                    print(f"[MQTT] Heartbeat update: {payload}")
                
            except json.JSONDecodeError as e:
                print(f"[MQTT] JSON decode error: {e}")
            except Exception as e:
                print(f"[MQTT] Message handling error: {e}")
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_disconnect = on_disconnect
        self.mqtt_client.on_message = on_message
    
    def connect_mqtt(self):
        """Connect to MQTT broker."""
        try:
            print(f"[MQTT] Connecting to {self.mqtt_host}:{self.mqtt_port}")
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            return True
        except Exception as e:
            print(f"[MQTT] Connection error: {e}")
            return False
    
    def run(self, debug=False):
        """Start the web server."""
        print(f"[Server] Starting dashboard server on port {self.web_port}")
        
        # Connect to MQTT in a separate thread
        mqtt_thread = threading.Thread(target=self.connect_mqtt)
        mqtt_thread.daemon = True
        mqtt_thread.start()
        
        # Start web server
        self.socketio.run(self.app, host='0.0.0.0', port=self.web_port, debug=debug)


def main():
    """Main entry point for dashboard server."""
    server = DashboardServer(team_id="team1")
    server.run(debug=False)


if __name__ == "__main__":
    main()
