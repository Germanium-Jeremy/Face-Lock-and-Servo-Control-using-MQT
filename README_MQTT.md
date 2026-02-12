# Face Recognition System with MQTT Camera Tracking

A modular face recognition system with real-time camera tracking via MQTT communication.

## Architecture Overview

```
┌─────────────────┐    MQTT     ┌─────────────────┐    WebSocket    ┌─────────────────┐
│   PC Vision    │ ──────────► │   MQTT Broker   │ ──────────────► │  Web Dashboard  │
│   System       │             │  (157.173.101. │                │                 │
│                │             │     159:1883)   │                │                 │
└─────────────────┘             └─────────────────┘                └─────────────────┘
        │                              │                                 │
        │ MQTT                         │ MQTT                             │ WebSocket
        ▼                              ▼                                 ▼
┌─────────────────┐             ┌─────────────────┐                ┌─────────────────┐
│   ESP8266      │             │   Backend       │                │   Browser       │
│   Servo Control │             │   (Optional)    │                │   Client        │
└─────────────────┘             └─────────────────┘                └─────────────────┘
```

## Features

### PC Vision System
- **Modular Architecture**: Separated into maintainable components
- **Face Recognition**: Haar + FaceMesh + ArcFace pipeline
- **Persistent Locking**: Lock faces across sessions
- **Activity Logging**: Track movements and expressions
- **MQTT Publishing**: Real-time movement data transmission
- **Heartbeat Monitoring**: System health status

### ESP8266 Camera Control
- **MQTT Subscription**: Receives movement commands
- **Servo Control**: Rotates camera to follow faces
- **WiFi Connectivity**: Connects to local network
- **Real-time Response**: Immediate servo movement

### Web Dashboard
- **Real-time Updates**: WebSocket-based live data
- **Movement Visualization**: Direction indicators and confidence
- **System Status**: Connection health and heartbeat
- **Activity Log**: Historical tracking data

## File Structure

```
FaceRecognition/
├── src/
│   ├── face_recognition_system.py    # Main recognition system
│   ├── face_lock.py               # Face locking logic
│   ├── activity_logger.py          # Activity tracking
│   ├── mqtt_client.py             # MQTT communication
│   ├── face_utils.py             # Face utilities
│   ├── dashboard_server.py        # WebSocket server
│   ├── recognise.py              # Original system (legacy)
│   ├── haar_5pt.py             # Face detection
│   └── tracker.py               # Face tracking
├── hardware/
│   ├── ESP8266_CameraTracker/
│   │   └── ESP8266_CameraTracker.ino  # ESP8266 code
│   └── ArduinoCameraTracker/
│       └── ArduinoCameraTracker.ino     # Original Arduino code
├── web/
│   └── dashboard.html           # Web dashboard
├── data/                       # Database and logs
├── models/                     # ONNX models
└── requirements_new.txt         # Dependencies
```

## Installation

### PC System Dependencies

```bash
pip install -r requirements_new.txt
```

### ESP8266 Setup

1. **Arduino IDE Setup**:
   - Install ESP8266 board manager
   - Add ESP8266 board URL: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
   - Install ESP8266 boards

2. **Required Libraries**:
   - PubSubClient (by Nick O'Leary)
   - ESP8266WiFi (built-in)
   - Servo (built-in)

3. **Hardware Connections**:
   - Servo signal pin → D2 (GPIO4)
   - Servo power → 5V
   - Servo ground → GND

## Configuration

### PC System

Edit `face_recognition_system.py`:
```python
# Change team ID
system = FaceRecognitionSystem(db_path, camera_id=1, team_id="your_team_id")

# MQTT broker settings
mqtt_client = FaceTrackingMQTTClient(
    broker_host="157.173.101.159",
    broker_port=1883,
    team_id="your_team_id"
)
```

### ESP8266

Edit `ESP8266_CameraTracker.ino`:
```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* team_id = "your_team_id";
```

## Usage

### 1. Start PC Vision System

```bash
python -m src.face_recognition_system
```

**Controls**:
- `q`: Quit
- `r`: Reload database
- `+/-`: Adjust threshold
- `d`: Toggle debug overlay
- `t`: Toggle tracking
- `l`: Lock selected face
- `u`: Unlock selected face
- `c`: Clear all locks
- `L`: Reload locks

### 2. Start Dashboard Server

```bash
python -m src.dashboard_server
```

Access dashboard at: `http://localhost:9002`

### 3. Upload ESP8266 Code

1. Open `ESP8266_CameraTracker.ino` in Arduino IDE
2. Select ESP8266 board
3. Configure WiFi credentials
4. Upload to ESP8266

## MQTT Message Format

### Movement Messages
**Topic**: `vision/{team_id}/movement`

```json
{
    "status": "MOVE_LEFT",
    "confidence": 0.87,
    "timestamp": 1730000000
}
```

**Status Values**:
- `NO_FACE_LOCKED`
- `MOVE_LEFT`
- `MOVE_RIGHT`
- `MOVE_UP`
- `MOVE_DOWN`
- `CENTERED`

### Heartbeat Messages
**Topic**: `vision/{team_id}/heartbeat`

```json
{
    "node": "pc",
    "status": "ONLINE",
    "timestamp": 1730000000
}
```

## System Workflow

1. **Face Detection**: PC system detects faces using Haar cascade
2. **Face Recognition**: ArcFace embedding matching identifies persons
3. **Face Locking**: Selected faces are locked and tracked
4. **Movement Detection**: Locked face movements are detected
5. **MQTT Publishing**: Movement data published to broker
6. **ESP8266 Control**: Servo rotates based on movement commands
7. **Dashboard Updates**: Real-time visualization via WebSocket

## Troubleshooting

### Common Issues

1. **MQTT Connection Failed**:
   - Check broker address: `157.173.101.159:1883`
   - Verify network connectivity
   - Check team ID configuration

2. **ESP8266 Not Responding**:
   - Verify WiFi credentials
   - Check MQTT topic configuration
   - Monitor Serial Monitor for errors

3. **Dashboard Not Updating**:
   - Ensure WebSocket server is running
   - Check browser console for errors
   - Verify MQTT message reception

### Debug Mode

Enable debug output:
```python
system = FaceRecognitionSystem(db_path, camera_id=1, team_id="team1")
# Debug automatically enabled via console output
```

## Performance

- **FPS**: 15-30 FPS (depending on hardware)
- **Latency**: <100ms for face detection
- **MQTT Delay**: <50ms for message transmission
- **Servo Response**: <200ms for movement

## Security Notes

- MQTT broker uses unencrypted communication
- WiFi credentials stored in plain text (ESP8266)
- Consider implementing authentication for production use

## Future Enhancements

- [ ] Encrypted MQTT communication
- [ ] Multiple face tracking
- [ ] PTZ camera control
- [ ] Mobile app interface
- [ ] Cloud backend integration
