/*
ESP8266 MQTT Servo Controller for Face Tracking
Receives movement commands via MQTT and controls servo motor accordingly
*/

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT broker settings
const char* mqtt_server = "157.173.101.159";
const int mqtt_port = 1883;
const char* team_id = "team1";  // Change this to your team ID

// MQTT topics
String movement_topic = String("vision/") + team_id + "/movement";

// Servo configuration
#define SERVO_PIN D2  // GPIO pin for servo
Servo trackingServo;

// Movement tracking
int current_angle = 90;  // Start at center position
int min_angle = 0;
int max_angle = 180;
int step_size = 5;  // Degrees to move per command

// WiFi and MQTT clients
WiFiClient espClient;
PubSubClient client(espClient);

// Timing
unsigned long last_wifi_check = 0;
unsigned long last_mqtt_check = 0;
const unsigned long check_interval = 5000;  // 5 seconds

void setup() {
  Serial.begin(115200);
  
  // Initialize servo
  trackingServo.attach(SERVO_PIN);
  trackingServo.write(current_angle);
  
  Serial.println("ESP8266 Face Tracking Servo Controller");
  Serial.println("Initializing...");
  
  // Connect to WiFi
  setup_wifi();
  
  // Configure MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(mqtt_callback);
  
  Serial.println("Setup complete");
}

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    if (millis() - last_wifi_check > check_interval) {
      Serial.println("WiFi disconnected, attempting to reconnect...");
      setup_wifi();
      last_wifi_check = millis();
    }
  }
  
  // Check MQTT connection
  if (!client.connected()) {
    if (millis() - last_mqtt_check > check_interval) {
      Serial.println("MQTT disconnected, attempting to reconnect...");
      reconnect_mqtt();
      last_mqtt_check = millis();
    }
  }
  
  // Process MQTT messages
  if (client.connected()) {
    client.loop();
  }
  
  delay(100);  // Small delay to prevent overwhelming
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("WiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("Failed to connect to WiFi");
  }
}

void reconnect_mqtt() {
  // Loop until we're reconnected
  int attempts = 0;
  while (!client.connected() && attempts < 3) {
    Serial.print("Attempting MQTT connection...");
    
    // Create a random client ID
    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      
      // Subscribe to movement topic
      client.subscribe(movement_topic.c_str());
      Serial.print("Subscribed to: ");
      Serial.println(movement_topic);
      
      // Send initial position
      send_servo_position();
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
    attempts++;
  }
}

void mqtt_callback(char* topic, byte* payload, unsigned int length) {
  // Convert payload to string
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  Serial.println(message);
  
  // Parse JSON message
  // Expected format: {"status": "MOVE_LEFT", "confidence": 0.87, "timestamp": 1730000000}
  
  if (message.indexOf("\"status\"") >= 0) {
    // Extract status
    int status_start = message.indexOf("\"status\":\"") + 10;
    int status_end = message.indexOf("\"", status_start);
    
    if (status_start > 9 && status_end > status_start) {
      String status = message.substring(status_start, status_end);
      
      // Process movement command
      process_movement_command(status);
    }
  }
}

void process_movement_command(String status) {
  Serial.print("Processing command: ");
  Serial.println(status);
  
  if (status == "MOVE_LEFT") {
    move_servo(-step_size);
  } else if (status == "MOVE_RIGHT") {
    move_servo(step_size);
  } else if (status == "MOVE_UP") {
    // For vertical movement, you might add a second servo
    Serial.println("Vertical movement not implemented (single servo setup)");
  } else if (status == "MOVE_DOWN") {
    // For vertical movement, you might add a second servo
    Serial.println("Vertical movement not implemented (single servo setup)");
  } else if (status == "CENTERED") {
    // Move to center position
    current_angle = 90;
    trackingServo.write(current_angle);
    Serial.print("Moved to center: ");
    Serial.println(current_angle);
  } else if (status == "NO_FACE_LOCKED") {
    // Could implement search pattern here
    Serial.println("No face locked - could implement search pattern");
  }
  
  delay(100);  // Small delay after movement
}

void move_servo(int angle_change) {
  current_angle += angle_change;
  
  // Constrain angle to valid range
  if (current_angle < min_angle) {
    current_angle = min_angle;
  } else if (current_angle > max_angle) {
    current_angle = max_angle;
  }
  
  trackingServo.write(current_angle);
  Serial.print("Servo moved to: ");
  Serial.println(current_angle);
  
  // Send position feedback
  send_servo_position();
}

void send_servo_position() {
  // Could send servo position back via MQTT if needed
  // This would require a separate topic for position feedback
  Serial.print("Current servo position: ");
  Serial.println(current_angle);
}
