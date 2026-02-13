#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// Initialize Servo
Servo myservo;
int servoPin = D1; // Servo connected to D1
int currentAngle = 90;
int stepSize = 2;

// Wi-Fi Credentials - CHANGE THESE!
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT Broker
const char* mqtt_server = "10.11.75.113";
const int mqtt_port = 1883;
const char* mqtt_topic_sub = "vision/Germany/movement"; // Change Germany to match Python

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
  String message;
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  Serial.println(message);

  // Parse JSON manually or just search for keywords
  // Expected format: {"status": "MOVE_LEFT", ...}
  
  if (message.indexOf("MOVE_LEFT") > 0) {
    currentAngle += stepSize;
    if (currentAngle > 180) currentAngle = 180;
    myservo.write(currentAngle);
    Serial.print("Moving LEFT -> ");
    Serial.println(currentAngle);
  } else if (message.indexOf("MOVE_RIGHT") > 0) {
    currentAngle -= stepSize;
    if (currentAngle < 0) currentAngle = 0;
    myservo.write(currentAngle);
    Serial.print("Moving RIGHT -> ");
    Serial.println(currentAngle);
  } else if (message.indexOf("CENTER") > 0) {
    // Optional: Only center if strictly requested, but usually we just hold position
    // currentAngle = 90;
    // myservo.write(currentAngle);
  }
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);
    
    // Attempt to connect
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      client.subscribe(mqtt_topic_sub);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  myservo.attach(servoPin);
  myservo.write(currentAngle);
  
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
