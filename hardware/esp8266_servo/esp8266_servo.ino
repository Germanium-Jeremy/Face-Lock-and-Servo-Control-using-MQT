#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

Servo myservo;
int servoPin = D4;
int currentAngle = 0;
int targetAngle = 90;
unsigned long lastMoveTime = 0;
int moveInterval = 15; // ms between 1-degree steps
unsigned long lastReconnectAttempt = 0;

WiFiClient espClient;
PubSubClient client(espClient);

// ===== Multiple WiFi networks =====
struct WiFiCred {
  const char* ssid;
  const char* password;
};

WiFiCred networks[] = {
  {"God-Only-Knows", "Arlene+250"},
  {"KARAHABUTAKA", "KARAHABUTAKA"},
  {"Main Hall", "Meeting@2024"},
  {"RCA-OUTDOOR", "RCA@2025"},
  {"EdNet", "Huawei@123"},
  {"GROUND", "RCA@2024"}
};

int currentWiFiIndex = 0;
const int totalNetworks = sizeof(networks)/sizeof(networks[0]);

// MQTT
const char* mqtt_server = "157.173.101.159";
const int mqtt_port = 1883;
const char* mqtt_topic_sub = "vision/Phoenix_team/movement";

void connectToWiFi(int index) {
  if (index < 0 || index >= totalNetworks) return;

  WiFi.disconnect();
  delay(1000);

  Serial.print("Connecting to: ");
  Serial.println(networks[index].ssid);

  WiFi.begin(networks[index].ssid, networks[index].password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi Connected");
    currentWiFiIndex = index;
  } else {
    Serial.println("\nFailed to connect");
  }
}



void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++)
    message += (char)payload[i];
    
  Serial.println(message);

  // Simple substring parsing to find the angle in the JSON payload
  int angleIdx = message.indexOf("\"angle\":");
  if (angleIdx >= 0) {
    int startIdx = angleIdx + 8; // skip past "angle":
    // skip spaces
    while (startIdx < message.length() && message.charAt(startIdx) == ' ') {
      startIdx++;
    }
    int endIdx = startIdx;
    while (endIdx < message.length() && isDigit(message.charAt(endIdx))) {
      endIdx++;
    }
    
    if (endIdx > startIdx) {
      String degStr = message.substring(startIdx, endIdx);
      targetAngle = degStr.toInt();
      if (targetAngle < 0) targetAngle = 0;
      if (targetAngle > 180) targetAngle = 180;
      Serial.print("Target angle: ");
      Serial.println(targetAngle);
    }
  }
}

void reconnectMQTT() {
  unsigned long now = millis();
  if (now - lastReconnectAttempt > 2000 || lastReconnectAttempt == 0) {
    lastReconnectAttempt = now;
    Serial.println("Attempting MQTT connection...");
    if (client.connect("esp8266_servo")) {
      Serial.println("MQTT connected");
      client.subscribe(mqtt_topic_sub);
      lastReconnectAttempt = 0;
    } else {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 2 seconds");
    }
  }
}

void checkSerialWiFiSwitch() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("WIFI")) {
      int index = cmd.substring(4).toInt();
      connectToWiFi(index);
    }
  }
}

void setup() {
  Serial.begin(115200);

  myservo.attach(servoPin, 500, 2400);
  myservo.write(currentAngle);

  connectToWiFi(currentWiFiIndex);

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  checkSerialWiFiSwitch();

  if (WiFi.status() == WL_CONNECTED) {
    if (!client.connected())
      reconnectMQTT();
    client.loop();
    
    // Smooth movement without blocking
    if (millis() - lastMoveTime >= moveInterval) {
      if (currentAngle < targetAngle) {
        currentAngle++;
        myservo.write(currentAngle);
      } else if (currentAngle > targetAngle) {
        currentAngle--;
        myservo.write(currentAngle);
      }
      lastMoveTime = millis();
    }
  }
}
