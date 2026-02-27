#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

Servo myservo;
int servoPin = D5;
int currentAngle = 0;
int stepSize = 5;

WiFiClient espClient;
PubSubClient client(espClient);

// ===== Multiple WiFi networks =====
struct WiFiCred {
  const char* ssid;
  const char* password;
};

WiFiCred networks[] = {
  {"RCA-OUTDOOR", "RCA@2025"},
  {"EdNet", "Huawei@123"},
  {"Main Hall", "Meeting@2024"},
  {"GROUND", "RCA@2024"},
  {"RCA-OFFICE", "RCA@2024"}
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

void moveServo(int delta) {
  currentAngle += delta;
  if (currentAngle < 0) currentAngle = 0;
  if (currentAngle > 180) currentAngle = 180;
  myservo.write(currentAngle);
}

void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++)
    message += (char)payload[i];

  if (message.indexOf("MOVE_LEFT") >= 0)
    moveServo(stepSize);

  if (message.indexOf("MOVE_RIGHT") >= 0)
    moveServo(-stepSize);

  if (message.indexOf("CENTER") >= 0) {
    currentAngle = 90;
    myservo.write(currentAngle);
  }
}

void reconnectMQTT() {
  while (!client.connected()) {
    if (client.connect("esp8266_servo")) {
      client.subscribe(mqtt_topic_sub);
    } else {
      delay(2000);
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
  }
}