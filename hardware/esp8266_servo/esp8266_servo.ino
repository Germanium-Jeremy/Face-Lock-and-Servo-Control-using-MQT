#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

Servo myservo;
int servoPin = D5;
int currentAngle = 90;
int stepSize = 5;

struct WiFiNetwork {
  const char* ssid;
  const char* password;
};

WiFiNetwork networks[] = {
  {"RCA-OUTDOOR", "RCA@2025"},
  {"EdNet", "Huawei@123"},
  {"Main Hall", "Meeting@2024"},  
  {"GROUND", "RCA@2024"},
  {"RCA-OFFICE", "RCA@2024"}
};

int currentNetwork = 0;

const char* mqtt_server = "10.12.74.5";
const int mqtt_port = 1883;
const char* mqtt_topic_sub = "vision/Germany/movement";

WiFiClient espClient;
PubSubClient client(espClient);

void connectWiFi(int index) {
  WiFi.disconnect();
  delay(500);

  Serial.print("Connecting to ");
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
    currentNetwork = index;
  } else {
    Serial.println("\nFailed");
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

  if (message.indexOf("MOVE_LEFT") >= 0) moveServo(stepSize);
  if (message.indexOf("MOVE_RIGHT") >= 0) moveServo(-stepSize);
  if (message.indexOf("CENTER") >= 0) {
    currentAngle = 90;
    myservo.write(currentAngle);
  }
}

void reconnectMQTT() {
  while (!client.connected() && WiFi.status() == WL_CONNECTED) {
    if (client.connect("esp8266_servo")) {
      client.subscribe(mqtt_topic_sub);
    } else {
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  myservo.attach(servoPin, 500, 2400);
  myservo.write(currentAngle);

  connectWiFi(currentNetwork);

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void checkSerialWiFiSwitch() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.startsWith("WIFI")) {
      int index = cmd.substring(4).toInt();
      if (index >= 0 && index < 3) {
        connectWiFi(index);
      } else {
        Serial.println("Invalid WiFi index");
      }
    }
  }
}

void loop() {
  checkSerialWiFiSwitch();

  if (WiFi.status() != WL_CONNECTED) return;

  if (!client.connected())
    reconnectMQTT();

  client.loop();
}