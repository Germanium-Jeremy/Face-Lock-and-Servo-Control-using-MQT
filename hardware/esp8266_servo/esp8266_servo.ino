#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

Servo myservo;
int servoPin = D5;     // try D5 first
int currentAngle = 90;
int stepSize = 5;

const char* ssid = "RCA-OUTDOOR";
const char* password = "RCA@2025";

const char* mqtt_server = "157.173.101.159";
const int mqtt_port = 1883;
const char* mqtt_topic_sub = "vision/Phoenix_Team/movement";

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi OK");
}

void moveServo(int delta) {
  currentAngle += delta;
  if (currentAngle < 0) currentAngle = 0;
  if (currentAngle > 180) currentAngle = 180;

  myservo.write(currentAngle);
  Serial.print("Angle: ");
  Serial.println(currentAngle);
}

void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (int i = 0; i < length; i++)
    message += (char)payload[i];

  Serial.println(message);

  if (message.indexOf("MOVE_LEFT") >= 0) {
    moveServo(stepSize);
  }

  if (message.indexOf("MOVE_RIGHT") >= 0) {
    moveServo(-stepSize);
  }

  if (message.indexOf("CENTER") >= 0) {
    currentAngle = 90;
    myservo.write(currentAngle);
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("MQTT...");
    if (client.connect("esp8266_servo")) {
      Serial.println("OK");
      client.subscribe(mqtt_topic_sub);
    } else {
      Serial.println("retry");
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);

  // IMPORTANT: attach with pulse range
  myservo.attach(servoPin, 500, 2400);
  myservo.write(currentAngle);

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();
}
