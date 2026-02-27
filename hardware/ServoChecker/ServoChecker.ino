#include <Servo.h>
Servo s;

void setup() {
  s.attach(D5);
}

void loop() {
  s.write(0);
  delay(2000);
  s.write(90);
  delay(2000);
  s.write(180);
  delay(2000);
}