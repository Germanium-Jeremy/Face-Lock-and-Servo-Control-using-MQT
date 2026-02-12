#include <Stepper.h>

// Number of steps per revolution for 28BYJ-48 stepper motor
const int stepsPerRevolution = 2048; // Using half-stepping
const float degreesPerStep = 360.0 / stepsPerRevolution; // Calculate degrees per step

// Define ULN2003 input pins connected to Arduino
const int IN1 = 8;
const int IN2 = 9;
const int IN3 = 10;
const int IN4 = 11;

// Initialize the stepper with the above pins
Stepper myStepper(stepsPerRevolution, IN1, IN3, IN2, IN4);

void setup() {
  myStepper.setSpeed(15); // Set speed in RPM (adjust as needed)
  Serial.begin(9600);     // Start serial communication
  Serial.println("Enter command: CW <degrees> for clockwise, CCW <degrees> for counterclockwise");
}

void loop() {
  // Check if data has been sent from the Serial Monitor
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n'); // Read the command

    // Print the received command
    Serial.print("Received command: ");
    Serial.println(command);

    // Process the command
    if (command.startsWith("CW")) {
      // Extract the number of degrees
      float degrees = command.substring(3).toFloat(); // Get the number after "CW "
      if (degrees > 0) {
        int steps = degrees / degreesPerStep; // Convert degrees to steps
        Serial.print("Rotating clockwise for ");
        Serial.print(degrees);
        Serial.println(" degrees.");
        myStepper.step(steps);  // Rotate clockwise
      } else {
        Serial.println("Invalid number of degrees.");
      }
    } else if (command.startsWith("CCW")) {
      // Extract the number of degrees
      float degrees = command.substring(4).toFloat(); // Get the number after "CCW "
      if (degrees > 0) {
        int steps = degrees / degreesPerStep; // Convert degrees to steps
        Serial.print("Rotating counterclockwise for ");
        Serial.print(degrees);
        Serial.println(" degrees.");
        myStepper.step(-steps); // Rotate counterclockwise
      } else {
        Serial.println("Invalid number of degrees.");
      }
    } else {
      Serial.println("Invalid command. Use 'CW <degrees>' or 'CCW <degrees>'.");
    }
  }
}