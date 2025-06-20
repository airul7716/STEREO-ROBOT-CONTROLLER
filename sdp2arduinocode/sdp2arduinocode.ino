#include <Servo.h>

Servo thumb, indexF, middle, ring, pinky, wrist;

void setup() {
  Serial.begin(9600);

  thumb.attach(3); //
  indexF.attach(9);
  middle.attach(5);//
  ring.attach(6);//
  pinky.attach(10);
  wrist.attach(11); // Use another pin for wrist if needed

  // Set all servos to neutral
  thumb.write(170);     // Open
  indexF.write(170);    // Open
  middle.write(170);
  ring.write(170);
  pinky.write(172);
  wrist.write(90);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int steps[6];
    int i = 0;
    char *token = strtok((char *)input.c_str(), ",");

    while (token != NULL && i < 6) {
      steps[i++] = atoi(token);
      token = strtok(NULL, ",");
    }

    if (i == 6) {
      int thumbStep = steps[0];
      int indexStep = steps[1];
      int middleStep = steps[2];
      int ringStep = steps[3];
      int pinkyStep = steps[4];
      int wristStep = steps[5];

      thumb.write(map(thumbStep, 0, 10, 170, 60));     // ✅ Custom thumb mapping
      indexF.write(map(indexStep, 0, 17, 170, 25));
      middle.write(map(middleStep, 0, 17, 170, 30));
      ring.write(map(ringStep, 0, 17, 170, 25));
      pinky.write(map(pinkyStep, 0, 17, 172, 40));
      wrist.write(map(wristStep, 0, 17, 180, 0));
    }

    // Optional debug:
    Serial.println("Received: " + input);
  }
}



