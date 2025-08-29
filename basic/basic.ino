#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <HX711.h>

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

#define DOUT 4
#define CLK 5
HX711 scale;

#define buzzerPin 12
#define ledPin 13
#define buttonPin 33
#define IR_SENSOR_PIN 15
#define TRIG_PIN 26
#define ECHO_PIN 27
#define LDR_PIN 34

long duration;
float distance;

void setup() {
  Serial.begin(115200);

  pinMode(buzzerPin, OUTPUT);
  pinMode(ledPin, OUTPUT);
  pinMode(buttonPin, INPUT_PULLUP);
  pinMode(IR_SENSOR_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  scale.begin(DOUT, CLK);
  scale.set_scale();
  scale.tare();

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
}

void loop() {
  // Load cell reading
  long weight = scale.get_units();
  Serial.print("Weight: ");
  Serial.print(weight);
  Serial.println(" g");

  // Ultrasonic distance
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  duration = pulseIn(ECHO_PIN, HIGH);
  distance = duration * 0.034 / 2;

  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // IR sensor reading
  int irState = digitalRead(IR_SENSOR_PIN);
  Serial.print("IR Sensor: ");
  Serial.println(irState == LOW ? "Detected" : "Not Detected");

  display.clearDisplay();
  display.setCursor(0, 0);

  if (irState == HIGH) {
    // If IR sensor is not detected, check distance
    if (distance > 35) {
      // FULL EMPTY case with buzzer for 5 seconds
      display.setTextSize(2);
      display.setCursor(0, 0);
      display.println("FULL EMPTY");
      display.display();
      digitalWrite(buzzerPin, HIGH);
      delay(5000);  // Buzzer sounds for 5 seconds
      digitalWrite(buzzerPin, LOW);
    } else {
      // Display PARTIALLY EMPTY message when IR is not detected
      display.setTextSize(2);
      display.setCursor(0, 0);
      display.println("PARTIALLY");
      display.setCursor(0, 30);
      display.println("EMPTY");
      display.display();
    }
  } else {
    // If IR sensor is detected (irState == LOW), display weight and distance
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("Wt: ");
    display.print(weight);
    display.print("g D: ");
    display.print(distance);
    display.println("cm");
    display.display();
  }

  delay(1000);
}
