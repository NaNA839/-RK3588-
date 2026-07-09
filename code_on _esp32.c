#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <HardwareSerial.h>

HardwareSerial wireSerial(2);

// =========================
#define MQ2_AO 0
#define MQ4_AO 1
#define MQ7_AO 6
#define BUZZER_PIN 12

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

#define SERVICE_UUID "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// =========================
class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
    Serial.println("Phone Connected");
  }

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
    Serial.println("Phone Disconnected");
    BLEDevice::startAdvertising();
  }
};

// =========================
void setup() {

  Serial.begin(115200);

  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, HIGH);

  BLEDevice::init("Genshin Impact");

  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);

  pCharacteristic = pService->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_READ |
      BLECharacteristic::PROPERTY_NOTIFY
  );

  pCharacteristic->addDescriptor(new BLE2902());

  pService->start();

  
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);

  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);

  pAdvertising->start();

  Serial.println("BLE Started");

  wireSerial.begin(115200, SERIAL_8N1, 16, 17);
}

// =========================
void loop() {

  int mq2_value = analogRead(MQ2_AO);
  int mq4_value = analogRead(MQ4_AO);
  int mq7_value = analogRead(MQ7_AO);

  bool danger = mq2_value > 2500 || mq4_value > 2500 || mq7_value > 2500;

  if (danger) digitalWrite(BUZZER_PIN, HIGH);
  else digitalWrite(BUZZER_PIN, LOW);

  String message =
      "MQ2:" + String(mq2_value) +
      ",MQ4:" + String(mq4_value) +
      ",MQ7:" + String(mq7_value);

  if (danger) message += " ALERT";

  Serial.println(message);
  wireSerial.println(message);

  if (deviceConnected) {
    pCharacteristic->setValue(message.c_str());
    pCharacteristic->notify();
  }

  delay(1000);
}
