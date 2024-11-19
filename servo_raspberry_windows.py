import time
from gpiozero import Servo, Device
from gpiozero.pins.mock import MockFactory, MockPWMPin
from time import sleep

# Configuración del simulador
Device.pin_factory = MockFactory(pin_class=MockPWMPin)

# Configuración del servo
servo = Servo(17)

def set_angle(angle):
    # Convertir el ángulo a un valor entre -1 y 1
    value = (angle / 180) * 2 - 1
    servo.value = value
    sleep(1)

try:
    while True:
        angle = float(input("Ingrese el ángulo (0 a 180): "))
        set_angle(angle)
except KeyboardInterrupt:
    pass