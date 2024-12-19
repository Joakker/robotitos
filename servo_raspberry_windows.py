import time
from gpiozero import Servo, Device
from gpiozero.pins.mock import MockFactory, MockPWMPin
from time import sleep

# Configuración del simulador
#Device.pin_factory = MockFactory(pin_class=MockPWMPin)

# Configuración del servo
servo = Servo(0)

def set_angle():
    while True:    
        servo.min()
        sleep(1)
        servo.mid()
        sleep(1)
        servo.max()
        sleep(1)


if __name__ == "__main__":
    set_angle();