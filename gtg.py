import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.timer import Timer
from turtlesim_msgs.msg import Pose


class GTG(Node):
    """
    Transforma la posición del robot a un mensaje tipo `Twist` para transportarse a la coordenada
    dada por `sys.argv`.
    """

    cmd_vel: Publisher
    """Recibe y publica el mensaje para transportar el robot"""
    pose_sub: Subscription
    """Usada para actualizar la pose actual del robot"""
    timer: Timer
    """Cada 0.1s publica el mensaje para moverse al destino"""
    pose: Pose
    """Mantenemos la pose actual para saber a donde estamos"""

    def __init__(self):
        super().__init__("GoToGoal")
        self.cmd_vel = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)
        self.pose_sub = self.create_subscription(
            Pose, "/turtle1/pose", self.pose_callback, 10
        )
        self.timer = self.create_timer(0.1, self.go_to_goal)
        self.pose = Pose()

    def pose_callback(self, data: Pose):
        """
        Actualiza la pose del robot en este momento. Se usa junto a una suscripcion
        a `/turtle1/pose` para saber dónde diantres estamos parados.

        Args:
            data: La pose actual recibida
        """
        self.pose = data

    def go_to_goal(self):
        """
        Dada la pose actual, calcula el `Twist` necesario para acercarse al destino
        """
        goal = Pose()
        goal.x = float(sys.argv[1])
        goal.y = float(sys.argv[2])
        goal.theta = float(sys.argv[3])

        new_vel = Twist()

        a2g = math.atan2(goal.y - self.pose.y, goal.x - self.pose.x)
        d2g = math.sqrt((goal.x - self.pose.x) ** 2 + (goal.y - self.pose.y) ** 2)

        dtol = 0.10
        atol = 0.01

        a_error = a2g - self.pose.theta
        self.log(f"{a_error = }")
        kp = 10.0

        if abs(a_error) > atol:
            new_vel.angular.z = a_error * kp
        elif d2g >= dtol:
            new_vel.linear.x = kp * d2g
        else:
            new_vel.linear.x = 0.0
            self.cmd_vel.publish(new_vel)
            self.log("Goal Reached")
            quit()

        self.cmd_vel.publish(new_vel)

    def log(self, msg: str):
        """Utilidad para loggear mensajes más fácil"""
        self.get_logger().info(msg)


def main(args=None):
    rclpy.init(args=args)
    minimal_pub = GTG()
    rclpy.spin(minimal_pub)
    minimal_pub.destroy_node()
