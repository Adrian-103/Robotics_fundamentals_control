#!/usr/bin/env python3

import rclpy                              # ROS2 client library
from rclpy.node import Node               # Node base class

from rcl_interfaces.msg import SetParametersResult #We will need this to tune the Q and R values
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, PointStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import scipy.linalg as la

def wrapToPi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def quaternion_to_yaw(z, w):
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller_node')

        #Variables:
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.theta = 0.0

        self.omega = 0.0

        self.goal_x = None
        self.goal_y = None

        self.dt = 0.1
        
        #Parameters setup:
        self.declare_parameter('Q_1_1', 1.0)
        self.declare_parameter('Q_2_2', 1.0)
        self.declare_parameter('R_1_1', 1.0)


        self.Q_1_1 = self.get_parameter('Q_1_1').value
        self.Q_2_2 = self.get_parameter('Q_2_2').value
        self.R_1_1 = self.get_parameter('R_1_1').value

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            'cmd_vel',
            10
        )
        
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            'amcl_pose',
            self.amcl_callback,
            10
        )

        self.clicked_point_sub = self.create_subscription(
            PointStamped,
            'clicked_point',
            self.clicked_point_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10
        )

        self.add_on_set_parameters_callback(self.parameters_callback)

        # Variables extra para el control y seguridad
        self.front_distance = 3.5 
        self.v_max = 0.2  # Velocidad máxima en m/s
        self.K = np.array([0.0, 0.0]) # Se actualizará enseguida

        # 1. Calculamos la matriz K por primera vez con los parámetros por defecto
        self.calculate_k_matrix()

        # 2. Creamos el Timer que ejecutará el control_loop cada dt (0.1s)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Controller node started.")
    
    def calculate_k_matrix(self):
        """Calcula las ganancias LQR usando el modelo de doble integrador"""
        # Matrices del sistema linealizado (A y B)
        A = np.array([[0.0, 1.0], 
                      [0.0, 0.0]])
        B = np.array([[0.0], 
                      [1.0]])
        
        # Matrices de costo con tus parámetros de ROS 2
        Q = np.array([[self.Q_1_1, 0.0],
                      [0.0, self.Q_2_2]])
        R = np.array([[self.R_1_1]])
        
        try:
            # Resolvemos la Ecuación Algebraica de Riccati
            P = la.solve_continuous_are(A, B, Q, R)
            # Calculamos K = R^-1 * B^T * P
            K_matrix = np.linalg.inv(R).dot(B.T).dot(P)
            self.K = K_matrix[0] # Nos quedamos con el arreglo 1D [k1, k2]
            
            self.get_logger().info(f"LQR actualizado -> K: [{self.K[0]:.4f}, {self.K[1]:.4f}]")
        except Exception as e:
            self.get_logger().error(f"Error al calcular LQR (revisa tus Q y R): {e}")
    
    def control_loop(self):
        # 1. Seguridad básica: Si hay un obstáculo muy cerca, nos detenemos.
        # (Esto será reemplazado después por la Fuerza Repulsiva real)
        if self.front_distance < 0.5:
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd) # Publica puros ceros
            return

        # 2. Si no hemos recibido un click en el mapa, no hacemos nada
        if self.goal_x is None or self.goal_y is None:
            return

        # --- CAMPOS POTENCIALES (Fuerza Atractiva) ---
        Fx = self.goal_x - self.pose_x
        Fy = self.goal_y - self.pose_y
        
        # Calculamos hacia qué ángulo debemos apuntar
        theta_d = math.atan2(Fy, Fx)
        
        # Calculamos el error de dirección (la diferencia entre adónde veo y adónde quiero ir)
        e_theta = wrapToPi(theta_d - self.theta)

        # --- LQR (Control de Dirección) ---
        # Definimos nuestro estado actual x = [error_angulo, velocidad_angular]
        x_state = np.array([e_theta, self.omega])
        
        # Ley de control LQR: u = -Kx (Esto nos da la aceleración angular)
        u = -np.dot(self.K, x_state)

        # Actualización de Euler (Suavizador de movimiento)
        self.omega = self.omega + (u * self.dt)
        
        # Clamp: Límite físico de los motores del TurtleBot para giro
        self.omega = np.clip(self.omega, -2.8, 2.8)

        # --- MODULACIÓN DE VELOCIDAD LINEAL ---
        # Solo acelera si el robot está mirando hacia la meta (evita ir de reversa)
        v = self.v_max * max(0.0, math.cos(e_theta))

        # --- PUBLICAR COMANDOS ---
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(self.omega)
        self.cmd_vel_pub.publish(cmd)

        # --- NUEVO: CONDICIÓN DE LLEGADA ---
        distancia_a_meta = math.hypot(Fx, Fy)
        if distancia_a_meta < 0.1: # Si está a menos de 10 cm de la meta
            self.get_logger().info("¡Meta alcanzada!")
            cmd = Twist() # Publica Twist vacío (puros ceros)
            self.cmd_vel_pub.publish(cmd)
            self.goal_x = None # Borra la meta para esperar el siguiente click
            self.goal_y = None
            return

    def parameters_callback(self, params):
        for param in params:
            if param.name == 'Q_1_1':
                if param.value <= 0:
                    self.get_logger().warn('No values below or equal to zero are accepted')
                    return SetParametersResult(successful=False, reason="Incorrect input value")
                self.Q_1_1 = param.value
                self.get_logger().info(f"Q_1_1 matrix value updated to: {self.Q_1_1}")
                
            elif param.name == 'Q_2_2':
                if param.value <= 0:
                    self.get_logger().warn('No values below or equal to zero are accepted')
                    return SetParametersResult(successful=False, reason="Incorrect input value")
                self.Q_2_2 = param.value
                self.get_logger().info(f"Q_2_2 matrix value updated to: {self.Q_2_2}")
                
            elif param.name == 'R_1_1':
                if param.value <= 0:
                    self.get_logger().warn('No values below or equal to zero are accepted')
                    return SetParametersResult(successful=False, reason="Incorrect input value")
                self.R_1_1 = param.value
                self.get_logger().info(f"R matrix value updated to: {self.R_1_1}")
        
        self.calculate_k_matrix()
        return SetParametersResult(successful=True)


    def clicked_point_callback(self, msg):
        # Guardamos el objetivo en la memoria del nodo
        self.goal_x = msg.point.x
        self.goal_y = msg.point.y
        self.get_logger().info(f"New objective received: X = {self.goal_x:.2f}, Y = {self.goal_y:.2f}")
    
    def amcl_callback(self, msg):
        # Guardamos la posición en la memoria
        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y
        
        # Extraemos el cuaternión y lo convertimos a radianes
        rot_z = msg.pose.pose.orientation.z
        rot_w = msg.pose.pose.orientation.w
        self.theta = quaternion_to_yaw(rot_z, rot_w)
        
        # Opcional: imprimir para depurar (con cuidado porque amcl publica muy rápido)
        # self.get_logger().info(f"Pose updated: X={self.pose_x:.2f}, Y={self.pose_y:.2f}, Theta={self.theta:.2f}")
    
    def scan_callback(self, msg):
        # Guardamos todo el arreglo por si lo necesitamos para los campos potenciales completos
        self.laser_ranges = msg.ranges

        # Lógica temporal solo para el frente (índice 0)
        front_distance = msg.ranges[0]
        if front_distance == float('inf'):
            self.front_distance = 3.5
        else:
            self.front_distance = front_distance
            
        if self.front_distance < 0.5:
            self.get_logger().warn(f"Obstacle detected in front, {self.front_distance:.2f} meters.")


def main(args=None):

    rclpy.init(args=args)             # Initialize communication
    node = TurtleBotController()      # Create node instance
    try:
        rclpy.spin(node)              # Keep node running
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()           # Destroy node
        rclpy.shutdown()              # Shutdown communication

if __name__ == '__main__':
    main()  # Run main only if script executed directly