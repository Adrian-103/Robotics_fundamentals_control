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

        self.F_rep_x = 0.0
        self.F_rep_y = 0.0
        
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
        # 1. Seguridad básica: Freno de emergencia. 
        # Lo bajamos a 0.25m porque ahora el algoritmo repulsivo lo hará girar mucho antes.
        if self.front_distance < 0.10:
            self.get_logger().warn("¡Freno de emergencia por colisión inminente!")
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return

        # 2. Si no hemos recibido un click en el mapa, no hacemos nada
        if self.goal_x is None or self.goal_y is None:
            return

        # --- 3. CAMPOS POTENCIALES (Fuerza Atractiva) ---
        k_att = 2.0  # Ganancia de atracción (qué tan fuerte jala la meta)
        F_att_x = k_att * (self.goal_x - self.pose_x)
        F_att_y = k_att * (self.goal_y - self.pose_y)
        
        # Condición de llegada (calculada usando la distancia real a la meta)
        distancia_a_meta = math.hypot(self.goal_x - self.pose_x, self.goal_y - self.pose_y)
        if distancia_a_meta < 0.1: # Si está a menos de 10 cm de la meta
            self.get_logger().info("¡Meta alcanzada!")
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            self.goal_x = None 
            self.goal_y = None
            return
        
        # --- NUEVO: EL FACTOR DE ATENUACIÓN (Solución al GNRON) ---
        # Si estamos a menos de 1 metro de la meta, empezamos a reducir el miedo al obstáculo.
        # Si estamos a 0.2m, la repulsión se multiplica por 0.2 (se vuelve muy débil).
        factor_atenuacion = min(1.0, distancia_a_meta)



        # --- 4. LA SUMA MÁGICA (Fuerza Total) ---
        # Multiplicamos la repulsión por el factor de atenuación
        F_tot_x = F_att_x + (self.F_rep_x * factor_atenuacion)
        F_tot_y = F_att_y + (self.F_rep_y * factor_atenuacion)
        
        # Calculamos hacia qué ángulo debemos apuntar para seguir la Fuerza Total
        theta_d = math.atan2(F_tot_y, F_tot_x)
        
        # Calculamos el error de dirección (recordando el orden correcto)
        e_theta = wrapToPi(self.theta - theta_d)

        # --- 5. LQR (Control de Dirección) ---
        x_state = np.array([e_theta, self.omega])
        u = -np.dot(self.K, x_state)

        # Actualización de Euler
        self.omega = self.omega + (u * self.dt)
        self.omega = np.clip(self.omega, -2.8, 2.8)

        # --- 6. MODULACIÓN DE VELOCIDAD LINEAL ---
        v = self.v_max * max(0.0, math.cos(e_theta))

        # --- PUBLICAR COMANDOS ---
        cmd = Twist()
        cmd.linear.x = float(v)
        cmd.angular.z = float(self.omega)
        self.cmd_vel_pub.publish(cmd)

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
        self.laser_ranges = msg.ranges

        # --- 1. Parámetros del Campo Repulsivo ---
        d_max = 1.0  # Radio de influencia: ignoramos obstáculos a más de 1 metro
        k_rep = 1.0  # Ganancia repulsiva: qué tan fuerte "patean" los obstáculos
        
        F_rep_x_local = 0.0
        F_rep_y_local = 0.0
        
        # --- 2. Procesamiento de los rayos del LIDAR ---
        dir_x_sum = 0.0
        dir_y_sum = 0.0
        d_min = float('inf') # Guardaremos la distancia al punto más cercano

        for i, dist in enumerate(msg.ranges):
            if dist < 0.1 or math.isinf(dist) or math.isnan(dist):
                continue
                
            if dist < d_max and (i <= 110 or i >= 250):
                # Guardar la distancia más corta para calcular la fuerza
                if dist < d_min:
                    d_min = dist
                
                # Solo sumar las direcciones (empujando en contra)
                angle_local = math.radians(i)
                dir_x_sum -= math.cos(angle_local)
                dir_y_sum -= math.sin(angle_local)

        # --- 3. NORMALIZAR Y APLICAR LA FUERZA (El consejo del profe) ---
        if d_min < d_max: # Si vimos al menos un obstáculo
            # Normalizar: Medimos cuánto mide el vector de direcciones sumadas
            norm = math.hypot(dir_x_sum, dir_y_sum)
            if norm > 0:
                # Al dividir entre la norma, hacemos que el vector mida exactamente 1.0
                dir_x_norm = dir_x_sum / norm
                dir_y_norm = dir_y_sum / norm
                
                # Calculamos UNA SOLA magnitud usando el punto más cercano
                magnitude = k_rep * (1.0/d_min - 1.0/d_max) / (d_min**2)
                magnitude = min(magnitude, 2.0) # Ahora sí, el tope funcionará
                
                # Multiplicamos la dirección normalizada por la magnitud
                F_rep_x_local = dir_x_norm * magnitude
                F_rep_y_local = dir_y_norm * magnitude

        # --- 4. Transformación al Marco Global ---
        self.F_rep_x = F_rep_x_local * math.cos(self.theta) - F_rep_y_local * math.sin(self.theta)
        self.F_rep_y = F_rep_x_local * math.sin(self.theta) + F_rep_y_local * math.cos(self.theta)

        # Mantenemos tu freno de emergencia intacto para el control_loop
        # Freno de emergencia más robusto
        dist_frente = msg.ranges[0]
        if math.isinf(dist_frente) or math.isnan(dist_frente) or dist_frente < 0.05:
            self.front_distance = 3.5 # Asumimos vía libre si hay error de lectura
        else:
            self.front_distance = dist_frente


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