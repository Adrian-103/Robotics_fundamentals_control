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


class MPC_controller(Node):
    def __init__(self):
        super().__init__('mpc_controller_node')

        #Variables:
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.theta = 0.0

        self.omega = 0.0

        self.goal_x = None
        self.goal_y = None
        self.obstaculos_xy = []

        self.dt = 0.1

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

        # Variables extra para el control y seguridad
        self.front_distance = 3.5 
        self.v_max = 0.2  # Velocidad máxima en m/s
        self.K = np.array([0.0, 0.0]) # Se actualizará enseguida

        # 2. Creamos el Timer que ejecutará el control_loop cada dt (0.1s)
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("MPC node started.")

    def simulate_trajectory(self, start_x, start_y, start_theta, v, w, H, dt):
        """
        Simula el futuro del robot dado un comando (v, w).
        Retorna una lista de puntos (x, y, theta) que forman la trayectoria predicha.
        """
        trajectory = []
        
        # Copiamos nuestra posición actual para empezar a simular desde aquí
        sim_x = start_x
        sim_y = start_y
        sim_theta = start_theta
        
        # Repetimos la simulación por H pasos (Horizonte)
        for _ in range(H):
            # Integración de Euler (El f(xk, u) de tu diapositiva)
            sim_x = sim_x + v * math.cos(sim_theta) * dt
            sim_y = sim_y + v * math.sin(sim_theta) * dt
            sim_theta = sim_theta + w * dt
            
            # Guardamos la foto de este instante futuro en nuestra lista
            trajectory.append((sim_x, sim_y, sim_theta))
            
        return trajectory
    
    def evaluate_trajectory(self, trajectory, w, goal_x, goal_y, obstaculos_xy):
        """
        Califica una trayectoria. El score más BAJO gana.
        obstaculos_xy es una lista de puntos (x, y) de lo que ve el LIDAR.
        """
        # --- PESOS (Las letras griegas de tu profe) ---
        alpha = 2.0   # Peso para llegar a la meta
        beta = 0.2    # Peso para los giros (bajito para que no le importe tanto girar)
        gamma = 0.7   # Peso para mantener distancia de seguridad
        
        # 1. EL DESEO DE LLEGAR (Distancia a la meta)
        dist_to_goal = min([math.hypot(goal_x - x, goal_y - y) for x, y, _ in trajectory])
        costo_meta = alpha * dist_to_goal
        
        # 2. LA FOBIA A LOS MAREOS (Penalizar giros bruscos)
        costo_giro = beta * abs(w) 
        
        # 3. EL INSTINTO DE SUPERVIVENCIA (Distancia a obstáculos)
        costo_choque = 0.0
        distancia_segura = 0.20 # 35 centímetros
        
        # Buscamos la distancia más corta entre la trayectoria y los obstáculos
        min_dist_obstaculo = float('inf')
        for traj_x, traj_y, _ in trajectory:
            for obs_x, obs_y in obstaculos_xy:
                dist = math.hypot(traj_x - obs_x, traj_y - obs_y)
                if dist < min_dist_obstaculo:
                    min_dist_obstaculo = dist
                    
        # Aplicamos la penalización
        if min_dist_obstaculo < distancia_segura:
            return float('inf') # ¡Descartada por choque inminente!
        elif min_dist_obstaculo < distancia_segura + 0.3:
            # Si está cerca pero no choca, sumamos penalización (entre más cerca, mayor costo)
            costo_choque = gamma * (1.0 / min_dist_obstaculo)
            
        # --- SCORE FINAL ---
        score_total = costo_meta + costo_giro + costo_choque
        return score_total
    
    def control_loop(self):
        # Si no hay meta o no sabemos dónde estamos, no hacemos nada
        if self.goal_x is None or self.pose_x is None:
            return

        # --- 1. EL MENÚ DE CANDIDATOS (v, w) ---
        # Tal como en la diapositiva, definimos nuestras opciones. 
        # Puedes tunear esto para tu TurtleBot.
        candidatos = [
            (0.15, 0.0),   # Avanzar recto
            (0.05, 0.5),   # Avanzar lento girando izq
            (0.10, -0.5),  # Avanzar lento girando der
            (0.20, 0.3),   # Avanzar rápido girando izq suave
            (0.20, -0.3),  # Avanzar rápido girando der suave
            (0.0,  0.8),   # Girar en su propio eje a la izq (útil si está atascado)
            (0.0, -0.8),   # Girar en su propio eje a la der
        ]

        H = 10       # Horizonte de predicción (10 pasos)
        dt = 0.1     # Delta de tiempo de cada paso (0.1 segundos)
        # Esto significa que estamos prediciendo 1 segundo hacia el futuro (10 * 0.1)

        mejor_score = float('inf')
        mejor_comando = (0.0, 0.0) # Comando por defecto (detenido)

        # --- 2. EL CICLO DE EVALUACIÓN ---
        for v, w in candidatos:
            # Paso A: Simular el futuro (Tu primera función)
            trayectoria = self.simulate_trajectory(
                self.pose_x, self.pose_y, self.theta, v, w, H, dt
            )
            
            # Paso B y C: Revisar seguridad y calificar (Tu segunda función)
            score = self.evaluate_trajectory(
                trayectoria, w, self.goal_x, self.goal_y, self.obstaculos_xy
            )
            
            # Paso D: ¿Es este el mejor futuro hasta ahora?
            if score < mejor_score:
                mejor_score = score
                mejor_comando = (v, w)

        # --- 3. EJECUTAR EL MEJOR COMANDO ---
        # Si el mejor score sigue siendo infinito, significa que TODAS las trayectorias chocan.
        if mejor_score == float('inf'):
            self.get_logger().warn("¡Todos los caminos llevan a un choque! Frenando.")
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # Publicamos la velocidad ganadora
            cmd = Twist()
            cmd.linear.x = mejor_comando[0]
            cmd.angular.z = mejor_comando[1]
            
        self.cmd_vel_pub.publish(cmd)
        
        # Opcional: Condición de llegada para detener el nodo
        dist_to_goal = math.hypot(self.goal_x - self.pose_x, self.goal_y - self.pose_y)
        if dist_to_goal < 0.15:
            self.get_logger().info("¡Llegada triunfal con Pre-MPC!")
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            self.goal_x = None


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
        self.obstaculos_xy = []
        # Asumimos que el LIDAR va de 0 a 359 grados
        for i, dist in enumerate(msg.ranges):
            # Filtramos basura y nos quedamos solo con lo que está a menos de 1.5m
            if 0.1 < dist < 1.5 and not math.isinf(dist) and not math.isnan(dist):
                angle_local = math.radians(i)
                # Sumamos el ángulo del robot para saber el ángulo global
                angle_global = self.theta + angle_local 
                
                # Convertimos polar a cartesiano (X, Y global)
                obs_x = self.pose_x + dist * math.cos(angle_global)
                obs_y = self.pose_y + dist * math.sin(angle_global)
                
                self.obstaculos_xy.append((obs_x, obs_y))


def main(args=None):

    rclpy.init(args=args)             # Initialize communication
    node = MPC_controller()      # Create node instance
    try:
        rclpy.spin(node)              # Keep node running
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()           # Destroy node
        rclpy.shutdown()              # Shutdown communication

if __name__ == '__main__':
    main()  # Run main only if script executed directly