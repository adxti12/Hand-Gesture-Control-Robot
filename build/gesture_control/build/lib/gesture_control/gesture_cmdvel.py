import cv2
import mediapipe as mp
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


def classify_gesture(landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    fingers_up = []

    # Thumb
    if landmarks[thumb_tip].x < landmarks[thumb_tip - 1].x:
        fingers_up.append(1)
    else:
        fingers_up.append(0)

    # Other fingers
    for tip in finger_tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

    if sum(fingers_up) == 0:
        return "fist"
    elif sum(fingers_up) == 5:
        return "palm"
    elif fingers_up[1] == 1 and sum(fingers_up) == 1:
        return "point"
    else:
        return "unknown"


def get_point_direction(landmarks):
    base = landmarks[5]
    tip = landmarks[8]
    dx = tip.x - base.x
    dy = tip.y - base.y
    angle = math.degrees(math.atan2(-dy, dx))  # invert y-axis
    if -45 <= angle <= 45:
        return "right"
    elif 45 < angle <= 135:
        return "up"
    elif -135 <= angle < -45:
        return "down"
    else:
        return "left"


class GestureCmdVel(Node):
    def __init__(self):
        super().__init__("gesture_cmd_vel")
        self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
        self.speed_scale = 0.2  # start speed
        self.angular_scale = 0.5

    def send_cmd(self, gesture, direction=""):
        twist = Twist()

        if gesture == "point":
            if direction == "up":
                twist.linear.x = self.speed_scale
            elif direction == "left":
                twist.angular.z = self.angular_scale
            elif direction == "right":
                twist.angular.z = -self.angular_scale
            # (ignore "down")

        elif gesture == "fist":
            # increase velocity by 5%
            self.speed_scale *= 1.05
            self.angular_scale *= 1.05
            self.get_logger().info(f"Speed boosted: {self.speed_scale:.2f}, {self.angular_scale:.2f}")

        elif gesture == "palm":
            # decrease velocity by 5%
            self.speed_scale *= 0.95
            self.angular_scale *= 0.95
            self.get_logger().info(f"Speed reduced: {self.speed_scale:.2f}, {self.angular_scale:.2f}")

        self.publisher_.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = GestureCmdVel()

    cap = cv2.VideoCapture(2)

    while rclpy.ok():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture = "none"
        direction = ""

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = classify_gesture(hand_landmarks.landmark)
                if gesture == "point":
                    direction = get_point_direction(hand_landmarks.landmark)
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            node.send_cmd(gesture, direction)

        cv2.putText(frame, f"Gesture: {gesture} {direction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
