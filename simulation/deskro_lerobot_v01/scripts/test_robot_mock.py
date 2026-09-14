from lerobot_robot_deskro import Deskro, DeskroConfig

config = DeskroConfig(mock=True, id="deskro_mock")
with Deskro(config) as robot:
    print("connected:", robot.is_connected)
    print("before:", robot.get_observation())
    sent = robot.send_action(
        {
            "head_yaw.target_deg": 30.0,
            "head_pitch.target_deg": -10.0,
            "face.state": 2.0,
        }
    )
    print("sent:", sent)
    print("after:", robot.get_observation())
