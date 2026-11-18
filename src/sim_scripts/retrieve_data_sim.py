from metadrive.envs.metadrive_env import MetaDriveEnv

env = MetaDriveEnv(config={"use_render": False, "traffic_density": 0.1})
obs, info = env.reset()

for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    
    traffic_manager = env.engine.traffic_manager
    
    for vehicle_id, vehicle in traffic_manager.spawned_objects.items():

        if hasattr(vehicle, "position"):
            # position [x, y]
            position = vehicle.position
            # current speed
            speed = vehicle.speed
            # velocity [vx, vy]
            velocity = vehicle.velocity
            # heading (orientation)
            heading = vehicle.heading
            
            print(f"Car {vehicle_id}: Pos={position}, Speed={speed}, Velocity={velocity}, Heading={heading}")
            
    if terminated or truncated:
        obs, info = env.reset()

env.close()
