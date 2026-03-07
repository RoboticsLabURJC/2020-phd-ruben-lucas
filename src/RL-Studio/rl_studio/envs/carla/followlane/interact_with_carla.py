import carla
import time


def main():
    try:
        # Connect to your specific port
        client = carla.Client('localhost', 5013)
        # No timeout as requested

        world = client.get_world()
        print("Connected to 5013. Attempting to kill shadows...")

        # Method 1: The most common for 0.9.12+
        try:
            client.send_console_command("r.ShadowQuality 0")
            client.send_console_command("r.Shadow.MaxResolution 4")
            print("Successfully sent via Client.")
        except AttributeError:
            # Method 2: For older versions (0.9.10 and below)
            try:
                world.get_settings().no_rendering_mode = False  # Ensure rendering is on
                # Some versions require calling it through the internal exec
                client.get_world().send_console_command("r.ShadowQuality 0")
                print("Successfully sent via World (Old API).")
            except AttributeError:
                print("Could not find send_console_command. Trying Weather fallback...")

        # Method 3: The "Noon Sun" trick (Works in ALL versions)
        # This is the most reliable way to hide shadows if commands fail
        weather = carla.WeatherParameters(
            sun_altitude_angle=90.0,
            cloudiness=0.0,
            precipitation=0.0,
            sun_azimuth_angle=0.0,
            fog_density=0.0
        )
        world.set_weather(weather)

        time.sleep(1)
        print("Environment update complete.")

    except Exception as e:
        print(f"Connection Error: {e}")


if __name__ == '__main__':
    main()