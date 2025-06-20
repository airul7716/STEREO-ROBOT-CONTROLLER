# All fingers are fully open (straight)
if all(a == 0 for a in finger_angles):
    steps = [9] * 6  # ➡️ Set all servos to neutral (90° mapped to step 9)

else:
    steps = [
        to_step(finger_angles[0], reverse=True),   # Thumb
        to_step(finger_angles[1], reverse=True),   # Index
        to_step(finger_angles[2], reverse=True),   # Middle
        to_step(finger_angles[3], reverse=True),   # Ring
        to_step(finger_angles[4], reverse=True),   # Pinky
        to_step(wrist_roll, reverse=False)         # Wrist
    ]