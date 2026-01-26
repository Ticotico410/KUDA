import argparse
import os
import sys
import time

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
XARM_CALIBRATE_DIR = os.path.join(ROOT_DIR, 'xarm-calibrate')
if XARM_CALIBRATE_DIR not in sys.path:
    sys.path.insert(0, XARM_CALIBRATE_DIR)

from real_world.real_env import RealEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_fixed_cams', type=int, default=4)
    parser.add_argument('--exposure_time', type=int, default=5)
    parser.add_argument('--out_dir', default='cameras_viz')
    args = parser.parse_args()

    calibration_result_dir = os.path.join(os.getcwd(), 'real_world', 'calibration_result')
    had_calibration_result_dir = os.path.isdir(calibration_result_dir)

    env = RealEnv(
        WH=[1280, 720],
        capture_fps=15,
        obs_fps=15,
        n_obs_steps=1,
        enable_color=True,
        enable_depth=False,
        process_depth=False,
        use_robot=False,
        use_wrist_cam=False,
        num_fixed_cams=args.num_fixed_cams,
        gripper_enable=False,
        verbose=False,
    )
    if not had_calibration_result_dir and os.path.isdir(calibration_result_dir):
        try:
            if not os.listdir(calibration_result_dir):
                os.rmdir(calibration_result_dir)
                parent_dir = os.path.dirname(calibration_result_dir)
                if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
        except OSError:
            pass
    env.calibrate_result_dir = None

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    try:
        env.start(exposure_time=args.exposure_time)
        time.sleep(args.exposure_time)
        obs = env.get_obs(get_color=True, get_depth=False)

        serials = env.serial_numbers[:args.num_fixed_cams]
        # with open(os.path.join(out_dir, 'serials.txt'), 'w') as f:
        #     f.write('\n'.join(serials))

        for i, serial in enumerate(serials):
            frame = obs[f'color_{i}'][-1]
            filename = os.path.join(out_dir, f'cam_{i}_{serial}.png')
            cv2.imwrite(filename, frame)
    finally:
        env.stop()


if __name__ == '__main__':
    main()
