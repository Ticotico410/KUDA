import argparse
import os
import sys
import time

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from real_world.real_env import RealEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_fixed_cams', type=int, default=3)
    parser.add_argument('--exposure_time', type=int, default=5)
    parser.add_argument('--out_dir', default='fixed_cam_captures')
    args = parser.parse_args()

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

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    out_dir = os.path.join(args.out_dir, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    try:
        env.start(exposure_time=args.exposure_time)
        time.sleep(args.exposure_time)
        obs = env.get_obs(get_color=True, get_depth=False)

        serials = env.serial_numbers[:args.num_fixed_cams]
        with open(os.path.join(out_dir, 'serials.txt'), 'w') as f:
            f.write('\n'.join(serials))

        for i, serial in enumerate(serials):
            frame = obs[f'color_{i}'][-1]
            filename = os.path.join(out_dir, f'cam_{i}_{serial}.png')
            cv2.imwrite(filename, frame)
    finally:
        env.stop()


if __name__ == '__main__':
    main()
