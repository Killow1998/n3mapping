#!/usr/bin/env python3
"""Generate a 100 x 100 m indoor SpRobo session, without editing its workspace."""
import argparse
import json
from pathlib import Path


def scene(variant="reference", spawn=(-44., 0., 0.), yaw=0.):
    result = dict(version=1, id="n3m-indoor-100", name="N3Mapping indoor 100 x 100 m",
                  floors=[dict(id="ground", name="Ground", height=0)], objects=[],
                  spawn=dict(floor="ground", position=list(spawn), yaw_deg=yaw))

    def box(name, kind, position, size):
        result["objects"].append(dict(id=name, name=name, floor="ground", kind=kind,
                                      position=position, size=size, quaternion=[1, 0, 0, 0]))

    box("floor", "floor", [0, 0, -.1], [100, 100, .2])
    box("ceiling", "floor", [0, 0, 3.6], [100, 100, .2])
    for sign in (-1, 1):
        box(f"outer-x-{sign}", "wall", [sign*49.9, 0, 1.75], [.2, 100, 3.5])
        box(f"outer-y-{sign}", "wall", [0, sign*49.9, 1.75], [100, .2, 3.5])
    # Four rows of repeated offices. Continuous cross-corridors at y=0,+/-24,
    # x=0,+/-48 allow separate mapping/query routes; room doors are 4 m wide.
    for row, y in enumerate((-34, -10, 10, 34)):
        front = y + (6 if y < 0 else -6)
        back = y - (6 if y < 0 else -6)
        for col, x in enumerate((-40, -28, -16, 16, 28, 40)):
            tag = f"{row}-{col}"
            box(f"back-{tag}", "wall", [x, back, 1.75], [12, .2, 3.5])
            for sign in (-1, 1):
                box(f"side-{tag}-{sign}", "wall", [x+sign*5.9, y, 1.75], [.2, 12, 3.5])
                box(f"front-{tag}-{sign}", "wall", [x+sign*4, front, 1.75], [4, .2, 3.5])
            shifted = variant == "changed" and (row+col) % 3 == 0
            box(f"desk-{tag}", "obstacle", [x+(2 if shifted else -2), y, .4], [2, 1, .8])
            box(f"cabinet-{tag}", "obstacle", [x+3, back+(1 if y > 0 else -1), 1], [1, .6, 2])
            if shifted:
                box(f"door-obstruction-{tag}", "obstacle", [x+1.25, front, 1.2], [1.5, .5, 2.4])
    # Repetitive variant removes distinguishing geometry, not competing rooms.
    if variant != "repetitive":
        for i, (x, y, height) in enumerate(((-35, 2.7, 1.2), (-9, -2.7, 2.2),
                                            (23, 2.7, 1.7), (3, 22, 2.5))):
            box(f"landmark-{i}", "obstacle", [x, y, height/2], [1+i*.2, .7, height])
    return result


def session(variant="reference", spawn=(-44., 0., 0.), yaw=0., duration=35.):
    return dict(robot="scout_mini", controller="auto", control_mode="tracked",
                backend="cpu", rays=20000, settle_s=2., duration_s=duration,
                sensors=dict(mid360=True, camera=False, depth=False, panorama=False),
                adapters=dict(ros1=False, ros2=True),
                ros=dict(publish_odom_tf=False, odom_topic="/ground_truth/odom"),
                mounts=dict(mid360=dict(position=[0, 0, .5], quaternion=[1, 0, 0, 0])),
                source="ros2", start_immediately=False,
                scene=scene(variant, spawn, yaw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["reference", "changed", "repetitive"], default="reference")
    parser.add_argument("--spawn", nargs=3, type=float, default=[-44, 0, 0])
    parser.add_argument("--yaw", type=float, default=0.)
    parser.add_argument("--duration", type=float, default=35.)
    args = parser.parse_args()
    config = session(args.variant, args.spawn, args.yaw, args.duration)
    with args.output.open("x") as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
    print(f"{len(config['scene']['objects'])} objects; {args.output}")


if __name__ == "__main__":
    main()
