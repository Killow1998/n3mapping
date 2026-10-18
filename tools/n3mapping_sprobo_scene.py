#!/usr/bin/env python3
"""Generate a 100 x 100 m indoor SpRobo session, without editing its workspace."""
import argparse
import json
import math
from pathlib import Path


def room_layout(variant="reference"):
    """Geometry and acquisition share door/room coordinates, not pose truth."""
    rows = (
        ((-40, -26, -12, 12, 28, 42), (-35, -33, -36, -34, -35, -33),
         (14, 10, 14, 12, 16, 10), (12, 10, 14, 10, 14, 8)),
        ((-41, -27, -13, 12, 27, 41), (-11, -9, -11, -10, -12, -9),
         (12, 15, 10, 14, 12, 14), (10, 12, 8, 12, 14, 8)),
        ((-40, -24, -10, 12, 27, 42), (10, 12, 9, 11, 9, 12),
         (14, 12, 10, 14, 12, 10), (12, 8, 10, 12, 8, 14)),
        ((-41, -27, -12, 12, 28, 42), (33, 35, 34, 36, 33, 35),
         (12, 14, 12, 14, 16, 10), (10, 14, 12, 8, 12, 14)),
    )
    rooms = []
    for row, (xs, ys, widths, depths) in enumerate(rows):
        for col, (x, y, width, depth) in enumerate(zip(xs, ys, widths, depths)):
            repetitive = variant == "repetitive"
            if repetitive:
                x, y, width, depth = (-40, -28, -16, 16, 28, 40)[col], (-34, -10, 10, 34)[row], 12, 12
            offset = 0 if repetitive else (-2, 1.5, 0, 2, -1.5, 1)[(row+col) % 6]
            door_width = 4 if repetitive else (3, 4, 5)[col % 3]
            front = y + (depth/2 if y < 0 else -depth/2)
            rooms.append(dict(row=row, col=col, x=x, y=y, width=width, depth=depth,
                              front=front, door_x=x+offset, door_width=door_width,
                              shape="rectangle" if repetitive else ("rectangle", "chamfer", "l")[col % 3],
                              furniture="office" if repetitive else
                              ("office", "shelves", "meeting", "pillars")[(row+col) % 4]))
    return rooms


def scene(variant="reference", spawn=(-44., 0., 0.), yaw=0.):
    result = dict(version=1, id="n3m-indoor-100-v2", name="N3Mapping indoor 100 x 100 m",
                  floors=[dict(id="ground", name="Ground", height=0)], objects=[],
                  spawn=dict(floor="ground", position=list(spawn), yaw_deg=yaw))

    def box(name, kind, position, size, yaw=0.):
        result["objects"].append(dict(id=name, name=name, floor="ground", kind=kind,
                                      position=position, size=size,
                                      quaternion=[math.cos(yaw/2), 0, 0, math.sin(yaw/2)]))

    box("floor", "floor", [0, 0, -.1], [100, 100, .2])
    box("ceiling", "floor", [0, 0, 3.6], [100, 100, .2])
    for sign in (-1, 1):
        box(f"outer-x-{sign}", "wall", [sign*49.9, 0, 1.75], [.2, 100, 3.5])
        box(f"outer-y-{sign}", "wall", [0, sign*49.9, 1.75], [100, .2, 3.5])
    # Continuous cross-corridors at y=0,+/-24 and x=+/-48. Shapes, doorway
    # offsets and useful interior surfaces vary; local repeated structures remain.
    for room in room_layout(variant):
        x, y, w, d = (room[k] for k in ("x", "y", "width", "depth"))
        tag = f"{room['row']}-{room['col']}"
        away = -1 if y < 0 else 1

        def xy(u, v):
            return x+u, room["front"]+away*v

        def wall(a, b, name):
            a, b = xy(*a), xy(*b)
            length = math.dist(a, b)
            box(f"wall-{tag}-{name}", "wall", [(a[0]+b[0])/2, (a[1]+b[1])/2, 1.75],
                [length, .2, 3.5], math.atan2(b[1]-a[1], b[0]-a[0]))

        vertices = [(-w/2, 0), (-w/2, d)]
        cut = min(w, d)*.25
        if room["shape"] == "l":
            vertices += [(w/2-cut, d), (w/2-cut, d-cut), (w/2, d-cut)]
        elif room["shape"] == "chamfer":
            vertices += [(w/2-cut, d), (w/2, d-cut)]
        else:
            vertices += [(w/2, d)]
        vertices += [(w/2, 0)]
        for i, (a, b) in enumerate(zip(vertices, vertices[1:])):
            wall(a, b, str(i))
        offset, half_door = room["door_x"]-x, room["door_width"]/2
        wall((-w/2, 0), (offset-half_door, 0), "front-left")
        wall((offset+half_door, 0), (w/2, 0), "front-right")

        def item(name, u, v, width, depth, height):
            px, py = xy(u, v)
            box(f"{name}-{tag}", "obstacle", [px, py, height/2], [width, depth, height])

        style = room["furniture"]
        if style == "shelves":
            for i, v in enumerate((.62, .77, .90)):
                item(f"shelf-{i}", -.25*w, v*d, .40*w, .6, 1.4+.4*i)
        elif style == "meeting":
            item("table", -.08*w, .75*d, .45*w, .20*d, .8)
            for i, u in enumerate((-.28*w, .12*w)):
                item(f"chair-{i}", u, .58*d, .5, .5, .9)
        elif style == "pillars":
            item("column-a", -.32*w, .72*d, .9, .9, 3.5)
            item("column-b", .32*w, .32*d, 1.3, .8, 3.5)
            item("workbench", -.25*w, .40*d, 2, 1, 1.1)
        else:
            shifted = variant == "changed" and (room["row"]+room["col"]) % 3 == 0
            item("desk", (-.15 if shifted else -.3)*w, .68*d, 2, 1, .8)
            item("cabinet", -.34*w, .87*d, 1, .6, 2)
        if variant == "changed" and (room["row"]+room["col"]) % 3 == 0:
            item("moved-cart", .30*w, .60*d, 1.5, .6, 1.2)
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
