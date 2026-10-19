"""Scene-specific acquisition route; truth drives the fixture, not the estimator."""
import math
from n3mapping_sprobo_scene import room_layout


def coverage_route(variant="reference"):
    points = [(-48, 0), (-48, -47), (48, -47), (48, -24)]

    layout = room_layout(variant)

    def rooms(corridor, rows, reverse=False):
        selected = sorted((r for r in layout if r["row"] in rows),
                          key=lambda r: (-r["col"] if reverse else r["col"], r["row"]))
        for room in selected:
            entry = (room["door_x"], room["front"])
            points.extend(((room["door_x"], corridor), entry,
                           (room["x"], room["y"]), entry, (room["door_x"], corridor)))

    rooms(-24, (0,), reverse=True)
    points.extend(((-48, -24), (-48, 0)))
    rooms(0, (1, 2))
    points.extend(((48, 0), (48, 24)))
    rooms(24, (3,), reverse=True)
    points.extend(((-48, 24), (-48, 47), (48, 47), (48, 0), (-44, 0)))
    return points


class RouteDriver:
    def __init__(self, speed=1.2, variant="reference", points=None):
        self.points = coverage_route(variant) if points is None else points
        if (len(self.points) < 2 or any(len(point) != 2 or
                not all(math.isfinite(value) for value in point) for point in self.points)):
            raise ValueError("A route needs at least two finite XY waypoints")
        self.speed = speed
        self.index = 0
        self.pose = None
        self.stamp = None
        self.visits = []
        self.completed_at = None
        self.distance_m = 0.
        self.progress_index = -1
        self.best_distance = math.inf
        self.best_heading_error = math.inf
        self.last_progress = None

    def observe(self, message, stamp):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        if self.pose is not None:
            self.distance_m += math.hypot(p.x-self.pose[0], p.y-self.pose[1])
        self.pose, self.stamp = (p.x, p.y, yaw), stamp/1e9

    def command(self, now):
        if now < 7 or self.pose is None:
            return 0., 0.
        if now-self.stamp > .5:
            raise RuntimeError("Route driver lost simulation pose updates")
        x, y, yaw = self.pose
        while self.index < len(self.points):
            target = self.points[self.index]
            dx, dy = target[0]-x, target[1]-y
            distance = math.hypot(dx, dy)
            if distance > .15:
                break
            self.visits.append(dict(index=self.index, stamp_s=now, position=[x, y]))
            self.index += 1
        if self.index == len(self.points):
            if self.completed_at is None:
                self.completed_at = now
            return 0., 0.
        error = math.atan2(math.sin(math.atan2(dy, dx)-yaw), math.cos(math.atan2(dy, dx)-yaw))
        direction = 1.
        if abs(error) > math.pi/2:
            direction = -1.
            error = math.atan2(math.sin(error-math.pi), math.cos(error-math.pi))
        if self.progress_index != self.index:
            self.progress_index = self.index
            self.best_distance = distance
            self.best_heading_error = abs(error)
            self.last_progress = now
        if distance < self.best_distance-.1 or abs(error) < self.best_heading_error-.05:
            self.best_distance = min(self.best_distance, distance)
            self.best_heading_error = min(self.best_heading_error, abs(error))
            self.last_progress = now
        if now-self.last_progress > 30:
            raise RuntimeError(f"No progress toward waypoint {self.index} for 30 s")
        # Stop before corners and use reverse for room exits. This is scripted
        # acquisition, not a navigation-system or wheel-odometry evaluation.
        speed = min(self.speed, distance) if abs(error) < .25 else 0.
        return direction*speed, max(-.8, min(.8, 2*error))

    def report(self):
        return dict(waypoints=self.points, reached=self.index, total=len(self.points),
                    completed_at_s=self.completed_at, distance_m=self.distance_m, visits=self.visits,
                    truth_use="route controller and scoring only; no estimator input")
