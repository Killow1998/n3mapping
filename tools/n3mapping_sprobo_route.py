"""Scene-specific acquisition route; truth drives the fixture, not the estimator."""
import math


def coverage_route():
    points = [(-47, 0), (-47, -47), (47, -47), (47, -24)]

    def rooms(corridor, centers, room_rows):
        for x in centers:
            points.append((x, corridor))
            for y in room_rows:
                points.extend(((x, y), (x, corridor)))

    columns = (-40, -28, -16, 16, 28, 40)
    rooms(-24, reversed(columns), (-34,))
    points.extend(((-47, -24), (-47, 0)))
    rooms(0, columns, (-10, 10))
    points.extend(((47, 0), (47, 24)))
    rooms(24, reversed(columns), (34,))
    points.extend(((-47, 24), (-47, 47), (47, 47), (47, 0), (-44, 0)))
    return points


class RouteDriver:
    def __init__(self, speed=1.2):
        self.points = coverage_route()
        self.speed = speed
        self.index = 0
        self.pose = None
        self.stamp = None
        self.visits = []
        self.completed_at = None

    def observe(self, message, stamp):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
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
        # Stop before corners and use reverse for room exits. This is scripted
        # acquisition, not a navigation-system or wheel-odometry evaluation.
        speed = min(self.speed, distance) if abs(error) < .25 else 0.
        return direction*speed, max(-.8, min(.8, 2*error))

    def report(self):
        return dict(waypoints=self.points, reached=self.index, total=len(self.points),
                    completed_at_s=self.completed_at, visits=self.visits,
                    truth_use="route controller and scoring only; no estimator input")
