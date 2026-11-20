"""Project limb directions onto the calibrated white-model flexion planes."""

import math

from mathutils import Quaternion, Vector


LENGTH_AXIS = Vector((0, 1, 0))


def flex_axis(upper_rest, lower_rest):
    """The rest bend calibrates the positive direction of the elbow or knee."""
    local = upper_rest.inverted() @ (lower_rest @ LENGTH_AXIS)
    return Vector((local.x, 0, local.z)).normalized()


def hinge_direction(upper, target, axis, max_flex, side_limit):
    """Keep the distal segment on the positive side of its hinge."""
    local = upper.inverted() @ target
    radial = local - LENGTH_AXIS * local.y
    if radial.length < 1e-7:
        return upper @ LENGTH_AXIS
    transverse = radial.normalized()
    support = axis.dot(transverse)
    if support <= 0.0:
        return upper @ LENGTH_AXIS
    side = math.atan2(LENGTH_AXIS.dot(axis.cross(transverse)), support)
    side = max(-side_limit, min(side_limit, side))
    flex = min(max_flex, math.atan2(radial.length, local.y))
    # Fade uncertain near-perpendicular bends towards extension rather than
    # inventing a strong bend around the wrong plane.
    flex *= min(1.0, support / 0.55)
    tangent = Quaternion(LENGTH_AXIS, side) @ axis
    return (upper @ (LENGTH_AXIS * math.cos(flex) + tangent * math.sin(flex))).normalized()
