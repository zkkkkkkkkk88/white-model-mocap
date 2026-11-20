"""Estimate torso lift from the walking rhythm while rejecting spike and gap edges."""

from bisect import bisect_left, bisect_right
import math
from statistics import median


def stable_translation(values, fps):
    """Suppress one-frame displacements in rig space without removing a step."""
    radius = round(0.09 * float(fps))
    if radius < 1 or len(values) < 3:
        return values
    weights = [0.5 + 0.5 * math.cos(math.pi * offset / (radius + 1))
               for offset in range(radius + 1)]
    output = []
    for index, value in enumerate(values):
        combined = value * 0
        total = 0.0
        for sample in range(max(0, index - radius), min(len(values), index + radius + 1)):
            weight = weights[abs(sample - index)]
            combined += weight * values[sample]
            total += weight
        output.append(combined / total)
    return output


def _limit_amplitude(cosine, sine, limit):
    amplitude = math.hypot(cosine, sine)
    if amplitude <= limit:
        return cosine, sine
    return cosine * limit / amplitude, sine * limit / amplitude


def stabilize_walk_lift(lift, clock, supported, leg_span, measured):
    """Keep a source-supported walking bob and prolong it through missing observations."""
    limit = 0.021 * leg_span
    if not measured:
        return [-0.015 * leg_span * math.cos(math.tau * phase) for phase in clock]

    visible = [value for value, valid in zip(lift, supported, strict=True) if valid]
    center = median(visible)
    spread = median(abs(value - center) for value in visible)
    cap = max(0.006 * leg_span, 2.8 * spread)
    # The phase clock comes from the source. Fit the first harmonic in that
    # coordinate system so neither a tracking spike nor a dropout supplies a step.
    basis = [(math.cos(math.tau * phase), math.sin(math.tau * phase)) for phase in clock]
    centered = [max(-cap, min(cap, value - center)) if valid else 0.0
                for value, valid in zip(lift, supported, strict=True)]
    first, second = _limit_amplitude(
        2 * sum(value * cosine for value, (cosine, _) in zip(centered, basis, strict=True)) / len(visible),
        2 * sum(value * sine for value, (_, sine) in zip(centered, basis, strict=True)) / len(visible),
        limit,
    )

    result = []
    radius = 2.4
    for i, phase in enumerate(clock):
        left, right = bisect_left(clock, phase - radius), bisect_right(clock, phase + radius)
        total, cosine_sum, sine_sum = 0.0, 0.0, 0.0
        for j in range(left, right):
            if supported[j]:
                weight = (1 - abs(clock[j] - phase) / radius) ** 2
                total += weight
                cosine, sine = basis[j]
                cosine_sum += weight * centered[j] * cosine
                sine_sum += weight * centered[j] * sine
        local_mix = 0.43 * min(1.0, total / max(1, (right - left) * 0.27))
        if total > 0:
            local_first, local_second = _limit_amplitude(2 * cosine_sum / total, 2 * sine_sum / total, limit)
            a = first + local_mix * (local_first - first)
            b = second + local_mix * (local_second - second)
        else:
            a, b = first, second
        cosine, sine = basis[i]
        result.append(center + a * cosine + b * sine)
    return result


def smooth_maximum(a, b, softness):
    """A differentiable upper bound that keeps support within reach."""
    return (a + b + math.hypot(a - b, softness)) / 2
