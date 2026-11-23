import numpy as np


def generate_quintic_spline_waypoints(start, end, num_points):
    start = np.asarray(start)
    end = np.asarray(end)
    s = np.linspace(0.0, 1.0, num_points)
    time_scale = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    return start + np.expand_dims(time_scale, axis=-1) * (end - start)


def pid_control(tracking_error_history, timestep, Kp=150.0, Ki=0.0, Kd=0.01):
    errors = np.asarray(tracking_error_history)
    if errors.shape[0] == 0:
        raise ValueError("tracking_error_history must contain at least one error")
    if timestep <= 0:
        raise ValueError("timestep must be positive")
    proportional = errors[-1]
    integral = errors.sum(axis=0) * timestep
    derivative = (
        (errors[-1] - errors[-2]) / timestep
        if errors.shape[0] > 1
        else np.zeros_like(proportional)
    )
    return Kp * proportional + Ki * integral + Kd * derivative
