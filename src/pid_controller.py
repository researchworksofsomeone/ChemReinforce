import numpy as np

class PIDController:
    def __init__(self, Kp=2.0, Ki=0.01, Kd=0.1, setpoint=350.0, output_limits=(250.0, 400.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, current_value, dt=0.1):
        # Reverse acting: If T is too high, we need colder coolant, so T_c must decrease.
        # Thus error should be setpoint - current_value.
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative + 350.0
        
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        self.prev_error = error
        return output
