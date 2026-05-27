#include "kalman_tracker.hpp"

#include <stdexcept>

namespace hw
{
    KalmanTracker::KalmanTracker() = default;

    bool KalmanTracker::isTracking() const
    {
        return tracking_;
    }

    void KalmanTracker::reset()
    {
        tracking_ = false;
        x_ = AxisFilter{};
        y_ = AxisFilter{};
        z_ = AxisFilter{};
    }

    void KalmanTracker::AxisFilter::reset(double measured_position)
    {
        position = measured_position;
        velocity = 0.0;
        p00 = 1.0;
        p01 = 0.0;
        p10 = 0.0;
        p11 = 1.0;
    }

    void KalmanTracker::AxisFilter::predict(double dt, double process_noise)
    {
        // TODO(student): Implement the constant-velocity Kalman predict step.
        // dt = max(dt, 0)
        // position = position + velocity * dt
        // F = [[1, dt],
        //      [0, 1]]
        // Q = process_noise * [[dt^4 / 4, dt^3 / 2],
        //                      [dt^3 / 2, dt^2]]
        // P = F * P * F^T + Q
        // store the updated position, velocity, and covariance
        dt = dt > 0.0 ? dt : 0.0;
        position = position + velocity * dt;
        double dt2 = dt * dt;
        double dt3 = dt2 * dt;
        double dt4 = dt3 * dt;
        double q00 = process_noise * (dt4 / 4.0);
        double q01 = process_noise * (dt3 / 2.0);
        double q10 = process_noise * (dt3 / 2.0);
        double q11 = process_noise * dt2;
        double next_p00 = p00 + dt * p10 + dt * p01 + dt2 * p11 + q00;
        double next_p01 = p01 + dt * p11 + q01;
        double next_p10 = p10 + dt * p11 + q10;
        double next_p11 = p11 + q11;
        p00 = next_p00;
        p01 = next_p01;
        p10 = next_p10;
        p11 = next_p11;
    }

    void KalmanTracker::AxisFilter::update(double measured_position, double measurement_noise)
    {
        // TODO(student): Implement the 1D position measurement update step.
        // residual = measured_position - position
        // H = [1, 0]
        // S = H * P * H^T + measurement_noise
        // if S is not positive:
        //     return without updating
        // K = P * H^T / S
        // position = position + K[0] * residual
        // velocity = velocity + K[1] * residual
        // P = (I - K * H) * P
        double residual = measured_position - position;
        double s = p00 + measurement_noise;  
        if (s <= 0.0)
        {
            return;
        }
        double k0 = p00 / s;
        double k1 = p10 / s;
        position = position + k0 * residual;
        velocity = velocity + k1 * residual;
        double next_p00 = (1.0 - k0) * p00;
        double next_p01 = (1.0 - k0) * p01;
        double next_p10 = -k1 * p00 + p10;
        double next_p11 = -k1 * p01 + p11;
        p00 = next_p00;
        p01 = next_p01;
        p10 = next_p10;
        p11 = next_p11;
    }

    TrackState KalmanTracker::update(const Vec3 &measurement, double dt)
    {
        // TODO(student): Update tracker state from one measured 3D point.
        // if tracker is not initialized:
        //     initialize x, y, z filters with measurement components
        //     set all velocities to zero
        //     mark tracker as active
        //     return current state
        // predict each axis filter using dt
        // update each axis filter with its measured coordinate
        // return position, velocity, and tracking flag
        if (!tracking_)
        {
            x_.reset(measurement.x);
            y_.reset(measurement.y);
            z_.reset(measurement.z);
            tracking_ = true;
            return stateFromFilters();
        }
        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);
        x_.update(measurement.x, measurement_noise_);
        y_.update(measurement.y, measurement_noise_);
        z_.update(measurement.z, measurement_noise_);
        return stateFromFilters();
    }

    TrackState KalmanTracker::predict(double dt)
    {
        // TODO(student): Predict target state when a detection is missing.
        // if tracker is not active:
        //     return a non-tracking state
        // predict x, y, z filters with dt
        // return predicted position and velocity
        if (!tracking_)
        {
            return {false, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        }

        x_.predict(dt, process_noise_);
        y_.predict(dt, process_noise_);
        z_.predict(dt, process_noise_);

        return stateFromFilters();
    }
    TrackState KalmanTracker::stateFromFilters() const
    {
        return {
            true,
            {x_.position, y_.position, z_.position},
            {x_.velocity, y_.velocity, z_.velocity},
        };
    }
} // namespace hw
