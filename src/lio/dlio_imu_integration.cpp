#include "n3mapping/lio/dlio_imu_integration.h"

#include <cstddef>

namespace n3mapping {
namespace lio {
namespace dlio {
namespace {

double sampleDt(const std::vector<ImuMeasurement>& samples, size_t index) {
    if (index == 0) {
        return samples[index].dt;
    }
    return samples[index].dt > 0.0
               ? samples[index].dt
               : samples[index].stamp - samples[index - 1].stamp;
}

Eigen::Quaternionf integrateQuaternion(const Eigen::Quaternionf& q,
                                       const Eigen::Vector3f& omega,
                                       double dt) {
    Eigen::Quaternionf integrated(
        q.w() - 0.5f *
                    (q.x() * omega[0] + q.y() * omega[1] + q.z() * omega[2]) *
                    static_cast<float>(dt),
        q.x() + 0.5f *
                    (q.w() * omega[0] - q.z() * omega[1] + q.y() * omega[2]) *
                    static_cast<float>(dt),
        q.y() + 0.5f *
                    (q.z() * omega[0] + q.w() * omega[1] - q.x() * omega[2]) *
                    static_cast<float>(dt),
        q.z() + 0.5f *
                    (q.x() * omega[1] - q.y() * omega[0] + q.w() * omega[2]) *
                    static_cast<float>(dt));
    integrated.normalize();
    return integrated;
}

Eigen::Vector3f worldAcceleration(const Eigen::Quaternionf& q,
                                  const Eigen::Vector3f& body_accel,
                                  double gravity) {
    Eigen::Vector3f accel = q._transformVector(body_accel);
    accel[2] -= static_cast<float>(gravity);
    return accel;
}

bool findIntegrationWindow(const std::vector<ImuMeasurement>& samples,
                           const ImuIntegrationRequest& request,
                           size_t& begin_index,
                           size_t& end_index) {
    if (samples.size() < 2 || request.sorted_timestamps.empty() ||
        request.start_time > request.sorted_timestamps.front()) {
        return false;
    }

    bool found_begin = false;
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        if (samples[i].stamp <= request.start_time &&
            samples[i + 1].stamp >= request.start_time) {
            begin_index = i;
            found_begin = true;
            break;
        }
    }
    if (!found_begin) {
        return false;
    }

    const double end_time = request.sorted_timestamps.back();
    for (size_t i = begin_index + 1; i < samples.size(); ++i) {
        if (samples[i].stamp >= end_time) {
            end_index = i;
            return true;
        }
    }
    return false;
}

}  // namespace

std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>
integrateImuStates(const std::vector<ImuMeasurement>& imu_samples,
                   const ImuIntegrationRequest& request) {
    std::vector<IntegratedPose, Eigen::aligned_allocator<IntegratedPose>>
        states;

    size_t begin_index = 0;
    size_t end_index = 0;
    if (!findIntegrationWindow(imu_samples, request, begin_index, end_index)) {
        return states;
    }

    const auto& f1 = imu_samples[begin_index];
    const auto& f2 = imu_samples[begin_index + 1];
    const double dt = sampleDt(imu_samples, begin_index + 1);
    if (dt <= 0.0) {
        return states;
    }

    const double idt = request.start_time - f1.stamp;
    const Eigen::Vector3f alpha_dt =
        f2.angular_velocity - f1.angular_velocity;
    const Eigen::Vector3f alpha = alpha_dt / static_cast<float>(dt);
    const Eigen::Vector3f omega_i =
        -(f1.angular_velocity + 0.5f * alpha * static_cast<float>(idt));

    Eigen::Quaternionf q_init =
        integrateQuaternion(request.q_init, omega_i, idt);
    const Eigen::Vector3f omega = f1.angular_velocity + 0.5f * alpha_dt;
    const Eigen::Quaternionf q2 = integrateQuaternion(q_init, omega, dt);
    const Eigen::Vector3f a1 =
        worldAcceleration(q_init, f1.linear_acceleration, request.gravity);
    const Eigen::Vector3f a2 =
        worldAcceleration(q2, f2.linear_acceleration, request.gravity);
    const Eigen::Vector3f jerk = (a2 - a1) / static_cast<float>(dt);

    Eigen::Vector3f v =
        request.v_init - a1 * static_cast<float>(idt) -
        0.5f * jerk * static_cast<float>(idt * idt);
    Eigen::Vector3f p =
        request.p_init - v * static_cast<float>(idt) -
        0.5f * a1 * static_cast<float>(idt * idt) -
        (1.0f / 6.0f) * jerk * static_cast<float>(idt * idt * idt);

    Eigen::Quaternionf q = q_init;
    Eigen::Vector3f a = a1;
    auto stamp_it = request.sorted_timestamps.begin();

    for (size_t i = begin_index + 1; i <= end_index; ++i) {
        const auto& f0 = imu_samples[i - 1];
        const auto& f = imu_samples[i];
        const double step_dt = sampleDt(imu_samples, i);
        if (step_dt <= 0.0) {
            return {};
        }

        const Eigen::Vector3f step_alpha_dt =
            f.angular_velocity - f0.angular_velocity;
        const Eigen::Vector3f step_alpha =
            step_alpha_dt / static_cast<float>(step_dt);
        const Eigen::Vector3f step_omega =
            f0.angular_velocity + 0.5f * step_alpha_dt;

        const Eigen::Quaternionf q0 = q;
        q = integrateQuaternion(q, step_omega, step_dt);
        const Eigen::Vector3f a0 = a;
        a = worldAcceleration(q, f.linear_acceleration, request.gravity);
        const Eigen::Vector3f jerk_dt = a - a0;
        const Eigen::Vector3f step_jerk =
            jerk_dt / static_cast<float>(step_dt);

        while (stamp_it != request.sorted_timestamps.end() &&
               *stamp_it <= f.stamp) {
            const double interp_dt = *stamp_it - f0.stamp;
            const Eigen::Vector3f interp_omega =
                f0.angular_velocity +
                0.5f * step_alpha * static_cast<float>(interp_dt);
            const Eigen::Quaternionf q_i =
                integrateQuaternion(q0, interp_omega, interp_dt);
            const Eigen::Vector3f p_i =
                p + v * static_cast<float>(interp_dt) +
                0.5f * a0 * static_cast<float>(interp_dt * interp_dt) +
                (1.0f / 6.0f) * step_jerk *
                    static_cast<float>(interp_dt * interp_dt * interp_dt);
            const Eigen::Vector3f v_i =
                v + a0 * static_cast<float>(interp_dt) +
                0.5f * step_jerk * static_cast<float>(interp_dt * interp_dt);

            Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
            T.block<3, 3>(0, 0) = q_i.toRotationMatrix();
            T.block<3, 1>(0, 3) = p_i;
            IntegratedPose state;
            state.stamp = *stamp_it;
            state.T = T;
            state.velocity = v_i;
            states.push_back(state);
            ++stamp_it;
        }

        p += v * static_cast<float>(step_dt) +
             0.5f * a0 * static_cast<float>(step_dt * step_dt) +
             (1.0f / 6.0f) * jerk_dt * static_cast<float>(step_dt * step_dt);
        v += a0 * static_cast<float>(step_dt) +
             0.5f * jerk_dt * static_cast<float>(step_dt);
    }

    return states;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
integrateImu(const std::vector<ImuMeasurement>& imu_samples,
             const ImuIntegrationRequest& request) {
    const auto states = integrateImuStates(imu_samples, request);
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
        poses;
    poses.reserve(states.size());
    for (const auto& state : states) {
        poses.push_back(state.T);
    }
    return poses;
}

}  // namespace dlio
}  // namespace lio
}  // namespace n3mapping
