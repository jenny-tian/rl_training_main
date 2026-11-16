// Offline probe of the actual deployment runner; no robot, sockets or physics.
#include "lite3_test_policy_runner_onnx.h"
#include "basic_function.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    Lite3TestPolicyRunnerONNX runner("offline_entry_audit");
    RobotBasicState state;
    state.base_rpy = Vec3f::Zero();
    state.base_rot_mat = Mat3f::Identity();
    state.projected_gravity = Vec3f(0, 0, -1);
    state.base_quat = Vec4f(1, 0, 0, 0);
    state.base_omega = Vec3f::Zero();
    state.base_acc = Vec3f::Zero();
    state.cmd_vel_normlized = Vec3f(.5f, .125f, -.625f);
    state.joint_pos = Vec3f(0, -.65f, 1.3f).replicate(4, 1);
    state.joint_vel = VecXf::Zero(12);
    state.joint_tau = VecXf::Zero(12);
    std::ofstream out(argv[1]);
    out << std::setprecision(10);
    runner.OnEnter();
    std::cout << "LAST_ACTION_SIZE_AT_FIRST_ENTRY=" << runner.AuditLastAction().size() << std::endl;
    for (int i = 0; i < 6; ++i) {
        if (i == 4) runner.OnEnter();
        if (i == 5) state.joint_vel.setConstant(150.f);
        RobotAction action = runner.GetRobotAction(state);
        out << i;
        for (float v : runner.AuditObservation()) out << ',' << v;
        for (float v : runner.AuditLastAction()) out << ',' << v;
        for (float v : action.goal_joint_pos) out << ',' << v;
        out << '\n';
    }
    return 0;
}
