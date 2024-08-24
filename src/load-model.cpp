#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/crba.hpp" // 引入计算关节空间惯性矩阵的函数
#include "pinocchio/algorithm/rnea.hpp" // 引入计算关节科氏力矩阵的函数
#include <iomanip> // 引入用于格式化输出的库
 
#include <iostream>
 
// PINOCCHIO_MODEL_DIR is defined by the CMake but you can define your own directory here.
#ifndef PINOCCHIO_MODEL_DIR
  #define PINOCCHIO_MODEL_DIR "/home/ywy/pino_ws/src"
#endif
 
int main(int argc, char ** argv)
{
  using namespace pinocchio;
  
  // You should change here to set up your own URDF file or just pass it as an argument of this example.
  const std::string urdf_filename = (argc<=1) ? PINOCCHIO_MODEL_DIR + std::string("/ur5.urdf") : argv[1];
  
  // Load the urdf model
  Model model;
  pinocchio::urdf::buildModel(urdf_filename,model);
  std::cout << "model name: " << model.name << std::endl;
  
  // Create data required by the algorithms
  Data data(model);
  
  // Sample a random configuration
  Eigen::VectorXd q = randomConfiguration(model);
  std::cout << "q: " << q.transpose() << std::endl;

  //目前采用一个随机的速度配置
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv); // 随机生成关节速度
  std::cout << "v: " << v.transpose() << std::endl;

  // Perform the forward kinematics over the kinematic tree
  forwardKinematics(model,data,q);
 
  // Print out the placement of each joint of the kinematic tree
  for(JointIndex joint_id = 0; joint_id < (JointIndex)model.njoints; ++joint_id)
    std::cout << std::setw(24) << std::left
              << model.names[joint_id] << ": "
              << std::fixed << std::setprecision(2)
              << data.oMi[joint_id].translation().transpose()
              << std::endl;
  
  // Compute the joint space inertia matrix
  crba(model, data, q);
  std::cout << "Joint Space Inertia Matrix (JSIM):\n";
  std::cout << std::fixed << std::setprecision(6); // 设置浮点数精度为小数点后四位
  std::cout << data.M << std::endl; // 打印关节空间惯性矩阵

  // Compute the non-linear effects (Coriolis and centrifugal forces)
  Eigen::VectorXd nle = nonLinearEffects(model, data, q, v);
  std::cout << "Non-Linear Effects (Coriolis and Centrifugal Forces):\n" << nle << std::endl;

  // Compute the gravity term
  Eigen::VectorXd g = computeGeneralizedGravity(model, data, q);
  std::cout << "Generalized Gravity:\n" << g << std::endl;

}
