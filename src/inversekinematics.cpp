#include "pinocchio/parsers/sample-models.hpp" // 引入 Pinocchio 库中的示例模型解析器
#include "pinocchio/spatial/explog.hpp" // 引入 Pinocchio 的指数映射和对数映射算法
#include "pinocchio/algorithm/kinematics.hpp" // 引入用于计算运动学的 Pinocchio 算法
#include "pinocchio/algorithm/jacobian.hpp" // 引入用于计算雅可比矩阵的 Pinocchio 算法
#include "pinocchio/algorithm/joint-configuration.hpp" // 引入用于关节配置操作的 Pinocchio 算法

int main(int /* argc */, char ** /* argv */) // 主函数
{
  pinocchio::Model model; // 创建一个 Pinocchio 模型对象
  pinocchio::buildModels::manipulator(model); // 构建一个示例操作臂模型
  pinocchio::Data data(model); // 根据模型创建一个数据对象，用于存储运算过程中的中间数据

  const int JOINT_ID = 6; // 设置要操作的关节ID
  const pinocchio::SE3 oMdes(Eigen::Matrix3d::Identity(), Eigen::Vector3d(1., 0., 1.)); // 设置目标位置和方向，没有旋转变化，但有特定的平移变换

  Eigen::VectorXd q = pinocchio::neutral(model); // 获取模型的中性（默认）关节位置
  const double eps  = 1e-4; // 设置迭代的精度
  const int IT_MAX  = 1000; // 设置最大迭代次数
  const double DT   = 1e-1; // 设置时间步长
  const double damp = 1e-6; // 设置阻尼系数

  pinocchio::Data::Matrix6x J(6,model.nv); // 创建一个雅可比矩阵
  J.setZero(); // 初始化雅可比矩阵为零

  bool success = false; // 设置一个标志，用于表示算法是否成功
  typedef Eigen::Matrix<double, 6, 1> Vector6d; // 定义一个6维向量类型
  Vector6d err; // 创建一个6维误差向量
  Eigen::VectorXd v(model.nv); // 创建一个速度向量
  for (int i=0;;i++) // 迭代循环
  {
    pinocchio::forwardKinematics(model,data,q); // 执行前向运动学计算
    const pinocchio::SE3 iMd = data.oMi[JOINT_ID].actInv(oMdes); // 计算当前关节到目标位置的变换
    err = pinocchio::log6(iMd).toVector();  // 计算误差向量
    if(err.norm() < eps) // 如果误差小于阈值，则成功
    {
      success = true;
      break;
    }
    if (i >= IT_MAX) // 如果迭代次数超过最大值，失败
    {
      success = false;
      break;
    }
    pinocchio::computeJointJacobian(model,data,q,JOINT_ID,J);  // 计算雅可比矩阵
    pinocchio::Data::Matrix6 Jlog; // 创建一个6x6矩阵
    pinocchio::Jlog6(iMd.inverse(), Jlog); // 计算对数映射的雅可比
    J = -Jlog * J; // 更新雅可比矩阵
    pinocchio::Data::Matrix6 JJt; // 创建一个6x6矩阵
    JJt.noalias() = J * J.transpose(); // 计算雅可比矩阵的乘积
    JJt.diagonal().array() += damp; // 添加阻尼，为了增加数值稳定性，特别是当 J * J.transpose() 接近奇异时
    v.noalias() = - J.transpose() * JJt.ldlt().solve(err); // 计算速度
    q = pinocchio::integrate(model,q,v*DT); // 更新关节位置
    if(!(i%10))
      std::cout << i << ": error = " << err.transpose() << std::endl; // 每10次迭代打印一次错误信息
  }

  if(success) 
  {
    std::cout << "Convergence achieved!" << std::endl; // 如果成功，打印成功信息
  }
  else 
  {
    std::cout << "\nWarning: the iterative algorithm has not reached convergence to the desired precision" << std::endl; // 如果失败，打印失败信息
  }

  std::cout << "\nresult: " << q.transpose() << std::endl; // 打印最终的关节位置
  std::cout << "\nfinal error: " << err.transpose() << std::endl; // 打印最终的误差
}