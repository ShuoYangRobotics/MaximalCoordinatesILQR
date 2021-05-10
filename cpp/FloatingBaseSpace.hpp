#pragma once
#include <Eigen/Dense>
#include <vector>
#include <assert.h>
using namespace Eigen;
class FloatingBaseSpace {
  public:
    FloatingBaseSpace();
    FloatingBaseSpace(int nb, bool orth=true);
    FloatingBaseSpace(int nb, std::vector<Eigen::Vector3d> _joint_directions);

  private:
    double body_mass;
    double body_size;
    double arm_mass;
    double arm_width;
    double arm_length;
    Matrix3d body_mass_mtx;
    Matrix3d arm_mass_mtx;
    Matrix3d body_inertias;
    Matrix3d arm_inertias;
    double g;
    
    int nb;
    int n;
    int np;
    int ne;

    std::vector<Eigen::Vector3d> joint_directions;
    std::vector<Eigen::Vector3d> joint_vertices;
    std::vector<Eigen::MatrixXd> joint_cmat;
};