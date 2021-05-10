#include "FloatingBaseSpace.hpp"

// default constructor makes four link orth robot
FloatingBaseSpace::FloatingBaseSpace() {
  int nb_ = 4; 
  FloatingBaseSpace(nb_, true);
}

FloatingBaseSpace::FloatingBaseSpace(int nb, bool orth) {
  assert(nb > 0);
  if (orth == true) {
    joint_
  } else {
      
  }
}

FloatingBaseSpace::FloatingBaseSpace(int nb, std::vector<Eigen::Vector3d> _joint_directions) {

}