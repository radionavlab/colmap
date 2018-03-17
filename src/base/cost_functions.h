// COLMAP - Structure-from-Motion and Multi-View Stereo.
// Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef COLMAP_SRC_BASE_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace colmap {

// Cost function to estimate camera pose given a measurement of its pose
class CameraPoseCostFunction {
 public:
  CameraPoseCostFunction(const Eigen::Vector4d& qvec,
                         const Eigen::Vector3d& tvec)
      : qw_(qvec(0)),
        qx_(qvec(1)),
        qy_(qvec(2)),
        qz_(qvec(3)),
        tx_(tvec(0)),
        ty_(tvec(1)),
        tz_(tvec(2)) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec) {
    return (new ceres::AutoDiffCostFunction<
            CameraPoseCostFunction, 6, 4, 3>(
        new CameraPoseCostFunction(qvec, tvec)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec, 
                  T* residuals) const {

    // Measurements and estimates are all in camera frame
    const T tvec_meas[3] = {T(tx_), T(ty_), T(tz_)};
    const T qvec_meas[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
    
    residuals[0] = T(0);
    residuals[1] = T(0);
    residuals[2] = T(0);
    
    residuals[3] = tvec[0] - tvec_meas[0];
    residuals[4] = tvec[1] - tvec_meas[1];
    residuals[5] = tvec[2] - tvec_meas[2];

    return true;
  }

 private:
  const double qw_;
  const double qx_;
  const double qy_;
  const double qz_;
  const double tx_;
  const double ty_;
  const double tz_;
};

// // Cost function to estimate camera pose given a measurement of its pose
// class CameraPoseCostFunction {
//  public:
//   CameraPoseCostFunction(const Eigen::Vector4d& qvec,
//                          const Eigen::Vector3d& tvec)
//       : qw_(qvec(0)),
//         qx_(qvec(1)),
//         qy_(qvec(2)),
//         qz_(qvec(3)),
//         tx_(tvec(0)),
//         ty_(tvec(1)),
//         tz_(tvec(2)) {}
// 
//   static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
//                                      const Eigen::Vector3d& tvec) {
//     return (new ceres::AutoDiffCostFunction<
//             CameraPoseCostFunction, 6, 4, 3>(
//         new CameraPoseCostFunction(qvec, tvec)));
//   }
// 
//   template <typename T>
//   bool operator()(const T* const qvec, const T* const tvec, 
//                   T* residuals) const {
//     
//     // Measurements describe the translation of the camera away from the visual
//     // origin as expressed in the visual frame followed by a rotation away from
//     // the visual frame.
//     const T tvec_meas_visual[3] = {T(tx_), T(ty_), T(tz_)};
//     const T qvec_meas_visual[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};
// 
//     // Measurements are in visual frame
//     // Data are in camera frame
//     // Rotate the data to the visual frame
//     // Multiply the negative distance by the inverse quaternion
//     T tvec_visual[3];
//     const T tvec_camera_neg[3] = {-tvec[0], -tvec[1], -tvec[2]};
//     const T qvec_camera_inv[4] = {qvec[0], -qvec[1], -qvec[2], -qvec[3]};
//     ceres::QuaternionRotatePoint(qvec_camera_inv, tvec_camera_neg, tvec_visual);
// 
//     // Conjugate measurement quaternion to calculate error
//     const T qvec_meas_visual_conj[4] = {
//          qvec_meas_visual[0], 
//         -qvec_meas_visual[1], 
//         -qvec_meas_visual[2], 
//         -qvec_meas_visual[3]
//     };
// 
//     // Rotation of camera away from visual frame is the inverse or rotation of
//     // visual frame away from camera 
//     const T qvec_visual[4] = {
//          qvec[0], 
//         -qvec[1], 
//         -qvec[2], 
//         -qvec[3]
//     };
// 
//     // Calculate quaternion error
//     T dq[4];
//     ceres::QuaternionProduct(qvec_visual, qvec_meas_visual_conj, dq);
// 
//     // Normalize quaternion error
//     const T norm = sqrt(dq[0]*dq[0] + dq[1]*dq[1] + dq[2]*dq[2] + dq[3]*dq[3]);
//     dq[0] /= norm;
//     dq[1] /= norm;
//     dq[2] /= norm;
//     dq[3] /= norm;
// 
// 
//     // Convert quaternion error to axis-angle representation
//     ceres::QuaternionToAngleAxis(dq, residuals); 
// 
//     residuals[0] = T(0);
//     residuals[1] = T(0);
//     residuals[2] = T(0);
//     
//     residuals[3] = tvec_visual[0] - tvec_meas_visual[0];
//     residuals[4] = tvec_visual[1] - tvec_meas_visual[1];
//     residuals[5] = tvec_visual[2] - tvec_meas_visual[2];
// 
//     return true;
//   }
// 
//  private:
//   const double qw_;
//   const double qx_;
//   const double qy_;
//   const double qz_;
//   const double tx_;
//   const double ty_;
//   const double tz_;
// };

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentCostFunction {
 public:
  explicit BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : x_(point2D(0)), y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 3,
            CameraModel::kNumParams>(
        new BundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += tvec[0];
    projection[1] += tvec[1];
    projection[2] += tvec[2];

    // Project to image plane.
    projection[0] /= projection[2];
    projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(x_);
    residuals[1] -= T(y_);

    return true;
  }

 private:
  const double x_;
  const double y_;
};

// Bundle adjustment cost function for variable
// camera calibration and point parameters, and fixed camera pose.
template <typename CameraModel>
class BundleAdjustmentConstantPoseCostFunction {
 public:
  BundleAdjustmentConstantPoseCostFunction(const Eigen::Vector4d& qvec,
                                           const Eigen::Vector3d& tvec,
                                           const Eigen::Vector2d& point2D)
      : qw_(qvec(0)),
        qx_(qvec(1)),
        qy_(qvec(2)),
        qz_(qvec(3)),
        tx_(tvec(0)),
        ty_(tvec(1)),
        tz_(tvec(2)),
        x_(point2D(0)),
        y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            BundleAdjustmentConstantPoseCostFunction<CameraModel>, 2, 3,
            CameraModel::kNumParams>(
        new BundleAdjustmentConstantPoseCostFunction(qvec, tvec, point2D)));
  }

  template <typename T>
  bool operator()(const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    const T qvec[4] = {T(qw_), T(qx_), T(qy_), T(qz_)};

    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += T(tx_);
    projection[1] += T(ty_);
    projection[2] += T(tz_);

    // Project to image plane.
    projection[0] /= projection[2];
    projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(x_);
    residuals[1] -= T(y_);

    return true;
  }

 private:
  double qw_;
  double qx_;
  double qy_;
  double qz_;
  double tx_;
  double ty_;
  double tz_;
  double x_;
  double y_;
};

// Rig bundle adjustment cost function for variable camera pose and calibration
// and point parameters. Different from the standard bundle adjustment function,
// this cost function is suitable for camera rigs with consistent relative poses
// of the cameras within the rig. The cost function first projects points into
// the local system of the camera rig and then into the local system of the
// camera within the rig.
template <typename CameraModel>
class RigBundleAdjustmentCostFunction {
 public:
  explicit RigBundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : x_(point2D(0)), y_(point2D(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& point2D) {
    return (new ceres::AutoDiffCostFunction<
            RigBundleAdjustmentCostFunction<CameraModel>, 2, 4, 3, 4, 3, 3,
            CameraModel::kNumParams>(
        new RigBundleAdjustmentCostFunction(point2D)));
  }

  template <typename T>
  bool operator()(const T* const rig_qvec, const T* const rig_tvec,
                  const T* const rel_qvec, const T* const rel_tvec,
                  const T* const point3D, const T* const camera_params,
                  T* residuals) const {
    // Concatenate rotations.
    T qvec[4];
    ceres::QuaternionProduct(rel_qvec, rig_qvec, qvec);

    // Concatenate translations.
    T tvec[3];
    ceres::UnitQuaternionRotatePoint(rel_qvec, rig_tvec, tvec);
    tvec[0] += rel_tvec[0];
    tvec[1] += rel_tvec[1];
    tvec[2] += rel_tvec[2];

    // Rotate and translate.
    T projection[3];
    ceres::UnitQuaternionRotatePoint(qvec, point3D, projection);
    projection[0] += tvec[0];
    projection[1] += tvec[1];
    projection[2] += tvec[2];

    // Project to image plane.
    projection[0] /= projection[2];
    projection[1] /= projection[2];

    // Distort and transform to pixel space.
    CameraModel::WorldToImage(camera_params, projection[0], projection[1],
                              &residuals[0], &residuals[1]);

    // Re-projection error.
    residuals[0] -= T(x_);
    residuals[1] -= T(y_);

    return true;
  }

 private:
  const double x_;
  const double y_;
};

// Cost function for refining two-view geometry based on the Sampson-Error.
//
// First pose is assumed to be located at the origin with 0 rotation. Second
// pose is assumed to be on the unit sphere around the first pose, i.e. the
// pose of the second camera is parameterized by a 3D rotation and a
// 3D translation with unit norm. `tvec` is therefore over-parameterized as is
// and should be down-projected using `HomogeneousVectorParameterization`.
class RelativePoseCostFunction {
 public:
  RelativePoseCostFunction(const Eigen::Vector2d& x1, const Eigen::Vector2d& x2)
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1)) {}

  static ceres::CostFunction* Create(const Eigen::Vector2d& x1,
                                     const Eigen::Vector2d& x2) {
    return (new ceres::AutoDiffCostFunction<RelativePoseCostFunction, 1, 4, 3>(
        new RelativePoseCostFunction(x1, x2)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec,
                  T* residuals) const {
    Eigen::Matrix<T, 3, 3, Eigen::RowMajor> R;
    ceres::QuaternionToRotation(qvec, R.data());

    // Matrix representation of the cross product t x R.
    Eigen::Matrix<T, 3, 3> t_x;
    t_x << T(0), -tvec[2], tvec[1], tvec[2], T(0), -tvec[0], -tvec[1], tvec[0],
        T(0);

    // Essential matrix.
    const Eigen::Matrix<T, 3, 3> E = t_x * R;

    // Homogeneous image coordinates.
    const Eigen::Matrix<T, 3, 1> x1_h(T(x1_), T(y1_), T(1));
    const Eigen::Matrix<T, 3, 1> x2_h(T(x2_), T(y2_), T(1));

    // Squared sampson error.
    const Eigen::Matrix<T, 3, 1> Ex1 = E * x1_h;
    const Eigen::Matrix<T, 3, 1> Etx2 = E.transpose() * x2_h;
    const T x2tEx1 = x2_h.transpose() * Ex1;
    residuals[0] = x2tEx1 * x2tEx1 /
                   (Ex1(0) * Ex1(0) + Ex1(1) * Ex1(1) + Etx2(0) * Etx2(0) +
                    Etx2(1) * Etx2(1));

    return true;
  }

 private:
  const double x1_;
  const double y1_;
  const double x2_;
  const double y2_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_COST_FUNCTIONS_H_
