// Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#ifndef COLMAP_SRC_BASE_COST_FUNCTIONS_H_
#define COLMAP_SRC_BASE_COST_FUNCTIONS_H_

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "pose.h"

namespace colmap {

// Cost function to estimate camera pose given a measurement of its pose in the
// ENU frame
class CameraPositionENUCostFunction {
 public:
  CameraPositionENUCostFunction(const Eigen::Vector3d& ricI,
                                const Eigen::Matrix3d& cov)
      : ricI_(ricI),
        cov_(cov)
    {}

  static ceres::CostFunction* Create(const Eigen::Vector3d& ricI,
                                     const Eigen::Matrix3d& cov) {
    return (new ceres::AutoDiffCostFunction<
            CameraPositionENUCostFunction, 3, 3, 4>(
        new CameraPositionENUCostFunction(ricI, cov)));
  }

  template <typename T>
  bool operator()(const T* rciC, 
                  const T* qic,
                  T* residuals) const {

    typedef Eigen::Matrix<T, 3, 1> tvec_t;
    typedef Eigen::Matrix<T, 3, 3> cov_t;

    // Measurements
    const tvec_t ricI_meas = ricI_.cast<T>();
    // const Eigen::Matrix3d cov = Eigen::Vector3d(0.0025, 0.0025, 0.0025).asDiagonal();

    // Square root of information matrix
    const Eigen::LLT<Eigen::Matrix<double, 3, 3> > chol(cov_);
    const Eigen::Matrix<double, 3, 3> lower = chol.matrixL();
    const cov_t sqrt_info = lower.inverse().cast<T>();

    // Estimates
    T ricI[3];
    T R_IFC[4]; 
    T D_I2C[4]; 

    std::memcpy(R_IFC, qic, 4*sizeof(T));
    std::memcpy(D_I2C, R_IFC, 4*sizeof(T));

    const T D_C2I[4] = { T(-1)*D_I2C[0], D_I2C[1], D_I2C[2], D_I2C[3] };
    const T ricC[3] = { T(-1)*rciC[0], T(-1)*rciC[1], T(-1)*rciC[2] };

    ceres::UnitQuaternionRotatePoint(D_C2I, ricC, ricI);

    tvec_t ricI_est(ricI[0], ricI[1], ricI[2]);

    // tvec residual
    const tvec_t ricI_res = ricI_est - ricI_meas;

    // Scale by square root info
    tvec_t res = sqrt_info * ricI_res; 

    // Output
    residuals[0] = res(0);
    residuals[1] = res(1);
    residuals[2] = res(2);

    return true;
  }

 private:
  const Eigen::Vector3d ricI_;
  const Eigen::Matrix3d cov_;
};

// Cost function to estimate camera pose given a measurement of its position
class CameraPositionCostFunction {
 public:
  CameraPositionCostFunction(const Eigen::Vector3d& tvec,
                             const Eigen::Vector4d& qvec,
                             const Eigen::Matrix<double, 3, 3>& cov)
      : t_(tvec),
        q_(qvec),
        cov_(cov) 
    {}

  static ceres::CostFunction* Create(const Eigen::Vector3d& tvec,
                                     const Eigen::Vector4d& qvec,
                                     const Eigen::Matrix<double, 3, 3>& cov) {
    return (new ceres::AutoDiffCostFunction<
            CameraPositionCostFunction, 3, 3>(
        new CameraPositionCostFunction(tvec, qvec, cov)));
  }

  template <typename T>
  bool operator()(const T* const tvec, 
                  T* residuals) const {

    typedef Eigen::Matrix<T, 3, 1> tvec_t;
    typedef Eigen::Matrix<T, 3, 3> cov_t;

    // Measurements
    const tvec_t tvec_meas = t_.cast<T>();
    // Eigen::Matrix3d R = QuaternionToRotationMatrix(q_);
    // Eigen::Matrix3d cov = R.transpose() * Eigen::Vector3d(0.0004, 0.0004, 0.0004).asDiagonal() * R;

    // Square root of information matrix
    const Eigen::LLT<Eigen::Matrix<double, 3, 3> > chol(cov_);
    const Eigen::Matrix<double, 3, 3> lower = chol.matrixL();
    const cov_t sqrt_info = lower.inverse().cast<T>();

    // Estimates
    tvec_t tvec_est(tvec[0], tvec[1], tvec[2]);

    // tvec residual
    const tvec_t tvec_res = tvec_est - tvec_meas;

    // Scale by square root info
    tvec_t res = sqrt_info * tvec_res; 

    // Output
    residuals[0] = res(0);
    residuals[1] = res(1);
    residuals[2] = res(2);

    return true;
  }

 private:
  const Eigen::Vector3d t_;
  const Eigen::Vector4d q_;
  const Eigen::Matrix<double, 3, 3> cov_;
};

// Cost function to estimate camera pose given a measurement of its pose
class CameraPoseCostFunction {
 public:
  CameraPoseCostFunction(const Eigen::Vector4d& qvec,
                         const Eigen::Vector3d& tvec,
                         const Eigen::Matrix<double, 6, 6>& cov)
      : q_(qvec),
        t_(tvec),
        cov_(cov) 
    {}

  static ceres::CostFunction* Create(const Eigen::Vector4d& qvec,
                                     const Eigen::Vector3d& tvec,
                                     const Eigen::Matrix<double, 6, 6>& cov) {
    return (new ceres::AutoDiffCostFunction<
            CameraPoseCostFunction, 6, 4, 3>(
        new CameraPoseCostFunction(qvec, tvec, cov)));
  }

  template <typename T>
  bool operator()(const T* const qvec, const T* const tvec, 
                  T* residuals) const {

    typedef Eigen::Matrix<T, 4, 1> QVEC;
    typedef Eigen::Matrix<T, 3, 1> TVEC;
    typedef Eigen::Matrix<T, 6, 6> COV;

    // Measurements
    const QVEC qvec_meas = q_.cast<T>();
    const TVEC tvec_meas = t_.cast<T>();

    // Square root of information matrix
    const Eigen::LLT<Eigen::Matrix<double, 6, 6> > chol(cov_);
    const Eigen::Matrix<double, 6, 6> lower = chol.matrixL();
    const COV sqrt_info = lower.inverse().cast<T>();

    // Estimates
    const QVEC qvec_est(qvec[0], qvec[1], qvec[2], qvec[3]);
    const TVEC tvec_est(tvec[0], tvec[1], tvec[2]);

    // Conjugate/invert estimated quaternion
    // Conjugate is inverse when quaternion is unit
    const QVEC qvec_est_inv(qvec[0], -qvec[1], -qvec[2], -qvec[3]); 
    const QVEC qvec_meas_inv(qvec_meas(0), T(-1)*qvec_meas(1), T(-1)*qvec_meas(2), T(-1)*qvec_meas(3));

    // Calculate quaternion error
    T dq[4];
    ceres::QuaternionProduct(qvec_est.data(), qvec_meas_inv.data(), dq);

    // Normalize quaternion error
    const T norm = sqrt(dq[0]*dq[0] + dq[1]*dq[1] + dq[2]*dq[2] + dq[3]*dq[3]);
    dq[0] /= norm;
    dq[1] /= norm;
    dq[2] /= norm;
    dq[3] /= norm;

    // Convert quaternion error to euler angle error
    // http://www.sedris.org/wg8home/Documents/WG80485.pdf
    // Page 39
    T re[3] = {
        atan2((dq[0]*dq[1] + dq[2]*dq[3]), T(0.5) - (dq[1]*dq[1] + dq[2]*dq[2])),
        asin(T(2)*(dq[0]*dq[2] - dq[3]*dq[1])),
        atan2((dq[0]*dq[3] + dq[1]*dq[2]), T(0.5) - (dq[2]*dq[2] + dq[3]*dq[3]))
    };

    // tvec residual
    const TVEC rt = tvec_est - tvec_meas;

    // Combined residuals
    const Eigen::Matrix<T, 6, 1> r = (Eigen::Matrix<T, 6, 1>() << rt(0), rt(1), rt(2), re[0], re[1], re[2]).finished();

    // Scale by square root info
    const Eigen::Matrix<T, 6, 1> rr = sqrt_info * r; 

    // Output
    residuals[0] = rr(0);
    residuals[1] = rr(1);
    residuals[2] = rr(2);
    residuals[3] = rr(3);
    residuals[4] = rr(4);
    residuals[5] = rr(5);

    return true;
  }

 private:
  const Eigen::Vector4d q_;
  const Eigen::Vector3d t_;
  const Eigen::Matrix<double, 6, 6> cov_;
};

// Standard bundle adjustment cost function for variable
// camera pose and calibration and point parameters.
template <typename CameraModel>
class BundleAdjustmentCostFunction {
 public:
  explicit BundleAdjustmentCostFunction(const Eigen::Vector2d& point2D)
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

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
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);

    // Covariance of pixels. Assumed to be independent. Divide by standard deviation.
    const T sig = T(5.0);
    residuals[0] = residuals[0] / sig;
    residuals[1] = residuals[1] / sig;

    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
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
        observed_x_(point2D(0)),
        observed_y_(point2D(1)) {}

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
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);

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
  const double observed_x_;
  const double observed_y_;
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
      : observed_x_(point2D(0)), observed_y_(point2D(1)) {}

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
    residuals[0] -= T(observed_x_);
    residuals[1] -= T(observed_y_);

    return true;
  }

 private:
  const double observed_x_;
  const double observed_y_;
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
      : x1_(x1(0)), y1_(x1(1)), x2_(x2(0)), y2_(x2(1))
    {}

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
