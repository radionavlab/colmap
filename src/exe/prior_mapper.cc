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

#include "controllers/incremental_mapper.h"
#include "controllers/bundle_adjustment.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include <Eigen/Geometry>

using namespace colmap;

void ReadCameraMeasurements(const std::string& path,
                         std::vector<std::string>* image_names,
                         std::vector<Eigen::Vector3d>* measured_camera_positions,
                         std::vector<Eigen::Vector4d>* measured_camera_orientations) {
  std::vector<std::string> lines = ReadTextFileLines(path);
  for (const auto line : lines) {
    std::stringstream line_parser(line);
    std::string image_name = "";
    Eigen::Vector3d camera_position;
    Eigen::Vector4d camera_orientation;
    line_parser >> 
        image_name >> 
        camera_position(0) >> 
        camera_position(1) >> 
        camera_position(2) >> 
        camera_orientation(0) >>
        camera_orientation(1) >>
        camera_orientation(2) >>
        camera_orientation(3);
    image_names->push_back(image_name);
    measured_camera_positions->push_back(camera_position);
    measured_camera_orientations->push_back(camera_orientation);
  }
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string import_path;
  std::string export_path;
  std::string image_pose_path;

  OptionManager options;
  options.AddRequiredOption("import_path", &import_path);
  options.AddRequiredOption("export_path", &export_path);
  options.AddRequiredOption("image_pose_path", &image_pose_path);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(import_path);

  /* GAME PLAN
  * 1) Read in the camera measurements
  * 2) Determine the similarity transform
  * 3) Apply the similarity transform to the measurements
  * 4) Insert the measurements into Colmap Image object 
  * 5) Run global BA
  */

  /* 1) Read in the camera measurements */
  std::vector<std::string> image_names;
  std::vector<Eigen::Vector3d> TvecPriorsGlobal;
  std::vector<Eigen::Vector4d> QvecPriorsGlobal;
  ReadCameraMeasurements(image_pose_path, &image_names, &TvecPriorsGlobal, &QvecPriorsGlobal);

  /* 2) Determine the similarity transform between camera locations in visual
   * frame and camera locations in global frame */
  SimilarityTransform3 tform;
  bool alignment_success = reconstruction.AlignMeasurements(
          image_names,
          TvecPriorsGlobal,
          3,
          tform);

  if (alignment_success) {
    std::cout << " => Alignment succeeded" << std::endl;
  } else {
    std::cout << " => Alignment failed" << std::endl;
    return EXIT_FAILURE;
  }
 
  /* 3) Apply the similarity transform to the priors */
  std::vector<Eigen::Vector3d> TvecPriorsVisual = TvecPriorsGlobal;
  std::vector<Eigen::Vector4d> QvecPriorsVisual = QvecPriorsGlobal;

  std::vector<Eigen::Vector3d> TvecPriorsCamera = TvecPriorsGlobal;
  std::vector<Eigen::Vector4d> QvecPriorsCamera = QvecPriorsGlobal;

  std::unordered_map< std::string, std::pair<Eigen::Vector3d, Eigen::Vector4d> > image_poses;
  for(size_t i = 0; i < TvecPriorsVisual.size(); i++) {

    // Apply similarity transform
    // Transform camera position from ECEF frame to visual frame
    tform.TransformPoint(&TvecPriorsVisual[i]);

    // Transform camera orientation
    // Unit vectors of camera frame
    const Eigen::Vector3d cx = Eigen::Vector3d::UnitX();
    const Eigen::Vector3d cy = Eigen::Vector3d::UnitY();
    const Eigen::Vector3d cz = Eigen::Vector3d::UnitZ();

    // Camera unit vectors expressed in body frame 
    // Important note: Camera frame has z along focal axis, x to the right, and
    // y down like computer graphics
    const Eigen::Vector4d cq = RotationMatrixToQuaternion(EulerAnglesToRotationMatrix(-M_PI/2, 0, -M_PI/2));
    const Eigen::Vector3d bx = QuaternionRotatePoint(cq, cx);
    const Eigen::Vector3d by = QuaternionRotatePoint(cq, cy);
    const Eigen::Vector3d bz = QuaternionRotatePoint(cq, cz);

    // Camera vectors expressed in ECEF frame
    // No longer unit
    Eigen::Vector3d gx = QuaternionRotatePoint(QvecPriorsGlobal[i], bx) + TvecPriorsGlobal[i]; 
    Eigen::Vector3d gy = QuaternionRotatePoint(QvecPriorsGlobal[i], by) + TvecPriorsGlobal[i]; 
    Eigen::Vector3d gz = QuaternionRotatePoint(QvecPriorsGlobal[i], bz) + TvecPriorsGlobal[i]; 

    // Camera vectors expressed in visual frame with respect to visual origin
    tform.TransformPoint(&gx);
    tform.TransformPoint(&gy);
    tform.TransformPoint(&gz);

    // Camera vectors with respect to camera origin
    gx = gx - TvecPriorsVisual[i];
    gy = gy - TvecPriorsVisual[i];
    gz = gz - TvecPriorsVisual[i];

    // Normalize. Camera vectors have been scaled by similarity transform
    gx = gx.normalized();
    gy = gy.normalized();
    gz = gz.normalized();

    // Stack vectors in a rotation matrix and cast them as a quaternion 
    const Eigen::Quaterniond q((Eigen::Matrix3d() << gx, gy, gz).finished());
    QvecPriorsVisual[i] = Eigen::Vector4d(q.w(), q.x(), q.y(), q.z()); 

    // Tranform priors from visual to camera frame
    TvecPriorsCamera[i] = QuaternionRotatePoint(InvertQuaternion(QvecPriorsVisual[i]), -1*TvecPriorsVisual[i]); 
    QvecPriorsCamera[i] = InvertQuaternion(QvecPriorsVisual[i]);
     
    // Insert camera priors
    image_poses.insert({
                image_names[i], 
                std::make_pair(TvecPriorsCamera[i], QvecPriorsCamera[i])
    });
  }

  /* 4) Insert the measurements into reconstruction */
  reconstruction.AddPriors(image_poses);

  /* 5) Run global BA */
  options.bundle_adjustment->compute_covariance = true;
  options.bundle_adjustment->normalize = false;
  options.bundle_adjustment->solver_options.max_num_iterations = 1000;
  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();

  /* 6) Apply inverse similarity transform to the model */
  SimilarityTransform3 tformInverse = tform.Inverse();
  reconstruction.ReAlign(tformInverse);
 
  // reconstruction.Write(export_path);
  reconstruction.WriteText(export_path);
  std::cout << "Success!" << std::endl;

  return EXIT_SUCCESS;
}
