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
                         std::vector<Eigen::Vector3d>* measured_camera_orientations) {
  std::vector<std::string> lines = ReadTextFileLines(path);
  for (const auto line : lines) {
    std::stringstream line_parser(line);
    std::string image_name = "";
    Eigen::Vector3d camera_position;
    Eigen::Vector3d camera_orientation;
    line_parser >> 
        image_name >> 
        camera_position(0) >> 
        camera_position(1) >> 
        camera_position(2) >> 
        camera_orientation(0) >>
        camera_orientation(1) >>
        camera_orientation(2);
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
  std::vector<Eigen::Vector3d> EvecPriorsGlobal;
  ReadCameraMeasurements(image_pose_path, &image_names, &TvecPriorsGlobal, &EvecPriorsGlobal);

  /* 1.1) Convert 3-2-1 Euler angle rotation of camera away from ENU to quaternion */
  std::vector<Eigen::Vector4d> QvecPriorsGlobal;
  std::for_each(
    EvecPriorsGlobal.begin(), 
    EvecPriorsGlobal.end(),
    [&](const Eigen::Vector3d e){ 
      QvecPriorsGlobal.push_back(
        RotationMatrixToQuaternion(
            EulerAnglesToRotationMatrix(e(0), e(1), e(2))
        )
      );
    }
  );


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
  std::unordered_map< std::string, std::pair<Eigen::Vector3d, Eigen::Vector4d> > image_poses;
  for(size_t i = 0; i < TvecPriorsVisual.size(); i++) {

    // Apply similarity 
    tform.TransformPoint(&TvecPriorsVisual[i]);
    tform.TransformQuaternion(&QvecPriorsVisual[i]);
     
    // Add transformed measurement to set
    image_poses.insert({
                image_names[i], 
                std::make_pair(TvecPriorsVisual[i], QvecPriorsVisual[i])
    });
  }

  /* 4) Insert the measurements into reconstruction */
  reconstruction.AddPriors(image_poses);

  /* 5) Run global BA */
  options.bundle_adjustment->compute_covariance = false;
  options.bundle_adjustment->normalize = false;
  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();

  /* 6) Apply inverse similarity transform to the model */
  SimilarityTransform3 tformInverse = tform.Inverse();
  reconstruction.ReAlign(tformInverse);
 
  // reconstruction.Write(export_path);
  // reconstruction.WriteText(export_path);
  std::cout << "Success!" << std::endl;

  return EXIT_SUCCESS;
}
