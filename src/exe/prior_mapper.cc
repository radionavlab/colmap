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
                         std::vector<Eigen::Vector3d>* TvecPriorsGlobal,
                         std::vector<Eigen::Vector3d>* TvecPriorsCamera,
                         std::vector<Eigen::Matrix3d>* RvecPriorsCamera,
                         std::vector< Eigen::Matrix<double, 6, 6> >* PriorsCovariance) {
  std::vector<std::string> lines = ReadTextFileLines(path);
  for (const auto line : lines) {
    std::stringstream line_parser(line);

    // Buffers to store data in
    std::string image_name = "";
    Eigen::Vector3d TvecPriorGlobal;
    Eigen::Vector3d TvecPriorCamera;
    Eigen::Matrix3d RvecPriorCamera;
    Eigen::Matrix<double, 6, 6> PriorCovariance;

    // Read data into buffers
    // Translation
    line_parser >> 
        image_name >> 
        TvecPriorGlobal(0) >> 
        TvecPriorGlobal(1) >> 
        TvecPriorGlobal(2) >> 
        TvecPriorCamera(0) >>
        TvecPriorCamera(1) >>
        TvecPriorCamera(2);

    // Rotation
    for(int i = 0; i < 3; i ++) {
        for(int j = 0; j < 3; j++) {
            line_parser >> RvecPriorCamera(j, i);
        }
    }

    // Covariance
    for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
            line_parser >> PriorCovariance(j, i);
        }
    }

    // Push data into vectors
    image_names->push_back(image_name);
    TvecPriorsGlobal->push_back(TvecPriorGlobal);
    TvecPriorsCamera->push_back(TvecPriorCamera);
    RvecPriorsCamera->push_back(RvecPriorCamera);
    PriorsCovariance->push_back(PriorCovariance);
  }
}

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string import_path;
  std::string export_path;
  std::string min_set_path;
  std::string max_set_path;
  std::string min_pose_path;
  std::string max_pose_path;

  OptionManager options;
  options.AddRequiredOption("import_path", &import_path);
  options.AddRequiredOption("export_path", &export_path);
  options.AddRequiredOption("min_set_path", &min_set_path);
  options.AddRequiredOption("max_set_path", &max_set_path);
  options.AddRequiredOption("min_pose_path", &min_pose_path);
  options.AddRequiredOption("max_pose_path", &max_pose_path);
  options.AddBundleAdjustmentOptions();
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(import_path);

  /* GAME PLAN
  * 1) Read in the measurements
  * 2) Apply similarity transform to data. Bring reconstruction into global frame.
  * 3) Insert the measurements into Colmap
  * 4) Run global BA
  */

  /* 1) Read in the camera measurements */
  std::vector<std::string> imageNamesMin;
  std::vector<Eigen::Vector3d> TvecPriorsGlobalMin;
  std::vector<Eigen::Vector3d> TvecPriorsCameraMin;
  std::vector<Eigen::Matrix3d> RvecPriorsCameraMin;
  std::vector< Eigen::Matrix<double, 6, 6> > PriorsCovarianceMin;
  ReadCameraMeasurements(
          min_pose_path, 
          &imageNamesMin, 
          &TvecPriorsGlobalMin, 
          &TvecPriorsCameraMin, 
          &RvecPriorsCameraMin, 
          &PriorsCovarianceMin);

  std::vector<std::string> imageNamesMax;
  std::vector<Eigen::Vector3d> TvecPriorsGlobalMax;
  std::vector<Eigen::Vector3d> TvecPriorsCameraMax;
  std::vector<Eigen::Matrix3d> RvecPriorsCameraMax;
  std::vector< Eigen::Matrix<double, 6, 6> > PriorsCovarianceMax;
  ReadCameraMeasurements(
          max_pose_path, 
          &imageNamesMax, 
          &TvecPriorsGlobalMax, 
          &TvecPriorsCameraMax, 
          &RvecPriorsCameraMax, 
          &PriorsCovarianceMax);

  /* 2) Align reconstruction with ECEF measurements to bring it into global frame. */
  bool alignmentSuccess = reconstruction.Align(
          imageNamesMin,
          TvecPriorsGlobalMin,
          3);

  if (alignmentSuccess) {
    std::cout << " => Alignment succeeded" << std::endl;
  } else {
    std::cout << " => Alignment failed" << std::endl;
    return EXIT_FAILURE;
  }
 

  /* 3) Insert all the measurements into reconstruction */
  std::unordered_map< std::string, std::tuple<Eigen::Vector3d, Eigen::Vector4d, Eigen::Matrix<double, 6, 6> > > imagePriors;
  for(size_t i = 0; i < TvecPriorsCameraMin.size(); i++) {
    imagePriors.insert({
                imageNamesMin[i], 
                std::make_tuple(TvecPriorsCameraMin[i], 
                                RotationMatrixToQuaternion(RvecPriorsCameraMin[i]), 
                                PriorsCovarianceMin[i])
    });
  }

  reconstruction.AddPriors(imagePriors);

  /* 4) Run global BA */
  options.bundle_adjustment->cov.compute = true;
  options.bundle_adjustment->cov.axle = Eigen::Vector3d(0,0,1);
  options.bundle_adjustment->cov.axle_threshhold = 0.5;
  options.bundle_adjustment->cov.alt_threshhold = 1.0;
  options.bundle_adjustment->normalize = false;
  options.bundle_adjustment->solver_options.max_num_iterations = 1000;
  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();
 
  // reconstruction.Write(export_path);
  reconstruction.WriteText(export_path);

  /* GAME PLAN 
   * 1) Min/Max image directory
   * 2) Read in all images. Extract/match all features.
   * 3) Reconstruct only minimal scene
   * 4) REPEAT
   * 4.1) Evaluate covariance and accuracy metric
   * 4.2) Apply image selection criteria to all unchosen images
   * 4.3) Select best image, add to reconstruction, reconstruct.
   * 4.4) END REPEAT
   */
  
  
  std::cout << "Success!" << std::endl;

  return EXIT_SUCCESS;
}
