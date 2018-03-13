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
  std::string image_list_path;
  std::string image_pose_path;

  OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddRequiredOption("import_path", &import_path);
  options.AddRequiredOption("export_path", &export_path);
  options.AddRequiredOption("image_pose_path", &image_pose_path);
  options.AddMapperOptions();
  options.Parse(argc, argv);

  if (!ExistsDir(export_path)) {
    std::cerr << "ERROR: `export_path` is not a directory." << std::endl;
    return EXIT_FAILURE;
  }

  if (!image_list_path.empty()) {
    const auto image_names = ReadTextFileLines(image_list_path);
    options.mapper->image_names =
        std::set<std::string>(image_names.begin(), image_names.end());
  }

  /* GAME PLAN
  * 1) Read in the camera measurements
  * 2) Determine the similarity transform
  * 3) Apply the similarity transform to the measurements
  * 4) Insert the measurements into Colmap Image object 
  * 5) Re-run the mapper with the new cost function
  */

  Reconstruction reconstruction;
  reconstruction.Read(import_path);

  /* 1) Read in the camera measurements */
  std::vector<std::string> image_names;
  std::vector<Eigen::Vector3d> measured_camera_positions;
  std::vector<Eigen::Vector3d> measured_camera_orientations;
  ReadCameraMeasurements(image_pose_path, &image_names, &measured_camera_positions, &measured_camera_orientations);

  /* 2) Determine the similarity transform */
  SimilarityTransform3 tform;
  bool alignment_success = reconstruction.AlignMeasurements(
          image_names,
          measured_camera_positions,
          3,
          tform);

  if (alignment_success) {
    std::cout << " => Alignment succeeded" << std::endl;
  } else {
    std::cout << " => Alignment failed" << std::endl;
    return EXIT_FAILURE;
  }
 
  /* 3) Apply the similarity transform to the measurements */
  // Copy measurement vectors for transformation
  std::vector<Eigen::Vector3d> measurement_locations = measured_camera_positions;
  std::vector<Eigen::Vector4d> measurement_orientations;
  std::unordered_map< std::string, std::pair<Eigen::Vector3d, Eigen::Vector4d> > image_poses;
  for(size_t i = 0; i < measured_camera_positions.size(); i++) {
    // Convert Euler angle to quaternion
    // 3-2-1 euler angles
    // yaw about body z axis, pitch about body y axis, roll about body x axis
    const double roll  = measured_camera_orientations[i](0);
    const double pitch = measured_camera_orientations[i](1);
    const double yaw   = measured_camera_orientations[i](2);
    
    const Eigen::Quaterniond q = Eigen::Quaterniond(
            Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ()) * 
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * 
            Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX()));

    Eigen::Vector4d measurement_orientation(q.w(), q.x(), q.y(), q.z());    
    measurement_orientations.push_back(measurement_orientation);

    // Apply similarity 
    // The new measurements are in the visual frame.
    tform.TransformPoint(&measurement_locations[i]);
    tform.TransformQuaternion(&measurement_orientations[i]);

    // Rotate the measurements to the camera frame. This aligns the QvecPriors
    // with Qvec, and the TvecPriors with Tvec
    // -1*QuaternionRotatePoint(QvecPrior, TvecPrior);
     
    // Add transformed measurement to set
    image_poses.insert({
                image_names[i], 
                std::make_pair(measurement_locations[i], measurement_orientations[i])
    });
  }

  // 4) Insert the measurements into reconstruction
  // options.mapper->image_poses = image_poses;
  reconstruction.AddPriors(image_poses);

  // 5) Re-run the mapper with the new cost function
  BundleAdjustmentController ba_controller(options, &reconstruction);
  ba_controller.Start();
  ba_controller.Wait();

  // 6) Apply inverse similarity transform to the model
  SimilarityTransform3 tformInverse = tform.Inverse();
  reconstruction.ReAlign(tformInverse);

  // In case the reconstruction is continued from an existing reconstruction, do
  // not create sub-folders but directly write the results.
  // if(import_path != "" && reconstruction_manager.Size() > 0) {
  //   reconstruction_manager.Get(0).Write(export_path);
  // }
  
  // reconstruction.Write(export_path);

  return EXIT_SUCCESS;
}
