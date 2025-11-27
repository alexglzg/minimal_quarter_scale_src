#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_util/ellipsoid_decomp.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

class GridDecomp {
private:
  ros::NodeHandle nh_;
  ros::Subscriber grid_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher poly_pub_;
  ros::Publisher es_pub_;
  
  double forward_distance_;
  double backward_distance_;
  double bbox_x_;
  double bbox_y_;
  int occupancy_threshold_;
  
  bool odom_received_;
  bool obs_cached_;
  
  Vec2f start_point_;
  Vec2f end_point_;
  vec_Vec2f obs2d_;           // Local obstacles for current iteration
  vec_Vec2f all_obs_;         // All obstacles (cached once)
  std::string frame_id_;

  nav_msgs::Odometry::ConstPtr latest_odom_;

public:
  GridDecomp() : nh_("~"), odom_received_(false), obs_cached_(false) {
    // Get parameters
    nh_.param("forward_distance", forward_distance_, 0.5);
    nh_.param("backward_distance", backward_distance_, 0.5);
    nh_.param("bbox_x", bbox_x_, 5.0);
    nh_.param("bbox_y", bbox_y_, 5.0);
    nh_.param("occupancy_threshold", occupancy_threshold_, 50);
    
    ROS_INFO("Forward distance: %.2f m", forward_distance_);
    ROS_INFO("Backward distance: %.2f m", backward_distance_);
    ROS_INFO("Bounding box: [%.2f, %.2f]", bbox_x_, bbox_y_);
    ROS_INFO("Occupancy threshold: %d", occupancy_threshold_);
    
    // Publishers
    poly_pub_ = nh_.advertise<decomp_ros_msgs::PolyhedronArray>("/polyhedron_array", 1, true);
    es_pub_ = nh_.advertise<decomp_ros_msgs::EllipsoidArray>("/ellipsoid_array", 1, true);
    
    // Subscribers
    grid_sub_ = nh_.subscribe("/map", 1, &GridDecomp::gridCallback, this);
    odom_sub_ = nh_.subscribe("/odometry/filtered", 1, &GridDecomp::odomCallback, this);
  }

  void gridCallback(const nav_msgs::OccupancyGrid::ConstPtr& grid_msg) {
    frame_id_ = grid_msg->header.frame_id;
    
    ROS_INFO("Received occupancy grid: %dx%d, resolution: %.3f m", 
             grid_msg->info.width, grid_msg->info.height, grid_msg->info.resolution);
    
    // Convert entire grid to obstacles ONCE
    convertGridToObstacles(grid_msg);
    obs_cached_ = true;
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& odom_msg) {
    latest_odom_ = odom_msg;
    odom_received_ = true;
  }
  
  void run() {
    if (!obs_cached_) {
      ROS_WARN_THROTTLE(2.0, "Waiting for occupancy grid...");
      return;
    }
    
    if (!odom_received_) {
      ROS_WARN_THROTTLE(2.0, "Waiting for odometry...");
      return;
    }

    // Update start/end points from latest odometry
    updatePointsFromOdom();
    
    // Filter cached obstacles to local region
    filterLocalObstacles();
    
    // Process decomposition
    processDecomposition();
  }
  
  void updatePointsFromOdom() {
    double x = latest_odom_->pose.pose.position.x;
    double y = latest_odom_->pose.pose.position.y;
    
    tf2::Quaternion q(
      latest_odom_->pose.pose.orientation.x,
      latest_odom_->pose.pose.orientation.y,
      latest_odom_->pose.pose.orientation.z,
      latest_odom_->pose.pose.orientation.w
    );
    
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    
    start_point_(0) = x - backward_distance_ * cos(yaw);
    start_point_(1) = y - backward_distance_ * sin(yaw);
    
    end_point_(0) = x + forward_distance_ * cos(yaw);
    end_point_(1) = y + forward_distance_ * sin(yaw);
  }

void convertGridToObstacles(const nav_msgs::OccupancyGrid::ConstPtr& grid_msg) {
  all_obs_.clear();
  
  const auto& info = grid_msg->info;
  
  // Extract rotation from map origin
  tf2::Quaternion q_map(
    info.origin.orientation.x,
    info.origin.orientation.y,
    info.origin.orientation.z,
    info.origin.orientation.w
  );
  double roll, pitch, yaw;
  tf2::Matrix3x3(q_map).getRPY(roll, pitch, yaw);
  
  double cos_yaw = cos(yaw);
  double sin_yaw = sin(yaw);
  
  int downsample_factor = 3;  // Skip every other cell (0.03m -> 0.09m effective)
  
  ROS_INFO("Map rotation: %.2f rad (%.1f deg)", yaw, yaw * 180.0 / M_PI);
  ROS_INFO("Converting grid to obstacles with downsample factor %d...", downsample_factor);
  ros::Time start = ros::Time::now();
  
  for (unsigned int i = 0; i < info.height; i += downsample_factor) {
    for (unsigned int j = 0; j < info.width; j += downsample_factor) {
      int idx = i * info.width + j;
      int8_t value = grid_msg->data[idx];
      
      if (value >= occupancy_threshold_) {
        // Grid coordinates
        double x_grid = (j + 0.5) * info.resolution;
        double y_grid = (i + 0.5) * info.resolution;
        
        // Rotate and translate to world frame
        double x = info.origin.position.x + x_grid * cos_yaw - y_grid * sin_yaw;
        double y = info.origin.position.y + x_grid * sin_yaw + y_grid * cos_yaw;
        
        all_obs_.push_back(Vec2f(x, y));
      }
    }
  }
  
  double elapsed = (ros::Time::now() - start).toSec();
  ROS_INFO("Converted %zu obstacles in %.3f seconds (effective resolution: %.3f m)", 
           all_obs_.size(), elapsed, info.resolution * downsample_factor);
}

  void filterLocalObstacles() {
    obs2d_.clear();
    
    double center_x = (start_point_(0) + end_point_(0)) / 2.0;
    double center_y = (start_point_(1) + end_point_(1)) / 2.0;
    
    double path_length = (end_point_ - start_point_).norm();
    double search_radius = std::max(bbox_x_, bbox_y_) + path_length;
    double search_radius_sq = search_radius * search_radius;  // Avoid sqrt in loop
    
    for (const auto& obs : all_obs_) {
      double dx = obs(0) - center_x;
      double dy = obs(1) - center_y;
      double dist_sq = dx*dx + dy*dy;
      
      if (dist_sq <= search_radius_sq) {
        obs2d_.push_back(obs);
      }
    }
    
    ROS_INFO_THROTTLE(5.0, "Using %zu/%zu obstacles within %.2fm", 
                      obs2d_.size(), all_obs_.size(), search_radius);
  }

  void processDecomposition() {
    if (obs2d_.empty()) {
      ROS_WARN_THROTTLE(2.0, "No obstacles in local region!");
      return;
    }
    
    ROS_INFO_THROTTLE(2.0, "Processing: Start [%.2f, %.2f], End [%.2f, %.2f]",
                      start_point_(0), start_point_(1), 
                      end_point_(0), end_point_(1));
    
    // Create path
    vec_Vec2f path;
    path.push_back(start_point_);
    path.push_back(end_point_);
    
    // Decomposition
    EllipsoidDecomp2D decomp_util;
    decomp_util.set_obs(obs2d_);
    decomp_util.set_local_bbox(Vec2f(bbox_x_, bbox_y_));
    decomp_util.dilate(path);
    
    auto polys = decomp_util.get_polyhedrons();
    if (polys.empty()) {
      ROS_WARN("No polyhedron generated!");
      return;
    }
    
    // Publish
    decomp_ros_msgs::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(polys);
    poly_msg.header.frame_id = frame_id_;
    poly_msg.header.stamp = ros::Time::now();
    poly_pub_.publish(poly_msg);
    
    decomp_ros_msgs::EllipsoidArray es_msg = DecompROS::ellipsoid_array_to_ros(decomp_util.get_ellipsoids());
    es_msg.header.frame_id = frame_id_;
    es_msg.header.stamp = ros::Time::now();
    es_pub_.publish(es_msg);
    
    // Print constraints occasionally
    static int count = 0;
    if (++count % 50 == 0) {
      const auto pt_inside = (start_point_ + end_point_) / 2;
      LinearConstraint2D cs(pt_inside, polys[0].hyperplanes());
      std::cout << "\n=== Polyhedron Constraints ===" << std::endl;
      std::cout << "A:\n" << cs.A() << std::endl;
      std::cout << "b:\n" << cs.b().transpose() << std::endl;
    }
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "grid_decomp");
  ROS_INFO("Grid-based Decomposition Started");
  
  GridDecomp node;
  
  ros::Rate rate(10); // 10 Hz
  
  while (ros::ok()) {
    ros::spinOnce();
    node.run();
    rate.sleep();
  }
  
  return 0;
}