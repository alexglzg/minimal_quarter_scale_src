#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <nav_msgs/Odometry.h>
#include <decomp_ros_utils/data_ros_utils.h>
#include <decomp_util/ellipsoid_decomp.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

class SimpleDecomp {
private:
  ros::NodeHandle nh_;
  ros::Subscriber cloud_sub_;
  ros::Subscriber odom_sub_;
  ros::Publisher poly_pub_;
  ros::Publisher es_pub_;
  
  double forward_distance_;
  double backward_distance_;
  double bbox_x_;
  double bbox_y_;
  
  bool odom_received_;
//   bool cloud_received_;
  
  Vec2f start_point_;
  Vec2f end_point_;
  vec_Vec2f obs2d_;
  std::string frame_id_;

  nav_msgs::Odometry::ConstPtr latest_odom_;

public:
  SimpleDecomp() : nh_("~"), odom_received_(false) {
    // Get parameters
    nh_.param("forward_distance", forward_distance_, 0.5);
    nh_.param("backward_distance", backward_distance_, 0.5);
    nh_.param("bbox_x", bbox_x_, 5.0);
    nh_.param("bbox_y", bbox_y_, 5.0);
    
    ROS_INFO("Forward distance: %.2f m", forward_distance_);
    ROS_INFO("Backward distance: %.2f m", backward_distance_);
    ROS_INFO("Bounding box: [%.2f, %.2f]", bbox_x_, bbox_y_);
    
    // Publishers
    poly_pub_ = nh_.advertise<decomp_ros_msgs::PolyhedronArray>("/polyhedron_array", 1, true);
    es_pub_ = nh_.advertise<decomp_ros_msgs::EllipsoidArray>("/ellipsoid_array", 1, true);
    
    // Subscribers
    odom_sub_ = nh_.subscribe("/odometry/filtered", 1, &SimpleDecomp::odomCallback, this);
    cloud_sub_ = nh_.subscribe("/filtered_cloud", 1, &SimpleDecomp::cloudCallback, this);
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& odom_msg) {
    // Just store the latest odometry, don't process
    latest_odom_ = odom_msg;
    odom_received_ = true;
  }

  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {

    if (!odom_received_) {
      ROS_WARN_THROTTLE(1.0, "Waiting for odometry...");
      return;
    }

    // Convert PointCloud2 to PointCloud
    sensor_msgs::PointCloud cloud;
    sensor_msgs::convertPointCloud2ToPointCloud(*cloud_msg, cloud);
    
    // Convert to vec_Vec3f
    vec_Vec3f obs = DecompROS::cloud_to_vec(cloud);
    
    // Convert to 2D (take x,y)
    obs2d_.clear();
    for(const auto& it: obs)
      obs2d_.push_back(it.topRows<2>());

    frame_id_ = cloud_msg->header.frame_id;

    // Update start/end points from latest odometry
    updatePointsFromOdom();

    // Process decomposition (only when new cloud arrives)
    processDecomposition();
    
  }
  
  void updatePointsFromOdom() {
    // Get robot position
    double x = latest_odom_->pose.pose.position.x;
    double y = latest_odom_->pose.pose.position.y;
    
    // Get robot orientation (yaw)
    tf2::Quaternion q(
      latest_odom_->pose.pose.orientation.x,
      latest_odom_->pose.pose.orientation.y,
      latest_odom_->pose.pose.orientation.z,
      latest_odom_->pose.pose.orientation.w
    );
    
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
    
    // Calculate points along x-axis in robot frame
    // Backward point (behind robot)
    start_point_(0) = x - backward_distance_ * cos(yaw);
    start_point_(1) = y - backward_distance_ * sin(yaw);
    
    // Forward point (in front of robot)
    end_point_(0) = x + forward_distance_ * cos(yaw);
    end_point_(1) = y + forward_distance_ * sin(yaw);
  }


  void processDecomposition() {
    
    ROS_INFO("Processing decomposition...");
    ROS_INFO("Start point (backward): [%.2f, %.2f]", start_point_(0), start_point_(1));
    ROS_INFO("End point (forward): [%.2f, %.2f]", end_point_(0), end_point_(1));
    ROS_INFO("Distance: %.2f m", (end_point_ - start_point_).norm());
    
    // Create path with two points
    vec_Vec2f path;
    path.push_back(start_point_);
    path.push_back(end_point_);
    
    // Perform ellipsoid decomposition
    EllipsoidDecomp2D decomp_util;
    decomp_util.set_obs(obs2d_);
    decomp_util.set_local_bbox(Vec2f(bbox_x_, bbox_y_));
    decomp_util.dilate(path);
    
    // Get the single polyhedron
    auto polys = decomp_util.get_polyhedrons();
    if (polys.empty()) {
      ROS_WARN("No polyhedron generated!");
      return;
    }
    
    ROS_INFO("Generated %zu polyhedron(s)", polys.size());
    
    // Publish polyhedron
    decomp_ros_msgs::PolyhedronArray poly_msg = DecompROS::polyhedron_array_to_ros(polys);
    poly_msg.header.frame_id = frame_id_;
    poly_msg.header.stamp = ros::Time::now();
    poly_pub_.publish(poly_msg);
    
    // Publish ellipsoid
    decomp_ros_msgs::EllipsoidArray es_msg = DecompROS::ellipsoid_array_to_ros(decomp_util.get_ellipsoids());
    es_msg.header.frame_id = frame_id_;
    es_msg.header.stamp = ros::Time::now();
    es_pub_.publish(es_msg);
    
    // Print constraint information (Ax <= b)
    const auto pt_inside = (start_point_ + end_point_) / 2;
    LinearConstraint2D cs(pt_inside, polys[0].hyperplanes());
    
    std::cout << "\n=== Polyhedron Constraints (Ax <= b) ===" << std::endl;
    std::cout << "A:\n" << cs.A() << std::endl;
    std::cout << "b:\n" << cs.b().transpose() << std::endl;
    
    std::cout << "\nStart point: " << start_point_.transpose();
    std::cout << (cs.inside(start_point_) ? " is inside!" : " is outside!") << std::endl;
    
    std::cout << "End point: " << end_point_.transpose();
    std::cout << (cs.inside(end_point_) ? " is inside!" : " is outside!") << std::endl;
    
    ROS_INFO("Decomposition complete!");
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "simple_decomp");
  SimpleDecomp node;
  ros::spin();
  return 0;
}