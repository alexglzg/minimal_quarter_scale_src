#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <tf/transform_listener.h>
#include <pcl_ros/transforms.h>

typedef pcl::PointXYZ PointType;

class SimplePCLFilter {
private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_cloud_;
  ros::Publisher pub_cloud_;
  ros::Publisher pub_scan_;
  tf::TransformListener tf_listener_;
  
  // Parameters
  double voxel_size_;
  double height_min_;
  double height_max_;
  double range_min_;
  double range_max_;

  // LaserScan parameters
  int num_beams_;
  std::string scan_frame_;
  
public:
  SimplePCLFilter() : nh_("~") {
    // Load parameters
    nh_.param("voxel_size", voxel_size_, 0.1);
    nh_.param("height_min", height_min_, -0.5);
    nh_.param("height_max", height_max_, 2.0);
    nh_.param("range_min", range_min_, 0.1);
    nh_.param("range_max", range_max_, 10.0);

    // LaserScan parameters
    nh_.param("num_beams", num_beams_, 720);
    nh_.param<std::string>("scan_frame", scan_frame_, "base_link");
    
    ROS_INFO("PCL Filter Parameters:");
    ROS_INFO("  Voxel size: %.2f", voxel_size_);
    ROS_INFO("  Height range: [%.2f, %.2f]", height_min_, height_max_);
    ROS_INFO("  Range: [%.2f, %.2f]", range_min_, range_max_);
    ROS_INFO("  LaserScan beams: %d, frame: %s", num_beams_, scan_frame_.c_str());
    
    // Subscribers and Publishers
    sub_cloud_ = nh_.subscribe("/points_raw", 1, &SimplePCLFilter::cloudCallback, this);
    pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 1);
    pub_scan_  = nh_.advertise<sensor_msgs::LaserScan>("/filtered_scan", 1);
  }
  
  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    bool need_cloud = (pub_cloud_.getNumSubscribers() > 0);
    bool need_scan  = (pub_scan_.getNumSubscribers() > 0);
    if (!need_cloud && !need_scan) return;
    
    // Get transform from sensor frame to map
    tf::StampedTransform transform;
    try {
      tf_listener_.waitForTransform("map", cloud_msg->header.frame_id, 
                                    cloud_msg->header.stamp, ros::Duration(0.1));
      tf_listener_.lookupTransform("map", cloud_msg->header.frame_id, 
                                   cloud_msg->header.stamp, transform);
    } catch (tf::TransformException& ex) {
      ROS_WARN_THROTTLE(5.0, "TF lookup failed: %s", ex.what());
      return;
    }
    
    // Convert to PCL
    pcl::PointCloud<PointType>::Ptr cloud_in(new pcl::PointCloud<PointType>());
    pcl::fromROSMsg(*cloud_msg, *cloud_in);
    
    // Transform to map frame
    pcl::PointCloud<PointType>::Ptr cloud_map(new pcl::PointCloud<PointType>());
    pcl_ros::transformPointCloud(*cloud_in, *cloud_map, transform);
    
    // Get robot position in map frame
    double robot_x = transform.getOrigin().x();
    double robot_y = transform.getOrigin().y();
    double robot_z = transform.getOrigin().z();
    
    // Filter by height (relative to robot)
    pcl::PointCloud<PointType>::Ptr cloud_height(new pcl::PointCloud<PointType>());
    pcl::PassThrough<PointType> pass_z;
    pass_z.setInputCloud(cloud_map);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(robot_z + height_min_, robot_z + height_max_);
    pass_z.filter(*cloud_height);
    
    // Filter by range from robot
    pcl::PointCloud<PointType>::Ptr cloud_filtered(new pcl::PointCloud<PointType>());
    for (const auto& point : cloud_height->points) {
      double dx = point.x - robot_x;
      double dy = point.y - robot_y;
      double range = sqrt(dx*dx + dy*dy);
      
      if (range >= range_min_ && range <= range_max_) {
        cloud_filtered->push_back(point);
      }
    }
    
    // Downsample
    pcl::PointCloud<PointType>::Ptr cloud_ds(new pcl::PointCloud<PointType>());
    pcl::VoxelGrid<PointType> voxel_filter;
    voxel_filter.setInputCloud(cloud_filtered);
    voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    voxel_filter.filter(*cloud_ds);
    
    // Publish PointCloud2 (unchanged)
    if (need_cloud) {
      sensor_msgs::PointCloud2 output;
      pcl::toROSMsg(*cloud_ds, output);
      output.header.stamp = cloud_msg->header.stamp;
      output.header.frame_id = "map";
      pub_cloud_.publish(output);
    }

    // Publish LaserScan
    if (need_scan) {
      publishLaserScan(cloud_ds, robot_x, robot_y, transform, cloud_msg->header.stamp);
    }
  }

private:
  void publishLaserScan(const pcl::PointCloud<PointType>::Ptr& cloud,
                        double robot_x, double robot_y,
                        const tf::StampedTransform& transform,
                        const ros::Time& stamp)
  {
    // Get robot yaw from transform
    double roll, pitch, yaw;
    transform.getBasis().getRPY(roll, pitch, yaw);

    // Build scan message
    sensor_msgs::LaserScan scan;
    scan.header.stamp = stamp;
    scan.header.frame_id = scan_frame_;
    scan.angle_min = -M_PI;
    scan.angle_max =  M_PI;
    scan.angle_increment = 2.0 * M_PI / num_beams_;
    scan.range_min = range_min_;
    scan.range_max = range_max_;
    scan.time_increment = 0.0;
    scan.scan_time = 0.0;
    scan.ranges.assign(num_beams_, std::numeric_limits<float>::infinity());

    // Project each point into an angular bin relative to robot pose
    for (const auto& pt : cloud->points) {
      double dx = pt.x - robot_x;
      double dy = pt.y - robot_y;
      double range = sqrt(dx * dx + dy * dy);

      if (range < range_min_ || range > range_max_) continue;

      // Angle in map frame, then relative to robot heading
      double angle = atan2(dy, dx) - yaw;

      // Wrap to [-pi, pi)
      while (angle >= M_PI)  angle -= 2.0 * M_PI;
      while (angle < -M_PI)  angle += 2.0 * M_PI;

      int idx = static_cast<int>((angle - scan.angle_min) / scan.angle_increment);
      if (idx < 0 || idx >= num_beams_) continue;

      // Keep closest hit per bin
      if (range < scan.ranges[idx]) {
        scan.ranges[idx] = range;
      }
    }

    pub_scan_.publish(scan);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_filter");
  ROS_INFO("Simple PCL Filter started");
  
  SimplePCLFilter filter;
  ros::spin();
  
  return 0;
}