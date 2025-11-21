#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
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
  tf::TransformListener tf_listener_;
  
  // Parameters
  double voxel_size_;
  double height_min_;
  double height_max_;
  double range_min_;
  double range_max_;
  
public:
  SimplePCLFilter() : nh_("~") {
    // Load parameters
    nh_.param("voxel_size", voxel_size_, 0.1);
    nh_.param("height_min", height_min_, -0.5);
    nh_.param("height_max", height_max_, 2.0);
    nh_.param("range_min", range_min_, 0.1);
    nh_.param("range_max", range_max_, 10.0);
    
    ROS_INFO("PCL Filter Parameters:");
    ROS_INFO("  Voxel size: %.2f", voxel_size_);
    ROS_INFO("  Height range: [%.2f, %.2f]", height_min_, height_max_);
    ROS_INFO("  Range: [%.2f, %.2f]", range_min_, range_max_);
    
    // Subscribers and Publishers
    sub_cloud_ = nh_.subscribe("/points_raw", 1, &SimplePCLFilter::cloudCallback, this);
    pub_cloud_ = nh_.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 1);
  }
  
  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    if (pub_cloud_.getNumSubscribers() == 0) return;
    
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
    // pcl_ros::transformPointCloud("map", transform, *cloud_in, *cloud_map);
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
    
    // Publish
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud_ds, output);
    output.header.stamp = cloud_msg->header.stamp;
    output.header.frame_id = "map";
    pub_cloud_.publish(output);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "pcl_filter");
  ROS_INFO("Simple PCL Filter started");
  
  SimplePCLFilter filter;
  ros::spin();
  
  return 0;
}