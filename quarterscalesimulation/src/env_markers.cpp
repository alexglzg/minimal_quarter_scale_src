/** ----------------------------------------------------------------------------
 * @file:     env_markers.cpp
 * @brief:    RViz arrow + text markers for wind and current speed/direction,
 *            drawn at a fixed position (not attached to the vessel).
 * ---------------------------------------------------------------------------*/

#include "ros/ros.h"
#include "geometry_msgs/Pose2D.h"
#include "visualization_msgs/MarkerArray.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sstream>
#include <iomanip>

class EnvMarkers
{
public:
    EnvMarkers()
    {
        ros::NodeHandle pn("~");
        pn.param("frame_id", frame_id, std::string("map"));
        pn.param("wind_marker_x", wind_x, 3.0);
        pn.param("wind_marker_y", wind_y, 4.0);
        pn.param("wind_marker_z", wind_z, 1.0);
        pn.param("current_marker_x", current_x, 3.0);
        pn.param("current_marker_y", current_y, 2.0);
        pn.param("current_marker_z", current_z, 1.0);

        wind_sub = n.subscribe("/roboat_wind", 1, &EnvMarkers::windCallback, this);
        current_sub = n.subscribe("/roboat_currents", 1, &EnvMarkers::currentCallback, this);
        marker_pub = n.advertise<visualization_msgs::MarkerArray>("/env_markers", 1);
    }

    void windCallback(const geometry_msgs::Pose2D::ConstPtr& msg)
    {
        publishVector("wind", wind_x, wind_y, wind_z, msg->x, msg->theta, 0.0, 1.0, 1.0);
    }

    void currentCallback(const geometry_msgs::Pose2D::ConstPtr& msg)
    {
        publishVector("current", current_x, current_y, current_z, msg->x, msg->theta, 0.2, 0.6, 1.0);
    }

private:
    void publishVector(const std::string& ns, double px, double py, double pz,
                        double speed, double dir_ned, double r, double g, double b)
    {
        // NED -> map/ROS frame: mirror across x, same convention qs_dynamics uses for psi.
        double yaw = -dir_ned;

        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);

        visualization_msgs::Marker arrow;
        arrow.header.stamp = ros::Time::now();
        arrow.header.frame_id = frame_id;
        arrow.ns = ns;
        arrow.id = 0;
        arrow.type = visualization_msgs::Marker::ARROW;
        arrow.action = visualization_msgs::Marker::ADD;
        arrow.pose.position.x = px;
        arrow.pose.position.y = py;
        arrow.pose.position.z = pz;
        arrow.pose.orientation = tf2::toMsg(q);
        arrow.scale.x = 0.5 + speed;  // shaft length grows with speed
        arrow.scale.y = 0.15;         // head diameter
        arrow.scale.z = 0.15;         // head length
        arrow.color.a = 1.0;
        arrow.color.r = r;
        arrow.color.g = g;
        arrow.color.b = b;

        std::ostringstream text;
        text << ns << ": " << std::fixed << std::setprecision(2) << speed << " m/s";

        visualization_msgs::Marker label;
        label.header = arrow.header;
        label.ns = ns;
        label.id = 1;
        label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        label.action = visualization_msgs::Marker::ADD;
        label.pose.position.x = px;
        label.pose.position.y = py;
        label.pose.position.z = pz + 0.4;
        label.scale.z = 0.3;
        label.color = arrow.color;
        label.text = text.str();

        visualization_msgs::MarkerArray array;
        array.markers.push_back(arrow);
        array.markers.push_back(label);
        marker_pub.publish(array);
    }

    ros::NodeHandle n;
    ros::Subscriber wind_sub;
    ros::Subscriber current_sub;
    ros::Publisher marker_pub;

    std::string frame_id;
    double wind_x, wind_y, wind_z;
    double current_x, current_y, current_z;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "env_markers");
    EnvMarkers env_markers;
    ros::spin();
    return 0;
}
