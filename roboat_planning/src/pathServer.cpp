#include "utility.h"

class PathServer : public ParamServer
{
public:

    ros::Timer pathUpdateTimer;

    ros::Publisher pubPathRaw;
    ros::Publisher pubPathSmooth;

    nav_msgs::Path pathRaw;
    nav_msgs::Path pathSmooth;

    tf::TransformListener listener;
    tf::StampedTransform transform;

    PointType robotPoint;

    double radius = 2;
    double length = 3;
    double width = radius * 2;

    PathServer()
    {
        pubPathRaw = nh.advertise<nav_msgs::Path> ("planning/server/path_blueprint_raw", 1);
        pubPathSmooth = nh.advertise<nav_msgs::Path> ("planning/server/path_blueprint_smooth", 1);

        pathUpdateTimer = nh.createTimer(ros::Duration(1.0), &PathServer::updatePath, this);
    };

    bool getRobotPosition()
    {
        try{listener.lookupTransform("map","base_link", ros::Time(0), transform); } 
        catch (tf::TransformException ex){ /*ROS_ERROR("Transfrom Failure.");*/ return false; }
        
        robotPoint.x = transform.getOrigin().x();
        robotPoint.y = transform.getOrigin().y();
        robotPoint.z = 0;

        return true;
    }

    void cdc_1()
    {
        pathRaw = nav_msgs::Path();
        // create raw path
        pathRaw.poses.push_back(createPoseStamped(-5, 0, 0));
        pathRaw.poses.push_back(createPoseStamped(5, 0, 0));
        // smooth path
        pathSmooth = processPath(pathRaw);
        ROS_INFO("CDC Test 1, straight line");
    }

    void cdc_2()
    {
        pathRaw = nav_msgs::Path();
        // create raw path
	double _x = 2.0;
	double _y = 1.0;
        pathRaw.poses.push_back(createPoseStamped(0+_x, 0+_y, 0));
        pathRaw.poses.push_back(createPoseStamped(length+_x, 0+_y, 0));

        for (double angle = 0; angle <= M_PI; angle += M_PI / 18)
        {
            float x = length + radius * sin(angle);
            float y = radius - radius * cos(angle);
            pathRaw.poses.push_back(createPoseStamped(x+_x, y+_y, 0));
        }

        pathRaw.poses.push_back(createPoseStamped(length+_x, width+_y, 0));
        pathRaw.poses.push_back(createPoseStamped(0+_x, width+_y, 0));
        // smooth path
        pathSmooth = processPath(pathRaw);
        ROS_INFO("CDC Test 2, half circle with straight line");
    }

    void intersection_path()
    {
        pathRaw = nav_msgs::Path();
        double R = 2.0;
        // Straight east segment
        pathRaw.poses.push_back(createPoseStamped(0, 0, 0));
        pathRaw.poses.push_back(createPoseStamped(4, 0, 0));
        // Quarter-circle arc: center (4, R), radius R, from -90deg to 0deg
        for (double angle = -M_PI/2; angle <= 0; angle += M_PI/18)
        {
            float x = 4 + R * cos(angle);
            float y = R + R * sin(angle);
            pathRaw.poses.push_back(createPoseStamped(x, -y, 0));
        }
        // Straight north segment
        pathRaw.poses.push_back(createPoseStamped(6, -9, 0));
        pathSmooth = processPath(pathRaw);
        ROS_INFO("Intersection path: east then north with arc corner");
    }

    void cdc_3()
    {
    	pathRaw = nav_msgs::Path();
        // create raw path
        double curve_length = 8;
        double curve_width = 2;
        double res = 0.1;
        double cycles = 8;

        for (double x = 0; x <= curve_length; x += res)
        {
        	double angle = (x - int(x / cycles) * cycles) / cycles * 2 * M_PI;
        	double y = curve_width * sin(angle);
        	pathRaw.poses.push_back(createPoseStamped(x+1, y+3, 0));
        	// cout << angle << " " << x << " " << y << endl;

        }

        // smooth path
        pathSmooth = processPath(pathRaw);
        ROS_INFO("CDC Test 3, swimming pool sine curve");
    }

    geometry_msgs::PoseStamped createPoseStamped(float x, float y, float z)
    {
        geometry_msgs::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp = ros::Time::now();
        pose.pose.position.x = x;
        pose.pose.position.y = y; 
        pose.pose.position.z = z;
        pose.pose.orientation = tf::createQuaternionMsgFromYaw(0);
        return pose;
    }

    void publishGlobalPath()
    {
        if (pubPathRaw.getNumSubscribers() != 0)
        {
            pathRaw.header.frame_id = "map";
            pathRaw.header.stamp = ros::Time::now();
            pubPathRaw.publish(pathRaw);
        }
        
        if (pubPathSmooth.getNumSubscribers() != 0)
        {
            pathSmooth.header.frame_id = "map";
            pathSmooth.header.stamp = ros::Time::now();
            pubPathSmooth.publish(pathSmooth);
        }
    }

    void updatePath(const ros::TimerEvent& event)
    {
        if (getRobotPosition() == false) return;
        PointType p;

        // cdc test 1
        // p.x = -5; p.y = 0; p.z = 0;
        // if (pointDistance(robotPoint, p) < 1.0)
        //     cdc_1();
        // else
        //     return;

        // intersection test
        p.x = 0; p.y = 0; p.z = 0;
        if (pointDistance(robotPoint, p) < 1.0)
            intersection_path();
        else
            return;

        // cdc test 3
        //p.x = 1; p.y = 3; p.z = 0;
        //if (pointDistance(robotPoint, p) < 0.5)
          //  cdc_3();
        //else
          //  return;

        publishGlobalPath();
    }
};


int main(int argc, char** argv){

    ros::init(argc, argv, "roboat_planning");
    
    PathServer ps;

    ROS_INFO("\033[1;32m----> roboat_planning: Path Server Started.\033[0m");

    ros::spin();

    return 0;
}
