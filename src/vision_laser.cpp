#include <ctime>
#include <string>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <iterator>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/LaserScan.h>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;

#define pi 3.1415926

typedef struct PPoint_
{
	double x;
	double y;
	double z;
} ppoint;

typedef struct Point_
{
	double x;
	double y;
	Point_(double a, double b)
	{
		x = a;
		y = b;
	}
	Point_()
	{
		x = 0;
		y = 0;
	}
} spoint;

Rect show;

//储存雷达点的像素坐标和二维坐标 poin2i为像素坐标 bool表示是否为边缘点 第二个Point2i储存空间坐标
typedef pair<pair<Point2i, bool>, spoint> laser_coor;

Point2i camera_2_rgb(ppoint point_camera);

double cx_ = 239.9 * 2;
double cy_ = 131.975 * 2;
double fx_ = 268.225 * 2;
double fy_ = 268.575 * 2;

//double laser2robot_x = 0.16;
//double laser2robot_y = 0.0;

double kinect2robot_x = 0.132;
double kinect2robot_y = -0.095;

double laser2kinect_x = 0.035;
double laser2kinect_y = -0.095;
double laser2kinect_z = 0.5;

//相机坐标系转像素坐标系
Point2i camera_2_rgb(ppoint point_camera)
{
	Point2i point_;
	point_.x = (fx_ * point_camera.x / point_camera.z + cx_);
	point_.y = (fy_ * point_camera.y / point_camera.z + cy_);
	return point_;
}


//激光雷达坐标系转换为像素坐标系 并判断激光点是否为边缘点
void laser_to_rgb(const sensor_msgs::LaserScanConstPtr &scan, vector<laser_coor> &laserPoint)
{

	double angle = 0;
	double increment_angle = 0.5 / 180 * 3.1415;
	//每一束激光
	for (int id = scan->ranges.size() - 1; id >= 0; id--) //倒置
	{
		double dist = scan->ranges[id];
		if (std::isinf(dist) || std::isnan(dist))
		{
			angle += increment_angle;
			continue;
		}
		double laser_x = dist * sin(angle);
		double laser_y = -dist * cos(angle);

		angle += increment_angle;
		ppoint point_camera;
		point_camera.z = (laser_x + laser2kinect_x);
		point_camera.x = -laser_y + laser2kinect_y;
		point_camera.y = laser2kinect_z;
		Point2i point_ = camera_2_rgb(point_camera);
		if ((point_.x > 0) && (point_.y > 0))
		{
			laserPoint.push_back(make_pair(make_pair(point_, false), spoint(laser_x, laser_y)));
		}
	}

}          

//接收到传感器数据后的回调函数
void combineCallback(const sensor_msgs::ImageConstPtr &rgb_image_qhd, const sensor_msgs::LaserScanConstPtr &laser_data)
{

	vector<laser_coor> laserPoint;
	laser_to_rgb(laser_data, laserPoint);

	cv_bridge::CvImagePtr rgb_ptr;
	cv::Mat resize_rgb_mat; //缩小尺寸后的图片
	int height;
	int width;

	try
	{
		rgb_ptr = cv_bridge::toCvCopy(rgb_image_qhd, sensor_msgs::image_encodings::BGR8);
		width = rgb_ptr->image.cols / 2;
		height = rgb_ptr->image.rows / 2;
		cv::resize(rgb_ptr->image, resize_rgb_mat, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
		imshow("origal", rgb_ptr->image);
	}
	catch (cv_bridge::Exception &e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		ROS_INFO("结束该帧数据处理, 等待下帧数据.....");
		return;
	}

    Mat LaserMat = rgb_ptr->image.clone();

    for (int i = 0; i < laserPoint.size(); i++)
	{

		circle(LaserMat, laserPoint[i].first.first, 1, Scalar(0, 0, 255), 1, 1); //红色显示边缘点
	}

	imshow("object", LaserMat);
	waitKey(1);

	//ROS_INFO("数据处理完成.等待下帧数据.\n");
}

int main(int argc, char **argv)
{

	ros::init(argc, argv, "occ_xc");
	ros::NodeHandle nh_;
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::LaserScan> rgb_laser_syncpolicy;
	//原图为1920*1080，rect为/2之后的了
	message_filters::Subscriber<sensor_msgs::Image> *rgb_image_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(nh_, "kinect2/qhd/image_color_rect", 1);
	message_filters::Subscriber<sensor_msgs::LaserScan> *laser_sub_ = new message_filters::Subscriber<sensor_msgs::LaserScan>(nh_, "/scan", 1);
	message_filters::Synchronizer<rgb_laser_syncpolicy> *sync_ = new message_filters::Synchronizer<rgb_laser_syncpolicy>(rgb_laser_syncpolicy(20), *rgb_image_sub_, *laser_sub_);
	sync_->registerCallback(boost::bind(&combineCallback, _1, _2));
	ros::spin();
	return 0;
}