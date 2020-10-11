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

//储存雷达点的像素坐标和二维坐标 poin2i为像素坐标 bool表示是否为边缘点 第二个Point2i储存空间坐标
typedef pair<pair<Point2i, bool>, spoint> laser_coor;

Point2i camera_2_rgb(ppoint point_camera);
bool isEdgePoint(spoint p1, spoint p2, spoint p3, double max);
Point getCrossPoint(vector<int> LineA, vector<int> LineB);
vector<int> getLineParam(Point start, Point end);
float lines_orientation(Point begin, Point end, int flag);
vector<Vec4i> houghlinedetect(Mat &roiImg);
void laser_to_rgb(const sensor_msgs::LaserScanConstPtr &scan, vector<laser_coor> &laserPoint);
void pointInRoi(Rect roi, vector<laser_coor> &allPoint, vector<laser_coor> &inPoint);
vector<laser_coor> getIndex(vector<laser_coor> &Point);
float getHorizonAngle(vector<laser_coor> &Point);
void combineCallback(const sensor_msgs::ImageConstPtr &rgb_image_qhd, const sensor_msgs::LaserScanConstPtr &laser_data);
void computerPiexDistance(vector<laser_coor> &Point, float result[2]);
void measurement(Mat &roiImg, vector<laser_coor> &laserPoint, int label, int x, int w);

string modelConfiguration = "/home/wode/configuration_folder/trash_ssd/opencv_mbssd_indoor/MobileNetSSD_deploy.prototxt";
string modelBinary = "/home/wode/configuration_folder/trash_ssd/opencv_mbssd_indoor/mobilenet_indoorone_120000.caffemodel";
string *class_array = new string[9]{"background", "window", "bed", "aricondition", "sofa", "chair", "cabinet", "trash", "door"};
Net net = readNetFromCaffe(modelConfiguration, modelBinary);

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
double laser2kinect_z = 0.3;

//计算每个像素点的实际距离(cm)
void computerPiexDistance(vector<laser_coor> &Point, float result[2])
{
	//float result[2];
	double dis = 0;
	double all_piex = 0;
	//截取中间激光点进行单位像素距离计算
	for (int i = Point.size()*0.25; i < Point.size()*0.75; i++)
	{
		float thita = lines_orientation(Point[i].first.first, Point[i - 1].first.first, 0);
		all_piex += fabs(sqrt(pow((Point[i - 1].first.first.x - Point[i].first.first.x), 2) + pow((Point[i - 1].first.first.y - Point[i].first.first.y), 2))*cos(thita));
		dis += fabs(sqrt(pow((Point[i - 1].second.x - Point[i].second.x), 2) + pow((Point[i - 1].second.y - Point[i].second.y), 2))*cos(thita));
	}
	result[0] = dis / all_piex * 100;

	for (int i = Point.size()*0.25; i < Point.size()*0.75; i++)
	{
		float thita = lines_orientation(Point[i].first.first, Point[i - 1].first.first, 0);
		all_piex += fabs(sqrt(pow((Point[i - 1].first.first.x - Point[i].first.first.x), 2) + pow((Point[i - 1].first.first.y - Point[i].first.first.y), 2))*sin(thita));
		dis += fabs(sqrt(pow((Point[i - 1].second.x - Point[i].second.x), 2) + pow((Point[i - 1].second.y - Point[i].second.y), 2))*sin(thita));
	}
	result[1] = dis / all_piex * 100;
}

//相机坐标系转像素坐标系
Point2i camera_2_rgb(ppoint point_camera)
{
	Point2i point_;
	point_.x = (fx_ * point_camera.x / point_camera.z + cx_);
	point_.y = (fy_ * point_camera.y / point_camera.z + cy_);
	return point_;
}

//判断点p2是否为边缘点 max为曲率阈值
bool isEdgePoint(spoint p1, spoint p2, spoint p3, double max)
{

	double a = (p1.x + p2.x) * (p2.x - p1.x) * (p3.y - p2.y);
	double b = (p1.x + p2.x) * (p3.x - p2.x) * (p2.y - p1.y);
	double c = (p1.y - p3.y) * (p2.y - p1.y) * (p3.y - p2.y);
	double d = 2 * ((p2.x - p1.x) * (p3.y - p2.y) - (p3.x - p2.x) * (p2.y - p1.y));
	double e = (p1.y + p2.y) * (p2.y - p1.y) * (p3.x - p2.x);
	double f = (p3.y + p2.y) * (p3.y - p2.y) * (p2.x - p1.x);
	double g = (p1.x - p3.x) * (p3.x - p1.x) * (p3.x - p2.x);

	double x0 = (a - b + c) / d;
	double y0 = (e - f + g) / (-d);

	double ri = 1 / sqrt((x0 - p2.x) * (x0 - p2.x) + (y0 - p2.y) * (y0 - p2.y)); //计算曲率
	//cout<<"ri is "<< ri<<endl;
	if (ri > max)
		return true;
	else
		return false;
}

//获得两直线交点
//传入的参数为两条直线的参数
Point getCrossPoint(vector<int> LineA, vector<int> LineB)
{

	int m = LineA[0] * LineB[1] - LineA[1] * LineB[0];

	if (m == 0)
	{
		cout << "无交点" << endl;
		return Point(-1, -1);
	}
	else
	{
		int x = (LineB[2] * LineA[1] - LineA[2] * LineB[1]) / m;
		int y = (LineA[2] * LineB[0] - LineB[2] * LineA[0]) / m;
		return Point(x, y);
	}
}

//获得直线的一般式方程参数Ax+By+C=0
vector<int> getLineParam(Point start, Point end)
{

	vector<int> result;
	result.push_back(end.y - start.y);
	result.push_back(start.x - end.x);
	result.push_back(end.x * start.y - start.x * end.y);
	return result;
}

//计算直线倾斜角度
float lines_orientation(Point begin, Point end, int flag)
{

	float kLine, lines_arctan;

	//根据直线端点计算斜率
	if (begin.x == end.x)
	{
		lines_arctan = pi / 2;
	}
	else
	{
		kLine = (begin.y - end.y) / (begin.x - end.x);
		lines_arctan = atan(kLine); //反正切获取夹角
	}

	//以角度或弧度形式返回夹角
	if (flag == 0)
	{
		return lines_arctan; //弧度
	}
	else
	{
		return lines_arctan * 180.0 / pi; //角度
	}
}

//识别物体轮廓并画框显示
//利用霍夫变换检测直线然后选择最外侧的直线作为轮廓线
vector<Vec4i> houghlinedetect(Mat &roiImg)
{

	int threshold_value = 135;

	Mat dst;
	//使用边缘检测将图片二值化
	Canny(roiImg, dst, threshold_value, 2 * threshold_value, 3, false);

	vector<Vec4i> lines;								   //存储直线数据
	HoughLinesP(dst, lines, 1, CV_PI / 180.0, 30, 30, 10); //源图需要是二值图像，HoughLines也是一样

	imshow("canny", dst);

	return lines;
}

void measurement(Mat &roiImg, vector<laser_coor> &laserPoint, int label, int x, int w)
{

	Mat LaserMat = roiImg.clone();
	vector<Vec4i> lines = houghlinedetect(roiImg);

	int rangeXMIN, rangeXMAX;
	auto di = getIndex(laserPoint);
	if (di.size() == 0)
	{
		ROS_INFO_STREAM("激光雷达数据过少,无法完成检测!");
		return;
	}
	else
	{
		rangeXMAX = di[0].first.first.x;
		rangeXMIN = di[di.size() - 1].first.first.x;
	}

	//circle(LaserMat, Point2i(rangeXMIN, di[0].first.first.y), 5, Scalar(255, 0, 0), 1, 1);
	//circle(LaserMat, Point2i(rangeXMAX, di[di.size()-1].first.first.y), 5, Scalar(0, 255, 0), 1, 1);
	//imshow("laser",  LaserMat);//显示可视化的激光雷达点

	auto Hangle = getHorizonAngle(di);
	float dis[2];
	computerPiexDistance(di, dis);
	//ROS_INFO_STREAM("angle is "<<Hangle);
	//ROS_INFO_STREAM("dis is "<<dis);
	//ROS_INFO_STREAM("all size is "<<inPoint.size()<<", cut size is "<<di.size());

	vector<vector<Point>> H_Line, V_Line; //储存水平线与竖直线 储存每条线的起点 中点和终点

	for (size_t i = 0; i < lines.size(); i++)
	{
		int maxAngle;
		Vec4i Points = lines[i];
		Point begin(Points[0], Points[1]), end(Points[2], Points[3]);
		float angle = fabs(lines_orientation(begin, end, 1));
		if (label == 7 || label == 8)
		{
			Hangle = 0;
			maxAngle = 30;
		}
		else
		{
			maxAngle = 30;
		}
		//ROS_INFO_STREAM("Han is "<<Hangle<<", angle is "<<angle<<", max is "<<maxAngle);
		if (fabs(Hangle - angle) <= maxAngle)
		{
			if ((((begin.x + end.x) / 2) >= rangeXMIN) && (((begin.x + end.x) / 2) <= rangeXMAX))
			{
				vector<Point> tmp;
				tmp.push_back(begin);
				tmp.push_back(end);
				tmp.push_back(Point((begin.x + end.x) / 2, (begin.y + end.y) / 2));
				H_Line.push_back(tmp);
			}
		}
		else if (angle > 85)
		{
			//if((((begin.x + end.x) / 2) >= rangeXMIN) && (((begin.x + end.x) / 2) <= rangeXMAX))
			//{
			vector<Point> tmp;
			tmp.push_back(begin);
			tmp.push_back(end);
			tmp.push_back(Point((begin.x + end.x) / 2, (begin.y + end.y) / 2));
			V_Line.push_back(tmp);
			//}
		}
	}

	// for(int i = 0; i<di.size(); i++)
	// {
	// 	circle(roiImg, Point(laserPoint[i].first.first.x, laserPoint[i].first.first.y), 2, Scalar(0, 0, 255), 2, LINE_AA);
	// }
	// imshow("circle",  roiImg);

	if (H_Line.size() < 2 || V_Line.size() < 2)
	{
		ROS_INFO_STREAM("检测到直线特征过少!");
		ROS_INFO_STREAM("H line size is " << H_Line.size());
		ROS_INFO_STREAM("V line size is " << V_Line.size());
		return;
	}

	ROS_INFO_STREAM("-------------------------------");
	ROS_INFO_STREAM("检测到" << class_array[label] << ",开始测量......");

	int top = 0, left = 0, bottom = 0, right = 0; //保存边框点的索引
	for (int i = 0; i < H_Line.size(); ++i)
	{ 
		//查找水平边缘线
		if (H_Line[i][2].y < H_Line[top][2].y)
			top = i;
		if (H_Line[i][2].y > H_Line[bottom][2].y)
			bottom = i;
	}

	int rDiff = 10000, lDiff = 10000;
	for (int i = 0; i < V_Line.size(); ++i)
	{ 
		//查找竖直边缘线
		if (fabs(V_Line[i][2].x - rangeXMIN) < lDiff)
		{
			left = i;
			lDiff = fabs(V_Line[i][2].x - rangeXMIN);
		}
		if (fabs(V_Line[i][2].x - rangeXMAX) < rDiff)
		{
			right = i;
			rDiff = fabs(V_Line[i][2].x - rangeXMAX);
		}
	}

	//将边缘线延长
	vector<int> paramA = getLineParam(H_Line[top][0], H_Line[top][1]);
	vector<int> paramB = getLineParam(V_Line[left][0], V_Line[left][1]);
	vector<int> paramC = getLineParam(H_Line[bottom][0], H_Line[bottom][1]);
	vector<int> paramD = getLineParam(V_Line[right][0], V_Line[right][1]);

	Mat gray_dst = roiImg.clone();

	//可视化激光点
	for (int i = 0; i < laserPoint.size(); i++)
	{
		if (laserPoint[i].first.second)
		{
			circle(LaserMat, laserPoint[i].first.first, 2, Scalar(0, 0, 255), 2, 1); //红色显示边缘点
		}
		else
		{
			circle(LaserMat, laserPoint[i].first.first, 1, Scalar(0, 255, 0), 1, 1); //绿色显示平面点
		}
	}

	//绘制轮廓
	Point crossPointTL = getCrossPoint(paramA, paramB);
	line(gray_dst, H_Line[top][0], crossPointTL, Scalar(0, 0, 255), 1, LINE_AA);
	line(gray_dst, V_Line[left][0], crossPointTL, Scalar(0, 0, 255), 1, LINE_AA);

	auto crossPointTR = getCrossPoint(paramA, paramD);
	line(gray_dst, H_Line[top][0], crossPointTR, Scalar(0, 0, 255), 1, LINE_AA);
	line(gray_dst, V_Line[right][0], crossPointTR, Scalar(0, 0, 255), 1, LINE_AA);

	auto crossPointBL = getCrossPoint(paramC, paramB);
	line(gray_dst, H_Line[bottom][0], crossPointBL, Scalar(0, 0, 255), 1, LINE_AA);
	line(gray_dst, V_Line[left][0], crossPointBL, Scalar(0, 0, 255), 1, LINE_AA);

	auto crossPointBR = getCrossPoint(paramC, paramD);
	line(gray_dst, H_Line[bottom][0], crossPointBR, Scalar(0, 0, 255), 1, LINE_AA);
	line(gray_dst, V_Line[right][0], crossPointBR, Scalar(0, 0, 255), 1, LINE_AA);

	float hi = sqrt((crossPointTL - crossPointBL).dot(crossPointTL - crossPointBL)) * dis[1] * 0.8;
	float wh = sqrt((crossPointBL - crossPointBR).dot(crossPointBL - crossPointBR)) * dis[0] * 0.8;

	ROS_INFO_STREAM("higet is " << hi << "cm, width is " << wh << "cm");
	ROS_INFO_STREAM("-------------------------------\n");

	char tx[20];
	sprintf(tx, "%.2f", hi);
	putText(gray_dst, tx, (crossPointTL + crossPointBL) / 2, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1.8);
	memset(tx, 0, 20);
	sprintf(tx, "%.2f", wh);
	putText(gray_dst, tx, (crossPointBR + crossPointBL) / 2, FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1.8);

	imshow("lines", gray_dst); //显示霍夫变换检测后框选的物体轮廓图
	imshow("laser", LaserMat); //显示可视化的激光雷达点
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

	for (int id = 4; id < laserPoint.size() - 4; id++)
	{

		if (isEdgePoint(laserPoint[id - 4].second, laserPoint[id].second, laserPoint[id + 4].second, 5.5))
		{
			laserPoint[id].first.second = true;
		}
	}

	vector<int> conPoints; //储存连续的特征点
	vector<int> edgePoints;

	for (int id = 5; id < laserPoint.size() - 5; id++)
	{
		//去除单个边缘点 单个边缘点基本是平面点
		if (laserPoint[id].first.second && (!laserPoint[id + 1].first.second) && (!laserPoint[id - 1].first.second))
		{
			laserPoint[id].first.second = false;
		}

		//对连续的边缘点取中间的作为最终的边缘点
		if (!laserPoint[id].first.second)
		{
			if (conPoints.size() > 3)
			{
				//ROS_INFO_STREAM("size is "<<conPoints.size());
				edgePoints.push_back(conPoints[conPoints.size() / 2]);
			}
			vector<int>().swap(conPoints);
		}

		//储存连续的边缘点
		if (laserPoint[id].first.second && ((laserPoint[id + 1].first.second) || (laserPoint[id - 1].first.second)))
		{
			conPoints.push_back(id);
		}
	}

	for (int id = 0; id < laserPoint.size(); id++)
	{
		laserPoint[id].first.second = false;
	}

	for (int id : edgePoints)
	{
		laserPoint[id].first.second = true;
	}
}

//判断哪些激光点位于识别到的物体上
//并判断激光线在照片中的倾斜角度
void pointInRoi(Rect roi, vector<laser_coor> &allPoint, vector<laser_coor> &inPoint)
{

	int Xmin = roi.tl().x * 1.1, Ymin = roi.tl().y * 1.1;
	int Xmax = roi.br().x * 0.95, Ymax = roi.br().y * 0.95;

	for (int i = 0; i < allPoint.size(); ++i)
	{
		if (allPoint[i].first.first.x >= Xmin && allPoint[i].first.first.x <= Xmax)
		{
			inPoint.push_back(allPoint[i]);
		}
	}
}

//计算没有边缘点的激光雷达数据序列
vector<laser_coor> getIndex(vector<laser_coor> &Point)
{

	//寻找最长的无边缘点的直线来计算角度
	int maxP = 0;			//该段直线的激光点数
	int begin = 0, end = 0; //直线的起点和终点下标
	auto it = find_if(Point.begin(), Point.end(), [](laser_coor x) { return x.first.second; });
	auto lastIt = Point.begin();
	do
	{

		int be = lastIt - Point.begin();
		int en = it - Point.begin();
		if ((en - be) > maxP)
		{
			begin = be;
			end = en;
		}
		maxP = end - begin;
		if (it == Point.end())
			break;
		lastIt = it;
		it = find_if(it + 1, Point.end(), [](laser_coor x) { return x.first.second; });

	} while (true);

	if (maxP > 8)
		return vector<laser_coor>(Point.begin() + begin, Point.begin() + end);
	else
		return vector<laser_coor>(0);
}

//计算激光雷达线在图片中的角度
float getHorizonAngle(vector<laser_coor> &result)
{

	if (result.size() > 2)
	{
		vector<cv::Point> linePoints;
		for (int i = 0; i < result.size(); i++)
		{
			linePoints.push_back(result[i].first.first);
		}
		cv::Vec4f linePara;
		cv::fitLine(linePoints, linePara, cv::DIST_L2, 0, 1e-2, 1e-2);
		return atan(linePara[1] / linePara[0]) * 180.0 / pi;
	}
	else
		return -1;
}

//接收到传感器数据后的回调函数
void combineCallback(const sensor_msgs::ImageConstPtr &rgb_image_qhd, const sensor_msgs::LaserScanConstPtr &laser_data)
{
	//ROS_INFO("----------------------------");
	//ROS_INFO("得到一帧同步数据, 开始处理......");
	//clock_t time_old = clock();

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
	}
	catch (cv_bridge::Exception &e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		ROS_INFO("结束该帧数据处理, 等待下帧数据.....");
		return;
	}

	imshow("original", rgb_ptr->image);

	//运行深度学习检测图片中的物体
	Mat delframe;
	resize(rgb_ptr->image, delframe, Size(300, 300));
	Mat inputBlob = blobFromImage(delframe, 0.007843, Size(300, 300), 127.5, false, false);
	net.setInput(inputBlob, "data");
	Mat detection = net.forward("detection_out");
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	//记录新帧中有哪些类别的识别
	std::vector<int> detection_record_new;
	//记录上面的识别对应detectionMat中的哪一行
	std::vector<int> detection_record_i;

	//新的一帧对应的识别，
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2); //置信度
		if (confidence > -1)
		{

			int labelidx = detectionMat.at<float>(i, 1); //识别物体类别
			if (labelidx == 6 || labelidx == 7 || labelidx == 8)
			{
				detection_record_new.push_back(labelidx); //图片中的框索引
				detection_record_i.push_back(i);
			}
		}
	}

	//对新物体进行添加，
	std::vector<int>::iterator new_detection_iterator = detection_record_new.begin();
	int detection_i = 0;

	for (; new_detection_iterator != detection_record_new.end(); new_detection_iterator++, detection_i++)
	{

		int xLeftTop = static_cast<int>(detectionMat.at<float>(detection_record_i[detection_i], 3) * width);
		int yLeftTop = static_cast<int>(detectionMat.at<float>(detection_record_i[detection_i], 4) * height);
		int xRightBottom = static_cast<int>(detectionMat.at<float>(detection_record_i[detection_i], 5) * width);
		int yRightBottom = static_cast<int>(detectionMat.at<float>(detection_record_i[detection_i], 6) * height);

		//抑制边界
		if (xLeftTop < 0)
			xLeftTop = 0;
		if (yLeftTop < 0)
			yLeftTop = 0;
		if (xRightBottom > width)
			xRightBottom = width - 1;
		if (yRightBottom > height)
			yRightBottom = height - 1;

		int x = xLeftTop * 0.95 * 2;
		int y = yLeftTop * 0.95 * 2;
		int w = (xRightBottom - xLeftTop) * 1.15 * 2;
		int h = (yRightBottom - yLeftTop) * 1.15 * 2;

		if ((x + w) > rgb_ptr->image.cols)
			w = rgb_ptr->image.cols - x;
		if ((y + h) > rgb_ptr->image.rows)
			h = rgb_ptr->image.rows - y;

		Rect object_rect(x, y, w, h);

		//ROS_INFO("运行grabcut函数......");
		//抠图 去除背景干扰
		Mat cut, bg, fg;
		grabCut(rgb_ptr->image, cut, object_rect, bg, fg, 4, GC_INIT_WITH_RECT);
		compare(cut, GC_PR_FGD, cut, CMP_EQ);
		Mat foreGround(rgb_ptr->image.size(), CV_8UC3, Scalar(255, 255, 255));
		rgb_ptr->image.copyTo(foreGround, cut);
		//ROS_INFO("grabcut函数完成");

		vector<laser_coor> inPoint;
		pointInRoi(object_rect, laserPoint, inPoint);
		//auto line = houghlinedetect(foreGround);
		measurement(foreGround, inPoint, *new_detection_iterator, x, w);

		//imshow("grabcut", foreGround);

		waitKey(1000);
	}

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