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
Point getCrossPoint(vector<int> LineA, vector<int> LineB);
vector<int> getLineParam(Point start, Point end);
float lines_orientation(Point begin, Point end, int flag);
void laser_to_rgb(const sensor_msgs::LaserScanConstPtr &scan, vector<laser_coor> &laserPoint);
bool pointInRoi(Rect roi, vector<laser_coor> &allPoint, vector<laser_coor> &inPoint);
void combineCallback(const sensor_msgs::ImageConstPtr &rgb_image_qhd, const sensor_msgs::LaserScanConstPtr &laser_data);
void computerPiexDistance(vector<laser_coor> &Point, float result[2]);
void measurement(Mat &roiImg, vector<laser_coor> &laserPoint, int label, int x, int w);
void getEdge(Mat &roiImg, vector<vector<Point>>& H_Line, vector<vector<Point>>& V_Line);
void CannyAndResult(Mat &roiImg, vector<vector<Point>>& H_Line, vector<vector<Point>>& V_Line);

string modelConfiguration = "/home/wode/configuration_folder/trash_ssd/newindoor_bob/deploy.prototxt";
string modelBinary = "/home/wode/configuration_folder/trash_ssd/newindoor_bob/_iter_90109.caffemodel";
//string *class_array = new string[9]{"background", "window", "bed", "aricondition", "sofa", "chair", "cabinet", "trash", "door"};
string *class_array = new string[12]{"background", "bed", "cabinet", "chair", "table", "sofa", "closestool", "door", "refrigerator", "washer", "corner", "trash"};

Net net = readNetFromCaffe(modelConfiguration, modelBinary);

double cx_ = 239.9 * 2;
double cy_ = 131.975 * 2;
double fx_ = 268.225 * 2;
double fy_ = 268.575 * 2;

//double laser2robot_x = 0.16;
//double laser2robot_y = 0.0;

double kinect2robot_x = 0.132;
double kinect2robot_y = -0.08;

double laser2kinect_x = 0.01;
double laser2kinect_y = -0.095;
double laser2kinect_z = 0.78;

double Y2=0,Y3=0;
double L2=0,L3=0;

void getCrossLine(Mat &im, Rect re)
{
	Mat result;
	cvtColor(im, result, CV_BGR2GRAY);
	threshold(result, result,50, 255, CV_THRESH_BINARY);
	vector<Point> crossPoint;
	int half = (re.tl().x + re.br().x)/ 2;
	int pre = 0, current = 0;
	int PointRows = 0;
	bool isSecond = false;
	for (int i = im.rows - 1; i >= 0; i--)
	{
		for (int j = 0; j < half; j++)
		{
			if (result.at<uchar>(i, j) < 100)
			{
				current++;
				PointRows = j;
			}
		}

		if ((current - pre) > 4)
		{
			circle(im, Point(PointRows, i), 2, Scalar(0, 0, 255));
			crossPoint.push_back(Point(PointRows, i));
			if (isSecond)
				break;
			isSecond = true;
		}
		pre = current;
		current = 0;
	}

	pre = 0, current = 0;
	PointRows = 0;
	isSecond = false;
	for (int i = im.rows - 1; i >= 0; i--)
	{
		for (int j = im.cols - 1; j > half; j--)
		{
			if (result.at<uchar>(i, j) < 100)
			{
				current++;
				PointRows = j;
			}
		}

		if ((current - pre) > 4)
		{
			circle(im, Point(PointRows, i), 2, Scalar(0, 0, 255));
			crossPoint.push_back(Point(PointRows, i));
			if (isSecond)
				break;
			isSecond = true;
		}
		pre = current;
		current = 0;
	}

	line(im, crossPoint[0], crossPoint[1], Scalar(0, 0, 255), 1);
	line(im, crossPoint[2], crossPoint[3], Scalar(0, 0, 255), 1);

	Y2=(crossPoint[1].y+crossPoint[3].y)/2;
	Y3=(crossPoint[0].y+crossPoint[2].y)/2;

	L2=sqrt(pow((crossPoint[1].x-crossPoint[3].x), 2)+pow((crossPoint[1].y-crossPoint[3].y), 2));
	L3=sqrt(pow((crossPoint[0].x-crossPoint[2].x), 2)+pow((crossPoint[0].y-crossPoint[2].y), 2));
    //ROS_INFO_STREAM("L2 is "<<L2);
	//ROS_INFO_STREAM("L3 is "<<L3);
	imshow("orignal", im);

}

//计算每个像素点的实际距离(cm)
void computerPiexDistance(vector<laser_coor> &Point, float result[2])
{
	// //float result[2];
	double dis = 0;
	double all_piex = 0;
	int y = 0;

	//截取中间激光点进行单位像素距离计算
	for (int i = 2; i < Point.size(); i++)
	{
		y+=Point[i].first.first.y;
		all_piex += sqrt(pow((Point[i - 2].first.first.x - Point[i].first.first.x), 2) + pow((Point[i - 2].first.first.y - Point[i].first.first.y), 2));
		dis += sqrt(pow((Point[i - 2].second.x - Point[i].second.x), 2) + pow((Point[i - 2].second.y - Point[i].second.y), 2));
	}
	result[0] = dis / all_piex * 100;
	result[1] = y/(Point.size()-2);//返回激光点Y坐标
	//ROS_INFO_STREAM("Y1 is "<<result[1]);
}

//相机坐标系转像素坐标系
Point2i camera_2_rgb(ppoint point_camera)
{
	Point2i point_;
	point_.x = (fx_ * point_camera.x / point_camera.z + cx_);
	point_.y = (fy_ * point_camera.y / point_camera.z + cy_);
	return point_;
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

//识别物体轮廓并画框显示
//利用霍夫变换检测直线然后选择最外侧的直线作为轮廓线
void CannyAndResult(Mat &roiImg, vector<vector<Point>>& H_Line, vector<vector<Point>>& V_Line)
{

	int threshold_value = 30;

	Mat dst;
	//使用边缘检测将图片二值化
	Canny(roiImg, dst, 10, 50, 3, false);

	vector<Vec4i> lines;								   //存储直线数据
	HoughLinesP(dst, lines, 1, CV_PI / 180.0, 30, 30, 10); //源图需要是二值图像，HoughLines也是一样
    
	Mat cannyShow = dst(show);
	getEdge(dst, H_Line, V_Line);
	imshow("canny", cannyShow);

}

void getEdge(Mat &roiImg, vector<vector<Point>>& H_Line, vector<vector<Point>>& V_Line)
{
	//查找最上面轮廓线
	int i = 0;
    for(; i<roiImg.rows; i++)
	{
		int count = 0;
		int j=0;
		for(; j<roiImg.cols; j++)
		{
			if(roiImg.at<uchar>(i, j) == 255)
			{
				count++;
			}
		}
		if(count>5)
		{
			vector<Point> tmp;
			Point begin(j/2-2, i), end(j/2+2, i);
			tmp.push_back(begin);
			tmp.push_back(end);
			tmp.push_back(Point((begin.x + end.x) / 2, (begin.y + end.y) / 2));
			H_Line.push_back(tmp);
			break;
		}
	}

	//查找最下面轮廓线
	i = roiImg.rows-1;
    for( ; i>=0; i--)
	{
		int count = 0;
		int j=0;
		for(; j<roiImg.cols; j++)
		{
			if(roiImg.at<uchar>(i, j) == 255)
			{
				count++;
			}
		}
		if(count>5)
		{
			vector<Point> tmp;
			Point begin(j/2-2, i), end(j/2+2, i);
			tmp.push_back(begin);
			tmp.push_back(end);
			tmp.push_back(Point((begin.x + end.x) / 2, (begin.y + end.y) / 2));
			H_Line.push_back(tmp);
			break;
		}
	}

	//查找最右面轮廓线
	i = roiImg.cols-1;
    for( ; i>=0; i--)
	{
		int count = 0;
		int j=0;
		for(; j<roiImg.rows; j++)
		{
			if(roiImg.at<uchar>(j, i) == 255)
			{
				count++;
			}
		}
		if(count>40)
		{
			vector<Point> tmp;
			Point begin(i, j/2-2), end(i, j/2+2);
			tmp.push_back(begin);
			tmp.push_back(end);
			tmp.push_back(Point((begin.x + end.x) / 2, (begin.y + end.y) / 2));
			V_Line.push_back(tmp);
			break;
		}
	}

	//查找最左面轮廓线
	i = 0;
    for( ; i<roiImg.cols; i++)
	{
		int count = 0;
		int j=0;
		for(; j<roiImg.rows; j++)
		{
			if(roiImg.at<uchar>(j, i) == 255)
			{
				count++;
			}
		}
		if(count>30)
		{
			vector<Point> tmp;
			Point begin(i, j/2-2), end(i, j/2+2);
			tmp.push_back(begin);
			tmp.push_back(end);
			tmp.push_back(Point((begin.x + end.x) / 2, (begin.y + end.y) / 2));
			V_Line.push_back(tmp);
			break;
		}
	}
}

void measurement(Mat &roiImg, vector<laser_coor> &laserPoint, int label, int x, int w)
{

	int rangeXMIN, rangeXMAX;
	auto di = laserPoint;
	if (di.size() == 0)
	{
		ROS_INFO_STREAM("-------------------------------");
		ROS_INFO_STREAM("连续平面激光点数过少");
		ROS_INFO_STREAM("无法进行有效测量,结束该帧测量");
		ROS_INFO_STREAM("-------------------------------\n");
		return;
	}

	float dis[2];
	computerPiexDistance(di, dis);

	vector<vector<Point>> H_Line, V_Line; //储存水平线与竖直线 储存每条线的起点 中点和终点

	
	ROS_INFO_STREAM("-------------------------------");
	ROS_INFO_STREAM("检测到" << class_array[label] << ",开始测量......");
    
	CannyAndResult(roiImg, H_Line, V_Line);

	int top =0, left = 1, right = 0, bottom =1;
	//将边缘线延长
	vector<int> paramA = getLineParam(H_Line[top][0], H_Line[top][1]);
	vector<int> paramB = getLineParam(V_Line[left][0], V_Line[left][1]);
	vector<int> paramC = getLineParam(H_Line[bottom][0], H_Line[bottom][1]);
	vector<int> paramD = getLineParam(V_Line[right][0], V_Line[right][1]);

	Mat gray_dst = roiImg.clone();

	//绘制轮廓
	auto crossPointTL = getCrossPoint(paramA, paramB);
	auto crossPointTR = getCrossPoint(paramA, paramD);
	auto crossPointBL = getCrossPoint(paramC, paramB);
	auto crossPointBR = getCrossPoint(paramC, paramD);
	line(gray_dst, crossPointTL, crossPointTR, Scalar(0, 0, 255), 1, LINE_AA);
	line(gray_dst, crossPointTL, crossPointBL, Scalar(0, 0, 255), 1, LINE_AA);
	line(gray_dst, crossPointTR, crossPointBR, Scalar(0, 0, 255), 1, LINE_AA);
	line(gray_dst, crossPointBR, crossPointBL, Scalar(0, 0, 255), 1, LINE_AA);

	int Y1 = dis[1];

	double a = (Y3-Y1)/(Y2-Y1);
	double L1 = (a*L2-L3)/(a-1);

	// ROS_INFO_STREAM("Y1 is "<<Y1);
	// ROS_INFO_STREAM("Y2 is "<<Y2);
	// ROS_INFO_STREAM("Y3 is "<<Y3);
	// ROS_INFO_STREAM("a is "<<a);
    // ROS_INFO_STREAM("L1 is "<<L1);

	float hi = (Y2-crossPointTL.y) * dis[0]* L1/L2 * 1.2;
	float wh = L1 * dis[0] * 1.4;
    
	ROS_INFO_STREAM("higet is " << hi << "cm, width is " << wh << "cm");
	ROS_INFO_STREAM("-------------------------------\n");

	char tx[20];
	sprintf(tx, "%.2f", hi);
	putText(gray_dst, tx, (crossPointTL + crossPointBL) / 2, FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1.8);
	memset(tx, 0, 20);
	sprintf(tx, "%.2f", wh);
	putText(gray_dst, tx, (crossPointBR + crossPointBL) / 2, FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1.8);
    
	Mat lunkuoMat = gray_dst(show);

	imshow("lines", lunkuoMat); //显示霍夫变换检测后框选的物体轮廓图

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

//判断哪些激光点位于识别到的物体上
//并判断激光线在照片中的倾斜角度
bool pointInRoi(Rect roi, vector<laser_coor> &allPoint, vector<laser_coor> &inPoint)
{
	int Xmin = roi.tl().x , Ymin = roi.tl().y;
	int Xmax = roi.br().x , Ymax = roi.br().y;
	for (int i = 0; i < allPoint.size(); ++i)
	{
		if (allPoint[i].first.first.x >= Xmin && allPoint[i].first.first.x <= Xmax)
		{
			inPoint.push_back(allPoint[i]);
		}
	}

	double maxDis = 0.1;
	int begin, end;
	for(int i = 1; i<inPoint.size(); i++)
	{
		auto dis = pow((inPoint[i].second.x - inPoint[i-1].second.x), 2) + pow((inPoint[i].second.y - inPoint[i-1].second.y), 2);
		//cout<<dis<<endl;
        if( dis > (maxDis * maxDis) )
		{
			begin = i;
			//ROS_INFO_STREAM("begin is "<<begin);
			break;
		}
	}
	//cout<<endl<<endl;
	for(int i = inPoint.size() - 1; i>0; i--)
	{
		auto dis = pow((inPoint[i].second.x - inPoint[i-1].second.x), 2) + pow((inPoint[i].second.y - inPoint[i-1].second.y), 2);
		//cout<<dis<<endl;
        if( dis > maxDis * maxDis )
		{
			end = i;
			//ROS_INFO_STREAM("end is "<<end);
			break;
		}
	}

	if(end == begin)
	{
		end = inPoint.size()-1;
	}

	vector<laser_coor> tmp;
	for(int i = inPoint.size()*0.25+2; i<end*0.85-2; i++)
	{
		tmp.push_back(inPoint[i]);
	}
	inPoint.swap(tmp);

    if(inPoint.size()<6)
	{
		ROS_INFO_STREAM("point size is "<<inPoint.size());
		ROS_INFO_STREAM("有效激光点太少 无法测量");
		return true;
	}
	for(int i = 0; i<3; i++)
    {
		//中值滤波
		double x, y;
		for(int i =2; i<inPoint.size()-2;i++)
		{
			x = 0;
			y = 0;
			x += inPoint[i-2].first.first.x + inPoint[i-1].first.first.x + inPoint[i].first.first.x + inPoint[i+1].first.first.x + inPoint[i+2].first.first.x;
			y += inPoint[i-2].first.first.y + inPoint[i-1].first.first.y + inPoint[i].first.first.y + inPoint[i+1].first.first.y + inPoint[i+2].first.first.y;
			inPoint[i].first.first.x = x/5;
			inPoint[i].first.first.y = y/5;

			x = 0;
			y = 0;
			x += inPoint[i-2].second.x + inPoint[i-1].second.x + inPoint[i].second.x + inPoint[i+1].second.x + inPoint[i+2].second.x;
			y += inPoint[i-2].second.y + inPoint[i-1].second.y + inPoint[i].second.y + inPoint[i+1].second.y + inPoint[i+2].second.y;
			inPoint[i].second.x = x/5;
			inPoint[i].second.y = y/5;
		}
	}
	return true;
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

	//运行深度学习检测图片中的物体
	Mat delframe;
	resize(rgb_ptr->image, delframe, Size(300, 300));
	Mat inputBlob = blobFromImage(delframe, 1, Size(300, 300), 127.5, false, false);
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
		//ROS_INFO_STREAM("confidence is "<<confidence);
		if (confidence > 0.5)
		{

			int labelidx = detectionMat.at<float>(i, 1); //识别物体类别
			if (labelidx==3)
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

		int x = xLeftTop * 2+10;
		int y = yLeftTop * 2-22;
		int w = (xRightBottom - xLeftTop) * 2;
		int h = (yRightBottom - yLeftTop) * 2 + 32;

		if ((x + w) > rgb_ptr->image.cols)
			w = rgb_ptr->image.cols - x;
		if ((y + h) > rgb_ptr->image.rows)
			h = rgb_ptr->image.rows - y;

		Rect object_rect(x, y, w, h);
		show = object_rect;
		// ROS_INFO_STREAM("rect x is "<<x);
		// ROS_INFO_STREAM("rest y is "<<y);
		// ROS_INFO_STREAM("rect w is "<<w);
		// ROS_INFO_STREAM("rest h is "<<h);

		//ROS_INFO("运行grabcut函数......");
		//抠图 去除背景干扰
		Mat cut, bg, fg;
		grabCut(rgb_ptr->image, cut, object_rect, bg, fg, 4, GC_INIT_WITH_RECT);
		compare(cut, GC_PR_FGD, cut, CMP_EQ);
		Mat foreGround(rgb_ptr->image.size(), CV_8UC3, Scalar(255, 255, 255));
		rgb_ptr->image.copyTo(foreGround, cut);
		imshow("grab", foreGround);
		//ROS_INFO("grabcut函数完成");
		Mat rectMat = rgb_ptr->image.clone();
		rectangle(rectMat, object_rect, Scalar(0, 0, 255));
		imshow("original", rgb_ptr->image);
		imshow("rect", rectMat);

        Mat inrangeMat;
		inRange(foreGround, Scalar(0,0,0),Scalar(100,100,100), inrangeMat);//再次分割
        imshow("123", inrangeMat);
        Mat changeMat(rectMat.rows, rectMat.cols, CV_8UC3, Scalar(255,255,255));
		//cvtColor(foreground, inrangeMat, CV_BGR2GRAY);
		//threshold(inrangeMat, inrangeMat, 50, 255, CV_THRESH_BINARY);
        foreGround.copyTo(changeMat, inrangeMat);
        imshow("change", changeMat);
		getCrossLine(changeMat, object_rect);

		vector<laser_coor> inPoint;
		Mat LaserMat =  rgb_ptr->image.clone();//储存显示雷达激光点的图像
		try
		{
			if(pointInRoi(object_rect, laserPoint, inPoint))
			{
                //ROS_INFO_STREAM(" size is "<<inPoint.size());
		        measurement(foreGround, inPoint, *new_detection_iterator, x, w);
			}
			
			for (int i = 0; i < inPoint.size(); i++)
			{
				
				circle(LaserMat, inPoint[i].first.first, 1, Scalar(0, 255, 0), 2, 2); //红色显示边缘点
				
			}

			imshow("laser", LaserMat(show));
		}
		catch(...)
		{
			waitKey(1000);
			continue;
		}

		waitKey(1000);
	}
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