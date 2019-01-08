#include<robot/robot.h>
#include<opencv2/opencv.hpp>
#include<librealsense2/rs.hpp>
#include<librealsense2/rsutil.h>
#include<iostream>
#include<stdio.h>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>
#include<opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
//const double u0=321.6100791100012;
//const double v0=246.1913397344533;
//const double fx=474.1516982222988;
//const double fy=473.8105969419406;
//const int depthScale=1000;
//const double height_jiazhua = 93;//夹爪的高度,单位为mm
const double height_jiazhua = 105;
int main(int argc, char * argv[])
{
    //测试opencv时用的代码
    //cv::Mat image(480,640,CV_8UC3);
    //image=cv::imread("/home/yaohui/catkin_ws/src/opencv_ros_train/1_Depth.png");
    //cv::imshow("image",image);
    //cv::waitKey(1);
  Robot r("192.168.2.101:50051");
  r.logout();
  r.login();
  //r.start();
  //首先到达初始拍照位置
  std::vector<float> initial_joints(6);
  initial_joints.at(0) = -2.59;
  initial_joints.at(1) = 122.17;
  initial_joints.at(2) = 91.64;
  initial_joints.at(3) = 55.58;
  initial_joints.at(4) = -90.12;
  initial_joints.at(5) = 87.79;

  //顺序依次是x,y,z,Rz,Ry,Rx
  std::vector<float> initial_pose(6);
  initial_pose.at(0) = -538.13;
  initial_pose.at(1) = 140.86;
  initial_pose.at(2) = 152.27;
  initial_pose.at(3) = -0.37;
  initial_pose.at(4) = 0.6;
  initial_pose.at(5) = -179.86;

  std::vector<float> final_joints1(6);
  final_joints1.at(0) = -92.59;
  final_joints1.at(1) = 122.17;
  final_joints1.at(2) = 91.64;
  final_joints1.at(3) = 55.58;
  final_joints1.at(4) = -90.12;
  final_joints1.at(5) = 87.79;

  std::vector<float> final_joints2(6);
  final_joints2.at(0) = 180-92.59;
  final_joints2.at(1) = 122.17;
  final_joints2.at(2) = 91.64;
  final_joints2.at(3) = 55.58;
  final_joints2.at(4) = -90.12;
  final_joints2.at(5) = 87.79;
  //initial_pose2matrix4d function
  Eigen::Matrix4d ret_hand_base = Eigen::MatrixXd::Identity(4,4);

  ret_hand_base.topRightCorner(3,1) << static_cast<double>(initial_pose[0]/1000),
                                 static_cast<double>(initial_pose[1]/1000),
                                 static_cast<double>(initial_pose[2]/1000);
  Eigen::Quaterniond m = Eigen::AngleAxisd(static_cast<double>(initial_pose[3])/180*M_PI, Eigen::Vector3d::UnitZ())
              * Eigen::AngleAxisd(static_cast<double>(initial_pose[4])/180*M_PI, Eigen::Vector3d::UnitY())
              * Eigen::AngleAxisd(static_cast<double>(initial_pose[5])/180*M_PI, Eigen::Vector3d::UnitX());
  ret_hand_base.topLeftCorner(3,3) << m.matrix();

  Eigen::Matrix4d ret_eye_hand;
  //the result of handeye_calibration
  ret_eye_hand << 0.128261,0.991425,-0.0250192,-0.0494969,
                  -0.991241,0.127355,-0.0349627,-0.0328113,
                  -0.0314766,0.0292844,0.999075,0.0232873,
                  0,0,0,1;
  Eigen::Matrix4d ret_eye_base = ret_hand_base * ret_eye_hand;
  //cout << ret_eye_base << endl;
  //cout << ret_hand_base;
  r.jointMove(initial_joints, 25);

  rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
  rs2::config cfg;
  ///配置深度图像流：分辨率640*480，图像格式：Z16， 帧率：30帧/秒
  cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
  //彩色图
  cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
  ///生成Realsense管道，用来封装实际的相机设备
  rs2::pipeline pipe;
  ///根据给定的配置启动相机管道
  pipe.start(cfg);
  rs2::frameset data;
  ///将深度图像转换为Opencv格式
  cv::Mat depthmat;
  cv::Mat image;
  RotatedRect rect;
  rs2::stream_profile profile_depth;
  rs2::stream_profile profile_color;
  int num=0;
  Point2f zhongxin={0};
  Point2f P[4];
  Point2f use_P[4];
  Point2f rgb_P[4];
  float pointzhongxin[3];
  //int red_num;
  bool goal;//根据信封正反进行不同移动
  Mat imageContours;
  while(1)
  {
    bool find_success = false;
    ///等待一帧数据，默认等待5s
    data = pipe.wait_for_frames();
    rs2::frame depth  = data.get_depth_frame(); ///获取深度图像数据
    rs2::frame color = data.get_color_frame();
    profile_depth = depth.get_profile();
    profile_color = color.get_profile();
    if (!depth) break;            ///如果获取不到数据则退出
    depthmat = cv::Mat(cv::Size(640, 480), CV_16U, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
    image=cv::Mat(cv::Size(640, 480), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);
    ///显示
    //deal with the image,get the information about the square of SFmail//
    Mat grad_x, abs_grad_x, grad_y, abs_grad_y, dst, Canny_image;
    Mat Dilate_image;
    namedWindow("depth",CV_WINDOW_AUTOSIZE);
    namedWindow("MinAreaRect",CV_WINDOW_AUTOSIZE);
    namedWindow("RGB",CV_WINDOW_AUTOSIZE);
    namedWindow("Shift",CV_WINDOW_AUTOSIZE);
    imshow("depth", depthmat);
    //求X方向梯度
    Sobel( depthmat, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    //求Y方向梯度
    Sobel( depthmat, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    //合并梯度
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );
    //canny find line
    Canny(dst, Canny_image, 30, 30*3, 3);
    //膨胀操作, 填充边缘缝隙
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(Canny_image, Dilate_image, element);
    for (int i = 0;i < 10;i++)
    {
        //erode(pDilateImage, pDilateImage, element, Point(-1,-1));
        dilate(Dilate_image, Dilate_image, element, Point(-1,-1));
    }
    for (int j = 0;j < 8;j++)
    {
        erode(Dilate_image, Dilate_image, element, Point(-1,-1));
        //dilate(pDilateImage, pDilateImage, element, Point(-1,-1));
    }
    //寻找最外层轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(Dilate_image,contours,hierarchy,RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    imageContours=Mat::zeros(Dilate_image.size(),CV_8UC1);	//最小外接矩形画布
    double  radio;
    for(int i=0;i<contours.size();i++)
    {
            //绘制轮廓
            drawContours(imageContours,contours,i,Scalar(255),1,8,hierarchy);

            //绘制轮廓的最小外结矩形
            rect=minAreaRect(contours[i]);
            Point2f P[4];
            rect.points(P);
            radio = rect.size.height>=rect.size.width ? rect.size.height/rect.size.width:rect.size.width/rect.size.height;
            cout<<"juxingdaxiao"<<rect.size.area()<<endl;
            if(rect.size.area()>=50000 && rect.size.area()<120000 && radio>=1 && radio<=1.8)
            {   find_success = true;
                for(int j=0;j<=3;j++)
                {
                    use_P[j]=P[j];
                    line(imageContours,P[j],P[(j+1)%4],Scalar(255),2);
                }
             zhongxin=rect.center;
             //cout<<"rectangle size:"<<rect.size.area()<<endl;
            }
    }

    if(!find_success)
        continue;
    else {
        if(zhongxin.x!=0)break;
    }

  }
  cout<<"rectangle mass center(x,y):"<<zhongxin<<endl;
  num++;
  imshow("MinAreaRect", imageContours);
  rs2::video_stream_profile dvsprofile(profile_depth);
  rs2::video_stream_profile cvsprofile(profile_color);
  rs2_intrinsics depth_intrin =  dvsprofile.get_intrinsics();
  rs2_intrinsics color_intrin =  cvsprofile.get_intrinsics();
  rs2_extrinsics depth2color_extrin=profile_depth.get_extrinsics_to(profile_color);

  float pixelzhongxin[2];
  pixelzhongxin[0]=int(zhongxin.x);
  pixelzhongxin[1]=int(zhongxin.y);
  unsigned int d = depthmat.ptr<unsigned short> (int(zhongxin.x))[int(zhongxin.y)];
  const float depthscale=8000.0;//sr300相机的depthscale
  float depth_dis=d/depthscale;
  rs2_deproject_pixel_to_point(pointzhongxin,&depth_intrin,pixelzhongxin,depth_dis);
  cout<<"图片中心在相机坐标系下的x"<<pointzhongxin[0]<<endl;
  cout<<"图片中心在相机坐标系下的y"<<pointzhongxin[1]<<endl;
  cout<<"图片中心在相机坐标系下的z"<<pointzhongxin[2]<<endl;

  float point[3];
  float pixel[2];

  auto middle = (use_P[0]+use_P[2])/2;
  bool depthinvalid=true;
  float step = 0.05;
  unsigned int dd;

  while(depthinvalid && step < 0.8)
  {
      depthinvalid = true;
      step += 0.01;
      for(int i=0; i<4; i++)
      {
          Point2f temp_P=use_P[i]+(middle-use_P[i])*step;
          if (0==depthmat.ptr<unsigned short> (int(temp_P.x))[int(temp_P.y)])
              continue;
          else
          {
              dd = depthmat.ptr<unsigned short> (int(temp_P.x))[int(temp_P.y)];
              depthinvalid = false;
              break;
          }
      }
  }
  if (depthinvalid)
        {
           return -1;
        }

  for (int n=0;n<4;n++) {
      pixel[0]=int(use_P[n].x);
      pixel[1]=int(use_P[n].y);
      const float depthscale=8000.0;//sr300相机的depthscale
      float ddepth_dis=dd/depthscale;
      rs2_deproject_pixel_to_point(point,&depth_intrin,pixel,ddepth_dis);
      float rgb_point[3];
      rs2_transform_point_to_point(rgb_point,&depth2color_extrin,point);
      float rgb_pixel[2];
      rs2_project_point_to_pixel(rgb_pixel,&color_intrin,rgb_point);
      rgb_P[n].x=rgb_pixel[0];
      rgb_P[n].y=rgb_pixel[1];
  }
  for(int j=0;j<=3;j++)
  {
      line(image,rgb_P[j],rgb_P[(j+1)%4],Scalar(255),2);
  }
  imshow("RGB",image);
  double width ,height;
  double max_x = 0, min_x = 640, max_y = 0, min_y = 480;
  Mat Shift_img;
  for(int j=0;j<=3;j++)
  {
      if(rgb_P[j].y >= max_y )
      {
          if(rgb_P[j].y>=480)
              max_y = 480;
          else
              max_y = rgb_P[j].y;
      }

      if(rgb_P[j].x >= max_x)
      {
          if(rgb_P[j].x>=640)
              max_x = 640;
          else
              max_x = rgb_P[j].x;
      }

      if(rgb_P[j].x < min_x)
      {
          if(rgb_P[j].x<=0)
              min_x = 0;
          else
              min_x = rgb_P[j].x;
      }

      if(rgb_P[j].y < min_y)
      {
          if(rgb_P[j].y<=0)
              min_y = 0;
          else
              min_y = rgb_P[j].y;
      }
  }
//  double angle = 0;

//  if((tmp_point_maxy.y-tmp_point_minx.y)*(tmp_point_maxy.y-tmp_point_minx.y)+(tmp_point_maxy.x-tmp_point_minx.x)*(tmp_point_maxy.x-tmp_point_minx.x)
//          > (tmp_point_maxy.y-tmp_point_maxx.y)*(tmp_point_maxy.y-tmp_point_maxx.y)+(tmp_point_maxy.x-tmp_point_maxx.x)*(tmp_point_maxy.x-tmp_point_maxx.x))
//  {
//      angle = atan2(tmp_point_maxy.y-tmp_point_minx.y, tmp_point_maxy.x-tmp_point_minx.x)*180/3.1415;
//      height = sqrt((tmp_point_maxy.y-tmp_point_minx.y)*(tmp_point_maxy.y-tmp_point_minx.y)+(tmp_point_maxy.x-tmp_point_minx.x)*(tmp_point_maxy.x-tmp_point_minx.x));
//      width =  sqrt((tmp_point_maxy.y-tmp_point_maxx.y)*(tmp_point_maxy.y-tmp_point_maxx.y)+(tmp_point_maxy.x-tmp_point_maxx.x)*(tmp_point_maxy.x-tmp_point_maxx.x));
//  }
//  else {
//      angle = atan2(tmp_point_maxy.y-tmp_point_maxx.y, tmp_point_maxy.x-tmp_point_maxx.x)*180/3.1415;
//      width  = sqrt((tmp_point_maxy.y-tmp_point_minx.y)*(tmp_point_maxy.y-tmp_point_minx.y)+(tmp_point_maxy.x-tmp_point_minx.x)*(tmp_point_maxy.x-tmp_point_minx.x));
//      height = sqrt((tmp_point_maxy.y-tmp_point_maxx.y)*(tmp_point_maxy.y-tmp_point_maxx.y)+(tmp_point_maxy.x-tmp_point_maxx.x)*(tmp_point_maxy.x-tmp_point_maxx.x));
//  }

//  cout<<"Maxy"<<tmp_point_maxy;
//  cout<<"Maxx"<<tmp_point_maxx;
//  cout<<"Minx"<<tmp_point_minx;
//  cout<<"angle:"<<angle<<endl;
//  cout<<"height:"<<height<<endl;
//  cout<<"width:"<<width<<endl;

//  Mat big_image=Mat::zeros(image.size().height*2,image.size().width*2,CV_8UC3);

//  for (int rows = 0; rows<image.rows; rows++)
//     for(int cols = 0; cols<image.cols; cols++)
//        {
//             big_image.at<Vec3b>(rows+image.rows-center_rgb.y, cols+image.cols-center_rgb.x)[0] = image.at<Vec3b>(rows, cols)[0];
//             big_image.at<Vec3b>(rows+image.rows-center_rgb.y, cols+image.cols-center_rgb.x)[1] = image.at<Vec3b>(rows, cols)[1];
//             big_image.at<Vec3b>(rows+image.rows-center_rgb.y, cols+image.cols-center_rgb.x)[2] = image.at<Vec3b>(rows, cols)[2];
//        }

//  cv::Size dst_sz(big_image.size().width, big_image.size().height);
//  //获取旋转矩阵（2x3矩阵）

//  Point2f center_point(image.cols, image.rows);
//  cv::Mat rot_mat = cv::getRotationMatrix2D(center_point, angle, 1.0);
  Rect rect_img(min_x , min_y , max_x-min_x, max_y-min_y);
//  //根据旋转矩阵进行仿射变换
//  cv::warpAffine(big_image, Shift_img, rot_mat, dst_sz);
  Mat small_img_rgb = image(rect_img);
  imshow("Shift", small_img_rgb);
/*  int red_sum = 0;
  for(int k=0; k<small_img_rgb.rows; k++)
  {for (int r=0; r<small_img_rgb.cols; r++) {
          if(small_img_rgb.at<Vec3b>(k,r)[0]>38 && small_img_rgb.at<Vec3b>(k,r)[0]<70 && small_img_rgb.at<Vec3b>(k,r)[1]>20 && small_img_rgb.at<Vec3b>(k,r)[1]<50 && small_img_rgb.at<Vec3b>(k,r)[3]>130 && small_img_rgb.at<Vec3b>(k,r)[3]<170)
              red_sum++;
      }
  }
  red_num=red_sum;
//  cout<<"红色像素个数:"<<red_num<<endl;

  Mat small_img_hsv;
  cvtColor(small_img_rgb ,small_img_hsv, CV_RGB2HSV);
  imshow("Shift_hsv", small_img_hsv);
  int red_sum = 0;
  for(int k=0; k<small_img_hsv.rows; k++)
      for (int r=0; r<small_img_hsv.cols; r++) {
          if(small_img_hsv.at<Vec3b>(k,r)[0]<50)
              red_sum++;
      }
  cout<<"red_sum:"<<red_sum<<endl;
*/
  //if(zhongxin.x!=0 && num>=50) break;
  Mat image_object = imread("/home/yaohui/robot_project/templet2.jpeg", IMREAD_GRAYSCALE);
  Mat image_scene ;
  cvtColor(small_img_rgb, image_scene,   CV_BGR2GRAY );
  imshow("templet", image_scene);

  //检测特征点
  const int minHessian = 700;
  Ptr<SIFT> sfd=SIFT::create(minHessian);
  vector<KeyPoint>keypoints_object, keypoints_scene;
  sfd->detect(image_object, keypoints_object);
  sfd->detect(image_scene, keypoints_scene);

  //计算特征点描述子
  Mat descriptors_object, descriptors_scene;
  sfd->compute(image_object, keypoints_object, descriptors_object);
  sfd->compute(image_scene, keypoints_scene, descriptors_scene);

  //使用FLANN进行特征点匹配
  FlannBasedMatcher matcher;
  vector<DMatch>matches;
  matcher.match(descriptors_object, descriptors_scene, matches);

  //计算匹配点之间最大和最小距离
  double max_dist = 0;
  double min_dist = 100;
  for (int i = 0; i < descriptors_object.rows; i++)
  {
      double dist = matches[i].distance;
      if (dist < min_dist)
      {
          min_dist = dist;
      }
      else if (dist > max_dist)
      {
          max_dist = dist;
      }
  }
  printf("Max dist: %f \n", max_dist);
  printf("Min dist: %f \n", min_dist);

  //绘制“好”的匹配点
  vector<DMatch>good_matches;
  for (int i = 0; i < descriptors_object.rows; i++)
  {
      if (matches[i].distance< 200)
      {
          good_matches.push_back(matches[i]);
      }
  }
  Mat image_matches;
  drawMatches(image_object, keypoints_object, image_scene, keypoints_scene, good_matches, image_matches,
      Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  cout<<"Point num:"<<good_matches.size()<<endl;

  if (good_matches.size()>10)
      goal = true;
  else {
      goal = false;
  }

  namedWindow("匹配图像", WINDOW_AUTOSIZE);
  imshow("匹配图像", image_matches);

  cv::waitKey(1);

  Eigen::Vector4d mail_camera;
  mail_camera << pointzhongxin[0],
                  pointzhongxin[1],
                  pointzhongxin[2],
                  1;
  Eigen::Vector4d mail_world = ret_eye_base*mail_camera;
  std::vector<float> current_pose(6);
  current_pose.at(0) =  mail_world(0)*1000;
  current_pose.at(1) =  mail_world(1)*1000;
  current_pose.at(2) =  (mail_world(2)*1000) + height_jiazhua;
  current_pose.at(3) =  -178.69;
  current_pose.at(4) =  2.38;
  current_pose.at(5) =  0;
  cout<<current_pose[0]<<endl;
  cout<<current_pose[1]<<endl;
  cout<<current_pose[2]<<endl;

//  std::vector<float> working_joints(6);
//  working_joints.at(0) = -2.58;
//  working_joints.at(1) = 162.89;
//  working_joints.at(2) = 80.78;
//  working_joints.at(3) = 25.06;
//  working_joints.at(4) = -87.28;
//  working_joints.at(5) = 88.34;
//  r.jointMove(working_joints, 15);

  r.go(current_pose, 10);
  sleep(2);
  r.controlIO(3,true);
  //r.jointMove(initial_joints, 20);
  if (goal)
      r.jointMove(final_joints1, 20);
  else {
      r.jointMove(final_joints2, 20);
  }
  r.controlIO(3,false);
  return 0;
}
