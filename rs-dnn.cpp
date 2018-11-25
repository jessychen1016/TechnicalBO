// This example is derived from the ssd_mobilenet_object_detection opencv demo
// and adapted to be used with Intel RealSense Cameras
// Please see https://github.com/opencv/opencv/blob/master/LICENSE
// under the Intel® Core™ i5-3570 CPU @ 3.40GHz × 4 

#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include "cv-helpers.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <cmath>

#include "iostream"
#include "time.h"


const size_t inWidth      = 600;
const size_t inHeight     = 900;
const float WHRatio       = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal       = 127.5;
int mat_columns;
int mat_rows;
int length_to_mid;

double depth_length_coefficient(double depth){
    
    double length;
    length = 48.033*depth+5.4556;
    return length;

}

int main(int argc, char** argv) try
{
    using namespace cv;
    using namespace cv::dnn;
    using namespace rs2;
    using namespace std;
    using namespace rs400;

    context ctx;
    auto devices = ctx.query_devices();
    size_t device_count = devices.size();
    if (!device_count)
    {
        cout <<"No device detected. Is it plugged in?\n";
        return EXIT_SUCCESS;
    }

    // Get the first connected device
    auto dev = devices[0];

    if (dev.is<rs400::advanced_mode>())
    {
        auto advanced_mode_dev = dev.as<rs400::advanced_mode>();
        // Check if advanced-mode is enabled
        if (!advanced_mode_dev.is_enabled())
        {
            // Enable advanced-mode
            advanced_mode_dev.toggle_advanced_mode(true);
        }
    }
    else
    {
        cout << "Current device doesn't support advanced-mode!\n";
        return EXIT_FAILURE;
    }

    // rs2_frame_queue* rs2_create_frame_queue;
    // rs2_processing_block* rs2_create_disparity_transform_block;
    // rs2_processing_block* rs2_create_decimation_filter_block;
    // rs2_processing_block* rs2_create_hole_filling_filter_block;
    // rs2_processing_block* rs2_create_spatial_filter_block;
    // rs2_processing_block* rs2_create_temporal_filter_block;
    

    // //Disparity_Transform
    // rs2_frame_queue* rs2_create_frame_queue;
    // rs2_processing_block* rs2_create_disparity_transform_block;
    // void rs2_start_processing_queue(rs2_create_disparity_transform_block, rs2_create_frame_queue, rs2_error** error);
    // void rs2_set_option(rs2_create_disparity_transform_block, RS2_OPTION_FILTER_MAGNITUDE, float value, rs2_error** error);
    // void rs2_set_option(rs2_create_disparity_transform_block, RS2_OPTION_FILTER_SMOOTH_ALPHA, float value, rs2_error** error);
    // void rs2_set_option(rs2_create_disparity_transform_block, RS2_OPTION_FILTER_SMOOTH_DELTA, float value, rs2_error** error);
    // void rs2_set_option(rs2_create_disparity_transform_block, RS2_OPTION_HOLES_FILL, float value, rs2_error** error);


    // // Decimation
    // rs2_frame_queue* rs2_create_frame_queue;
    // void rs2_start_processing_queue(rs2_create_decimation_filter_block, rs2_create_frame_queue, rs2_error** error);
    // void rs2_set_option(rs2_create_decimation_filter_block, RS2_OPTION_FILTER_MAGNITUDE, float value, rs2_error** error);

    // // Spacial_Filter
    // rs2_frame_queue* rs2_create_frame_queue;
    // void rs2_start_processing_queue(rs2_create_spatial_filter_block, rs2_create_frame_queue, rs2_error** error);
    // void rs2_set_option(rs2_create_spatial_filter_block, RS2_OPTION_FILTER_MAGNITUDE, float value, rs2_error** error);
    // void rs2_set_option(rs2_create_spatial_filter_block, RS2_OPTION_FILTER_SMOOTH_ALPHA, float value, rs2_error** error);
    // void rs2_set_option(rs2_create_spatial_filter_block, RS2_OPTION_FILTER_SMOOTH_DELTA, float value, rs2_error** error);
    // void rs2_set_option(rs2_create_spatial_filter_block, RS2_OPTION_HOLES_FILL, float value, rs2_error** error);

    // // Temporal_Filter
    // rs2_frame_queue* rs2_create_frame_queue;
    // void rs2_start_processing_queue(rs2_create_temporal_filter_block, rs2_create_frame_queue, rs2_error** error);
    // void rs2_set_option(rs2_create_temporal_filter_block, RS2_OPTION_FILTER_SMOOTH_ALPHA, float value, rs2_error** error);
    // void rs2_set_option(rs2_create_temporal_filter_block, RS2_OPTION_FILTER_SMOOTH_DELTA, float value, rs2_error** error);
    // void rs2_set_option(rs2_create_temporal_filter_block, RS2_OPTION_HOLES_FILL, float value, rs2_error** error);
    // // Hole_Filling
    // rs2_frame_queue* rs2_create_frame_queue;
    // void rs2_start_processing_queue(rs2_create_hole_filling_block, rs2_create_frame_queue, rs2_error** error);
    // void rs2_set_option(rs2_create_hole_filling_filter_block, RS2_OPTION_HOLES_FILL, float value, rs2_error** error);


   
    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    std::ifstream t("/home/jessy/Desktop/rs_examples/config/labconfig4351.json");
    std::string str((std::istreambuf_iterator<char>(t)),
                std::istreambuf_iterator<char>());
    rs400::advanced_mode dev4json = config.get_device();
    dev4json.load_json(str);
    

    auto profile = config.get_stream(RS2_STREAM_COLOR)
                         .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),
                        profile.height());
    }
    else
    {
        cropSize = Size(profile.width(),
                        static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
                    (profile.height() - cropSize.height) / 2),
              cropSize);

    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);


    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"
       
    int iLowH = 0;
    int iHighH = 38;
    int iLowS = 71; 
    int iHighS = 255;
    
    int iLowV = 203;
    int iHighV = 255;

    //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighH, 179);

    cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);

    cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);

    // double distance[60] = {0};
    double last_x_meter = 0;
    double this_x_meter = 0;
    double y_vel = 0;
    double x_vel = 0;
    double velocity;
    // int para1 = 200 ,para2 = 50;


     while (cvGetWindowHandle(window_name))
    // for(int i = 0; i<60 && cvGetWindowHandle(window_name) ; i++)
    {

        auto start_time = clock();

        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, 
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        // imshow ("image", color_mat);
        auto depth_mat = depth_frame_to_meters(pipe, depth_frame);
        // imshow ("image_depth", depth_mat);
        Mat inputBlob = blobFromImage(color_mat, inScaleFactor,
                                      Size(inWidth, inHeight), meanVal, false); //Convert Mat to batch of images
        

        Mat Gcolor_mat;
        Mat Gdepth_mat;


        GaussianBlur(color_mat,Gcolor_mat,Size(11,11),0);
        

        // Crop both color and depth frames
        Gcolor_mat = Gcolor_mat(crop);
        depth_mat = depth_mat(crop);

         //start of mod
        mat_rows = Gcolor_mat.rows;
        mat_columns =Gcolor_mat.cols;
        // cout<< mat_columns<< endl;
        Mat imgHSV;
        vector<Mat> hsvSplit;
        cvtColor(Gcolor_mat, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    
        //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做
        split(imgHSV, hsvSplit);
        equalizeHist(hsvSplit[2],hsvSplit[2]);
        merge(hsvSplit,imgHSV);
        Mat imgThresholded;
    
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
    
        //开操作 (去除一些噪点)
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element);
    
        //闭操作 (连接一些连通域)
        morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);
    
        imshow("Thresholded Image", imgThresholded); //show the thresholded image
    //   imshow("Original", Gcolor_mat); //show the original image
    
        char key = (char) waitKey(1);
                
        //end of mod
    //    vector<Vec3f> circles;
    //    HoughCircles(imgThresholded,circles,CV_HOUGH_GRADIENT,1,1000,para1,para2,0,1000);
    //    Point hough_center(cvRound(circles[0][0]), cvRound(circles[0][1]));

        vector<vector<cv::Point>> contours ;
        cv::findContours(imgThresholded,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
        double maxArea = 0;
        vector<cv::Point> maxContour;
        for(size_t i = 0; i < contours.size(); i++)
        {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea)
            {
                maxArea = area;
                maxContour = contours[i];
            }
        }
        cv::Rect maxRect = cv::boundingRect(maxContour);

        // auto object =  maxRect & Rect (0,0,depth_mat.cols, depth_mat.rows );
        auto object =  maxRect ;
        auto moment = cv::moments(maxContour,true);
        // std::cout<<"x="<< moment.m10 / moment.m00<<"  y="<<moment.m01/moment.m00<<std::endl;
        // std::cout<<"depth"<< depth_mat.at<double>((int)moment.m01 / moment.m00,(int)moment.m10/moment.m00)<<std::endl;
        
        // Calculate mean depth inside the detection region
        // This is a very naive way to estimate objects depth
        // but it is intended to demonstrate how one might 
        // use depht data in general
        Scalar depth_m;            
        if (moment.m00==0 ){
            moment.m00 = 1;
        }
        Point moment_center (moment.m10 / moment.m00, moment.m01/moment.m00);
        depth_m = depth_mat.at<double>((int)moment.m01 / moment.m00,(int)moment.m10/moment.m00);
        double magic_distance = depth_m[0] * 1.062;
        std::ostringstream ss;
        ss << " Ball Detected ";
        ss << std::setprecision(3) << magic_distance << " meters away" ;
        String conf(ss.str());
        // distance[i]=magic_distance;

        rectangle(Gcolor_mat, object, Scalar(0, 255, 0));
        int baseLine = 0;
        Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        auto center = (object.br() + object.tl())*0.5;
        center.x = center.x - labelSize.width / 2;
        center.y = center.y + 30 ;


        velocity = sqrt(y_vel*y_vel + x_vel*x_vel);
        std::ostringstream ss_v;
        ss_v << " The speed is ";
        ss_v << std::setprecision(3) << velocity << " m/s" ;
        String conf_v(ss_v.str());
        Size labelSize_v = getTextSize(ss_v.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        auto center_v = (object.br() + object.tl())*0.5;
        center_v.x = 80- labelSize_v.width / 2;
        center_v.y = 475;

        rectangle(Gcolor_mat, Rect(Point(center_v.x, center_v.y - labelSize_v.height),
            Size(labelSize_v.width, labelSize_v.height + baseLine)),
            Scalar(255, 255, 255), CV_FILLED);
        putText(Gcolor_mat, ss_v.str(), center_v,
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
        

        rectangle(Gcolor_mat, Rect(Point(center.x, center.y - labelSize.height),
            Size(labelSize.width, labelSize.height + baseLine)),
            Scalar(255, 255, 255), CV_FILLED);
        putText(Gcolor_mat, ss.str(), center,
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

        // calculate length to midline
        length_to_mid = abs(moment.m10 / moment.m00-160)*depth_length_coefficient(magic_distance)/320;
        cout << length_to_mid << endl;


        imshow(window_name, Gcolor_mat);
        if (waitKey(1) >= 0) break;
        imshow("heatmap", depth_mat);
        this_x_meter = magic_distance;
        auto end_time = clock();
        x_vel = (this_x_meter - last_x_meter)/(end_time-start_time)*CLOCKS_PER_SEC;
        last_x_meter = this_x_meter;
        // std::cout<<1000.000*(end_time-start_time)/CLOCKS_PER_SEC<<std::endl;

        

    }
    
    // double dmean = 0;
    // double dvariance = 0; 

    // for(int i=30 ; i<60; i++){

    //     dmean += distance[i];
    // }
    // dmean /= 30.0;

    // for (int i=30; i<60; i++ ){
    //     dvariance += (distance[i]-dmean) * (distance[i]-dmean);
    // }
    
    // dvariance /= 29;

    // std::cout << dmean << ",      "<< sqrt(dvariance) << std::endl;  
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

