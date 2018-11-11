// This example is derived from the ssd_mobilenet_object_detection opencv demo
// and adapted to be used with Intel RealSense Cameras
// Please see https://github.com/opencv/opencv/blob/master/LICENSE
// under the Intel® Core™ i5-3570 CPU @ 3.40GHz × 4 

#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"
#include "time.h"


const size_t inWidth      = 300;
const size_t inHeight     = 300;
const float WHRatio       = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal       = 127.5;


int main(int argc, char** argv) try
{
    using namespace cv;
    using namespace cv::dnn;
    using namespace rs2;
    using namespace std;



    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
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

    while (cvGetWindowHandle(window_name))
    {

        // auto start_time = clock();

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
        

        // Crop both color and depth frames
        color_mat = color_mat(crop);
        depth_mat = depth_mat(crop);

         //start of mod
        
    
          Mat imgHSV;
          vector<Mat> hsvSplit;
          cvtColor(color_mat, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
       
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
        //   imshow("Original", color_mat); //show the original image
       
          char key = (char) waitKey(1);
          
         if(key == 27)
                break;
                                
           //end of mod

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

                auto object =  maxRect & Rect (0,0,depth_mat.cols, depth_mat.rows );


                // Calculate mean depth inside the detection region
                // This is a very naive way to estimate objects depth
                // but it is intended to demonstrate how one might 
                // use depht data in general
                Scalar m = mean(depth_mat(object));

                std::ostringstream ss;
                ss << " Ball Detected ";
                ss << std::setprecision(2) << m[0] << " meters away";
                String conf(ss.str());

                rectangle(color_mat, object, Scalar(0, 255, 0));
                int baseLine = 0;
                Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                auto center = (object.br() + object.tl())*0.5;
                center.x = center.x - labelSize.width / 2;
                center.y = center.y + 30 ;

                rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),
                    Size(labelSize.width, labelSize.height + baseLine)),
                    Scalar(255, 255, 255), CV_FILLED);
                putText(color_mat, ss.str(), center,
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));


        imshow(window_name, color_mat);
        if (waitKey(1) >= 0) break;

        // auto end_time = clock();
        // std::cout<<1000.000*(end_time-start_time)/CLOCKS_PER_SEC<<std::endl;
    }
    

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
