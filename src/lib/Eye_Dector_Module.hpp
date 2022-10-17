#include <iostream>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace dlib;

class EyeDetector
{
public:
    
    full_object_detection keypoints;
    Mat frame;
    bool show_processing=false;
    int eye_width;
    float gaze_score;

    float get_EAR(Mat in_frame, full_object_detection landmarks)
    {

        keypoints = landmarks;
        frame = in_frame;
        full_object_detection pts = keypoints;
        int i = 0;
        std::vector<point> eye_pts_l;
        std::vector<point> eye_pts_r;

        for (int n = 36; n < 42; n++)
        {
            eye_pts_l.push_back(point(pts.part(n).x(), pts.part(n).y()));
            eye_pts_r.push_back(point(pts.part(n + 6).x(), pts.part(n + 6).y()));
            i++;
        }

        float ear_left = EAR_eye(eye_pts_l);
        float ear_right = EAR_eye(eye_pts_r);

        float ear_avg = (ear_left + ear_right) / 2;
        return ear_avg;
    }


    float get_Gaze_Score(Mat in_frame, full_object_detection landmarks)
    // void get_Gaze_Score(Mat in_frame, full_object_detection landmarks)
    {
        
        keypoints = landmarks;
        frame = in_frame;
        Mat left_eye_ROI = get_ROI(36);
        Mat right_eye_ROI = get_ROI(42);
        float gaze_eye_left = get_gaze(left_eye_ROI);
        float gaze_eye_right = get_gaze(right_eye_ROI);

        if (gaze_eye_left && gaze_eye_right){
            float avg_gaze_score =  (gaze_eye_left + gaze_eye_left) / 2;
            return avg_gaze_score;
        }
        else
            return 0;
    }

private:
    
    Mat get_ROI(int left_corner_keypoint_num)
    {
        
        int kp_num = left_corner_keypoint_num;
        int xp[6], yp[6];
        for (int j = 0; j < 6; j++)
        {
            xp[j] = keypoints.part(kp_num + j).x();
            yp[j] = keypoints.part(kp_num + j).y();
        }
        
        int min_x = *min_element(xp, xp + 6) - 2;
        int max_x = *max_element(xp, xp + 6) + 2;
        int min_y = *min_element(yp, yp + 6) - 2;
        int max_y = *max_element(yp, yp + 6) + 1;
        
        cv::Rect roi(min_x, min_y, max_x-min_x, max_y-min_y);
        cv::Mat eye_roi = frame(roi);
        // string s = "./" + to_string(left_corner_keypoint_num) + ".jpg";
        // cv::imwrite(s, eye_roi);
        return eye_roi;
    }

    float get_gaze(Mat &eye_roi)
    {   
        float gaze_score = 0;
        cv::Scalar color(255, 255, 255);
        cv::Point eye_center((int)(eye_roi.cols / 2), (int)(eye_roi.rows / 2));
        
        // cv::bilateralFilter(inframe, eye_roi, 4, 40, 40);

        std::vector<Vec3f> circles;
        cv::HoughCircles(eye_roi, circles, cv::HOUGH_GRADIENT, 1, 10, 90, 6, 1, 9);
        
        
        if (!circles.empty() && circles.size()>0)
        {
            cv::Point pupil_position((int)circles.at(0)[0], (int)circles.at(0)[1]);
            if (show_processing){
                cv::circle(eye_roi, pupil_position, circles.at(0)[2], color, 1);
                cv::circle(eye_roi, pupil_position, 1, color, -1);
                
                // cv::line(eye_roi, (eye_center[0], eye_center[1]), (pupil_position[0], pupil_position[1]), color, 1);
            }
            gaze_score = LA_norm_cv(pupil_position, eye_center) / eye_center.x;
            return gaze_score;
        }
        return 0;
    }

    float LA_norm_cv(cv::Point &p1, cv::Point &p2)
    {
        float x = p1.x - p2.x;
        float y = p1.y - p2.y;
        return sqrt(pow(x, 2) + pow(y, 2));
    }

    float LA_norm(point &p1, point &p2)
    {
        float x = p1.x() - p2.x();
        float y = p1.y() - p2.y();
        return sqrt(pow(x, 2) + pow(y, 2));
    }

    float EAR_eye(std::vector<point> &eye_pts)
    {
        float ear_eye = (LA_norm(eye_pts.at(1), eye_pts.at(5)) + LA_norm(eye_pts.at(2), eye_pts.at(4))) / (2 * LA_norm(eye_pts.at(0), eye_pts.at(3)));
        return ear_eye;
    }
};