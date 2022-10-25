#include <iostream>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
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
    int eye_width;
    float ear;

    void get_EAR(Mat in_frame, full_object_detection landmarks)
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

        ear = (ear_left + ear_right) / 2;
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
        return eye_roi;
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