#include <iostream>
#include "cmath"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace dlib;

class HeadPoseEstimator
{
public:
    full_object_detection keypoints;
    Mat frame;
    bool verbose = false;
    float roll;
    float pitch;
    float yaw;

    void get_pose(cv::Mat frame, full_object_detection landmarks)
    {
        float A[3][3] = {200, 0, 0, 0, 200, 0, 0, 0, 200};
        float K[3][3] = {(float)frame.cols, 0, ((float)frame.rows / 2), 0, (float)frame.cols, ((float)frame.cols / 2), 0., 0., 1.};
        float D[4][1] = {0., 0., 0., 0.};

        cv::Mat axis = cv::Mat(3, 3, CV_32FC1, A);
        cv::Mat camera_matrix = cv::Mat(3, 3, CV_32FC1, K);
        cv::Mat dist_coeffs = cv::Mat(4, 1, CV_32FC1, D);

        std::vector<cv::Point3f> model_points;
        model_points.push_back(cv::Point3f(0.0, 0.0, 0.0));
        model_points.push_back(cv::Point3f(0.0, -330.0, -65.0));
        model_points.push_back(cv::Point3f(-225.0, 170.0, -135.0));
        model_points.push_back(cv::Point3f(255.0, 170.0, -135.0));
        model_points.push_back(cv::Point3f(-150.0, -150.0, -125.0));
        model_points.push_back(cv::Point3f(150.0, -150.0, -125.0));

        std::vector<cv::Point2f> image_points;
        image_points.push_back(cv::Point2f(landmarks.part(30).x(), landmarks.part(30).y()));
        image_points.push_back(cv::Point2f(landmarks.part(8).x(), landmarks.part(8).y()));
        image_points.push_back(cv::Point2f(landmarks.part(36).x(), landmarks.part(36).y()));
        image_points.push_back(cv::Point2f(landmarks.part(45).x(), landmarks.part(45).y()));
        image_points.push_back(cv::Point2f(landmarks.part(48).x(), landmarks.part(48).y()));
        image_points.push_back(cv::Point2f(landmarks.part(54).x(), landmarks.part(54).y()));

        cv::Point nose(image_points.at(0).x, image_points.at(0).y);

        cv::Mat rvec, tvec;
        cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rvec, tvec); //

        if (!rvec.empty() && !tvec.empty())
        {
            cv::solvePnPRefineVVS(model_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);

            std::vector<cv::Point2f> nose_end_point2D;
            cv::projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs, nose_end_point2D);

            cv::Mat Rmat;

            cv::Rodrigues(rvec, Rmat);

            // cout << Rmat.at<Vec3d>(0,0)[0]<<endl;
            // cout << Rmat << endl;
            float arr[3][3] = {};
            for (int i = 0; i < 3; i++){
                for (int j = 0; j < 3; j++)
                    arr[i][j] = Rmat.at<Vec3d>(i)[j];
            }
            rotationMatrixToEulerAngles(arr);
            rotationMatrixToEulerAngles2(arr);
            // rotationMatrixToEulerAngles3(rvec);
            

            if (verbose)
            {
                draw_pose_info(frame, nose, nose_end_point2D);
            }
        }
    }

private:
    void rotationMatrixToEulerAngles(float R[3][3])
    {
        float roll1, pitch1, yaw1;
        float sy = sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);

        if (!(sy < 1e-6))
        {
            roll1 = atan2(R[2][1], R[2][2]);
            pitch1 = atan2(-R[2][0], sy);
            yaw1 = atan2(R[1][0], R[0][0]);
        }
        else
        {
            roll1 = atan2(-R[1][2], R[1][1]);
            pitch1 = atan2(-R[2][0], sy);
            yaw1 = 0;
        }

        // roll = roll * 180 / M_PI;
        pitch = pitch1 * 180 / M_PI; // turn around head
        // yaw = yaw * 180 / M_PI;
        //cout << pitch << endl; // 正面的基準在角度在10多 // need to moderfy
    }

    void rotationMatrixToEulerAngles2(float R[3][3])
    {

        float q0 = sqrt(1 + R[0][0] + R[1][1] + R[2][2]) / 2.0;
        float q1 = (R[2][1] - R[1][2]) / (4.0 * q0);
        float q2 = (R[0][2] - R[2][0]) / (4.0 * q0);
        float q3 = (R[1][0] - R[0][1]) / (4.0 * q0);

        // float t1 = 2.0 * (q0 * q2 + q1 * q3);

        // float roll = asin(2.0 * (q0 * q2 + q1 * q3));
        // float pitch = atan2(2.0 * (q0 * q1 - q2 * q3), (q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3));
        yaw = atan2(2.0 * (q0 * q3 - q1 * q2), (q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3)); // lower head      
    }

    // void rotationMatrixToEulerAngles3(cv::Mat rotation_vector)
    // {
    //     double theta = cv::norm(rotation_vector, CV_L2);
    //     double w = cos(theta / 2);
    //     double x = sin(theta / 2) * rotation_vector.at<double>(0,0) / theta;
    //     double y = sin(theta / 2) * rotation_vector.at<double>(0,1) / theta;
    //     double z = sin(theta / 2) * rotation_vector.at<double>(0,2) / theta;

    //     double ysqr = y*y;

    //     // x-axis
    //     double t0 = +2.0 * (w*x + y*z);
    //     double t1 = +1.0 - 2.0 * (x*x + ysqr);
    //     double pitch3 = atan2(t0, t1);

    //     // y-axis
    //     double t2 = +2.0 * (w*y - z*x);
    //     t2 = t2 > 1.0 ? 1.0:t2;
    //     t2 = t2 < -1.0 ? -1.0:t2;
    //     double yaw3 = asin(t2);

    //     // z-axis
    //     double t3 = +2.0 * (w*z + x*y);
    //     double t4 = +1.0 - 2.0 * (ysqr + z*z);
    //     double roll3 = atan2(t3, t4);

    // }

    void draw_pose_info(cv::Mat frame, cv::Point img_point, std::vector<cv::Point2f> point_proj)
    {
        cv::line(frame, img_point, point_proj.at(0), Scalar(255, 0, 0));
        cv::line(frame, img_point, point_proj.at(1), Scalar(0, 255, 0));
        cv::line(frame, img_point, point_proj.at(2), Scalar(0, 0, 255));
    }
};