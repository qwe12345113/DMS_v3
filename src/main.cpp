#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "lib/utils_math.hpp"
#include "lib/Eye_Dector_Module.hpp"
#include "lib/Yawn_Dector_Module.hpp"
#include "lib/Pose_Estimation_Module.hpp"
#include "lib/Register.hpp"
#include "lib/Attention_Scorer_Module.hpp"

using namespace std;
using namespace cv;
using namespace dlib;

/*---------------------------------------------------------------------------------------- */

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                                                  alevel0<
                                                      alevel1<
                                                          alevel2<
                                                              alevel3<
                                                                  alevel4<
                                                                      max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

/*---------------------------------------------------------------------------------------- */

#define INPUT_COL 640
#define INPUT_ROW 360

#define ROI_Xl 0
#define ROI_Xm (INPUT_COL / 4)
#define ROI_Xr (INPUT_COL / 2)
#define ROI_Y 0

#define ROI_COL (INPUT_COL / 2)
#define ROI_ROW (INPUT_ROW)

#define LAG 30

/*---------------------------------------------------------------------------------------- */

string reg_command = "";
string rec_file = "";
string user_name = "";
unsigned long waitTemp = 1;
unsigned long WAIT_CAPTURE = 100;

/*---------------------------------------------------------------------------------------- */

void getCin()
{
  string temp;
  while (1)
  {
    // if (strcmp(reg_command.c_str(), "") == 0)
    if (reg_command == "")
    {
      cout << "input command :";
      getline(cin, temp);
      reg_command = temp;
    }
    else if (reg_command == "exit")
      break;
    sleep(waitTemp);
  }
}

void runFunc()
{
  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor sp;
  deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;

  anet_type net;
  deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;

  cv::VideoCapture cam;
  cam.open(0);

  cv::Mat input;
  cv::Scalar color(0, 0, 255);

  int lag = 0, fps_lim = 12;
  // float time_lim = 1. / fps_lim ;
  bool find_normal_satus_OK = false, record = true, show_detail = false, check = true, lock_cin = false;

  std::vector<full_object_detection> shapes, tmp_shapes;
  std::vector<float> threshold;

  cv::VideoWriter writer;
  if (record)
    writer.open("./demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.0, cv::Size(INPUT_COL, INPUT_ROW), true);

  EyeDetector Eye_det;
  HeadPoseEstimator Head_pose;
  YawnDetector yawn_det;
  AttentionScorer Scorer;
  Register usr_reg;

  string out = "", out_usr="Hi User";
  float ear = 0, m_ear = 0, gaze = 0, avg_pitch = 0;
  usr_reg.net = net;

  while (cam.read(input))
  {
    // clock_t start(clock());
    cv::resize(input, input, cv::Size(INPUT_COL, INPUT_ROW));

    cv::Rect myROI(ROI_Xm, ROI_Y, ROI_COL, ROI_ROW);
    cv::Mat frame = input(myROI);
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // if the img is gray scale concat image
    // three time to simulate the color image
    if (frame.channels() == 1)
      cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

    //------------ opencv format to dlib format ---------------//
    array2d<rgb_pixel> img;
    assign_image(img, cv_image<bgr_pixel>(frame));

    //----- Number of faces detected -----//
    std::vector<dlib::rectangle> dets = detector(img);
    dlib::rectangle face;
    std::vector<int> x_pos;

    if (dets.size() > 0)
    {
      if ((dets.size() == 1))
        face = dets[0];
      else
      {
        for (int i = 0; i < (int)dets.size(); i++)
        {
          x_pos.push_back((int)abs(frame.cols / 2 - (dets[i].right() - dets[i].left()) / 2));
        }
        int index = std::min_element(x_pos.begin(), x_pos.end()) - x_pos.begin();
        face = dets[index];
      }

      if (strcmp(reg_command.substr(0, 3).c_str(), "reg") == 0)
      {

        if (check)
        {
          user_name = reg_command.substr(reg_command.find(" ") + 1);

          cout << "start to register" << endl;
          usr_reg.dirExists(user_name); // only need to do one time
          check = false;
        }

        if (usr_reg.registered && usr_reg.enough_photo)
        {
          reg_command = "";
          out = user_name + ", you are already registered! \n";
          cout << out;
          out = "";
          lock_cin = false;
          check = true;
        }

        else
        {
          lock_cin = true;
          usr_reg.registor(face, frame);
          sleep(WAIT_CAPTURE);
          if (usr_reg.n == usr_reg.photo_amount_need)
          {
            reg_command = "";
            out = user_name + ", register finish! \n";
            cout << out;
            out = "";
            check = true;
            lock_cin = false;
          }
        }
        cout << endl;
      }

      else if (strcmp(reg_command.c_str(), "rec") == 0)
      {
        usr_reg.recognized = false;
        usr_reg.TakePhoto(face, frame);
        usr_reg.recognize_usr();

        if(usr_reg.recognized){
          out_usr="Hi " + usr_reg.usr;
        }
        else{
          out_usr="Hi User";
        }
        reg_command = "";
        cout << endl;
      }

      else
      {
        full_object_detection landmarks = sp(img, face); // get face landmark

        shapes.push_back(landmarks);
        // start detect
        if (shapes.size() == 1)
        {
          if (find_normal_satus_OK)
          {
            ear = Eye_det.get_EAR(frame, landmarks);
            Scorer.get_PERCLOS(ear); // get the tired and perclos_score

            m_ear = yawn_det.get_EAR(frame, landmarks); // get mouth EAR

            Head_pose.get_pose(frame, landmarks); // frame, roll, pitch, yaw

            if (show_detail)
            {
              out = "EAR: " + to_string(ear);
              cv::putText(frame, out, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 1);

              out = "Gaze Score: " + to_string(gaze);
              cv::putText(frame, out, Point(10, 80), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 1);

              out = "PERCLOS: " + to_string(Scorer.perclos_score);
              cv::putText(frame, out, Point(10, 110), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 1);

              out = "MEAR: " + to_string(m_ear);
              cv::putText(frame, out, Point(10, 130), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 1);
            }

            Scorer.eval_scores(ear, m_ear, Head_pose.pitch, Head_pose.yaw, shapes.at(0).part(30).y());

            if (Scorer.is_asleep)
              cv::putText(frame, "CLOSE EYES !", Point(10, 280), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
            if (Scorer.is_yawn)
              cv::putText(frame, "YAWN !", Point(10, 300), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
            // if(Scorer.is_looking_away)
            //   cv::putText(frame, "LOOKING AWAY!", Point(400, 300), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 1);
            if (Scorer.is_lower_head)
              cv::putText(frame, "LOWER HEAD !", Point(10, 320), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
            if (Scorer.is_distracted)
              cv::putText(frame, "DISTRACTED !", Point(10, 340), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
          }
          else
          {
            cout << "normal status calculating" << endl;
            if (lag > LAG)
            {
              threshold = threshold_calculate(tmp_shapes);
              avg_pitch = avg_pitch / lag;
              cout << "calculate finish" << endl;
              cout << "input command :" << endl;
              find_normal_satus_OK = true;
              tmp_shapes.clear();
              Scorer.init(fps_lim, threshold.at(0), 3, 0.2, 3, 27, 0.1, 2, threshold.at(1), 2, 1, threshold.at(3), avg_pitch);
              Head_pose.pitch = 0; // init pitch
            }
            else
            {
              tmp_shapes.push_back(shapes.at(0));
              Head_pose.get_pose(frame, landmarks);
              avg_pitch = avg_pitch + abs(Head_pose.pitch);
              lag++;
            }
          }
        }
        // double duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
      }
    }
    else
      cv::putText(frame, "No Face!", Point(10, 320), FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);

    if(usr_reg.recognized)
      cv::putText(frame, out_usr, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);


    if (record)
      writer.write(frame);

    cv::imshow("123", frame);
    char key = cv::waitKey(1);
    if (reg_command == "exit")
      key = 27;

    if (key == 27)
    {
      break;
    }

    if (!lock_cin)
    {
      reg_command = "";
      check = true;
    }
    shapes.clear();
  }
  cam.release();
  cv::destroyAllWindows();
}

int main(int argc, char *argv[])
try
{
  if (argc == 1)
  {
    // user_name = argv[1];
    thread mThread1(runFunc);
    thread mThread2(getCin);
    mThread1.join();
    mThread2.join();
  }

  else
  {
    cout << "Wrong command" << endl;
  }

  cout << "finish the application" << endl;
  return 0;
}
catch (std::exception &e)
{
  cout << e.what() << endl;
}
