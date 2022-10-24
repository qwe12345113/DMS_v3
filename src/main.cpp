#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
// #include <dlib/opencv/cv_image.h>
// #include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "lib/Eye_Dector_Module.hpp"
#include "lib/Yawn_Dector_Module.hpp"
#include "lib/Pose_Estimation_Module.hpp"
#include "lib/Register.hpp"
#include "lib/Attention_Scorer_Module.hpp"
#include "lib/Config_Module.hpp"

using namespace std;
using namespace dlib;

/*---------------------------------------------------------------------------------------- */
// defind model structure 
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
    // sleep(waitTemp);
  }
}

void runFunc()
{
  Config cfg;
  cfg.init("../config.cfg");

  frontal_face_detector detector = get_frontal_face_detector();
  shape_predictor sp;
  deserialize("../Model/shape_predictor_68_face_landmarks.dat") >> sp;

  anet_type net;
  deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;

  cv::VideoCapture cam(0);
  cam.set(cv::CAP_PROP_FPS, 30);
  cam.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
  cam.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  cam.set(cv::CAP_PROP_CONVERT_RGB, 1);

  cv::Mat input;

  int lag = 0, frame_count=0, showName=0;  
  bool find_normal_satus_OK = false, show_detail = false, check = true, lock_cin = false;

  std::vector<full_object_detection> shapes, tmp_shapes;
  std::vector<float> threshold;

  cv::VideoWriter writer;
  if (cfg.record)
    writer.open("./demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15.0, cfg.mySize, true);

  EyeDetector Eye_det;
  HeadPoseEstimator Head_pose;
  YawnDetector yawn_det;
  AttentionScorer Scorer;
  Register usr_reg;
  usr_reg.net = net;

  string out = "", out_usr="";
  
  auto start = std::chrono::high_resolution_clock::now();

  while (cam.read(input))
  {
    cv::resize(input, input, cfg.mySize);
    cv::Mat frame = input(cfg.myROI);

    // if the img is gray scale concat image to simulate the color image
    if (frame.channels() == 1)
      cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);    
    
    //------------ opencv format to dlib format ---------------//
    array2d<rgb_pixel> img;
    assign_image(img, cv_image<bgr_pixel>(frame));

    dlib::rectangle face;
    std::vector<int> x_pos;
    //----- Number of faces detected -----// // spend 50...ms
    std::vector<dlib::rectangle> dets = detector(img);
    
    if (dets.size() > 0)
    {
      /*if more than 1 face, find the face which is nearest the INPUT_COL/2*/
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

      if(strcmp(reg_command.substr(0, 3).c_str(), "reg") == 0)
      {
        auto start_reg = std::chrono::high_resolution_clock::now();
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
          // sleep(WAIT_CAPTURE);
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
        auto finish_reg = std::chrono::high_resolution_clock::now();
        cout << "during " << (std::chrono::duration_cast<std::chrono::milliseconds>(finish_reg-start_reg).count()) << " ms" << endl;
        cout << "during " << (std::chrono::duration_cast<std::chrono::seconds>(finish_reg-start_reg).count()) << " s" << endl;
      }

      else if (strcmp(reg_command.c_str(), "rec") == 0)
      {
        auto start_rec = std::chrono::high_resolution_clock::now();
        usr_reg.recognized = false;
        usr_reg.TakePhoto(face, frame);
        usr_reg.recognize_usr();

        if(usr_reg.recognized){
          out_usr="Hi " + usr_reg.usr;
          showName = 0;
        }
        
        reg_command = "";
        cout << endl;
        auto finish_rec = std::chrono::high_resolution_clock::now();
        cout << "during " << (std::chrono::duration_cast<std::chrono::milliseconds>(finish_rec-start_rec).count()) << " ms" << endl;
        cout << "during " << (std::chrono::duration_cast<std::chrono::seconds>(finish_rec-start_rec).count()) << " s" << endl;
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
            Eye_det.get_EAR(frame, landmarks);
            yawn_det.get_EAR(frame, landmarks);
            Head_pose.get_pose(frame, landmarks);
            if (show_detail)
            {
              out = "EAR: " + to_string(Eye_det.ear);
              cv::putText(frame, out, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 1);

              out = "MEAR: " + to_string(yawn_det.m_ear);
              cv::putText(frame, out, Point(10, 130), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 1);
            }

            Scorer.eval_scores(Eye_det.ear, yawn_det.m_ear, Head_pose.pitch, Head_pose.yaw, shapes.at(0).part(30).y());

            if (Scorer.is_asleep)
              cv::putText(frame, "CLOSE EYES !", Point(10, cfg.INPUT_ROW-80), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
            if (Scorer.is_yawn)
              cv::putText(frame, "YAWN !", Point(10, cfg.INPUT_ROW-60), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
            if (Scorer.is_lower_head)
              cv::putText(frame, "LOWER HEAD !", Point(10, cfg.INPUT_ROW-40), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
            if (Scorer.is_distracted)
              cv::putText(frame, "DISTRACTED !", Point(10, cfg.INPUT_ROW-20), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
          }
          else // calculate normal face state threshold
          {
            if (lag > cfg.LAG)
            {
              cfg.cal_normal_state(tmp_shapes);
              cfg.cal_head_tresh(Head_pose.pitch, Head_pose.yaw, true);
              cout << "calculate finish\n" << "input command : ";
                            
              Scorer.init(cfg.fps_lim, cfg.mear_tresh, cfg.ear_tresh, cfg.avg_yaw, cfg.head_basic, cfg.head_moveY, cfg.avg_pitch,
                          cfg.pitch_tresh, cfg.mear_time_tresh, cfg.ear_time_tresh, cfg.yaw_time_tresh, cfg.pitch_time_tresh);
            
              Head_pose.init(); // init pitch, yaw
              tmp_shapes.clear();
              find_normal_satus_OK = true;
            }
            else
            {
              tmp_shapes.push_back(shapes.at(0));
              Head_pose.get_pose(frame, landmarks);
              cfg.cal_head_tresh(Head_pose.pitch, Head_pose.yaw, false);
              lag++;
            }
          }
        }        
      }
    }
    else
      cv::putText(frame, "No Face!", Point(10, cfg.INPUT_ROW-40), FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255), 2);

    if(usr_reg.recognized){      
      if(showName < 150){
        cv::putText(frame, out_usr, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
      }
      showName++;
    }
    if (cfg.record)
      writer.write(frame);
    
    cv::imshow("dms", frame);
    char key = cv::waitKey(1);
    if (reg_command == "exit")
      key = 27;

    if (key == 27)
      break;

    if (!lock_cin)
    {
      reg_command = "";
      check = true;
    }
    shapes.clear();
    frame_count++;
  }
  cam.release();
  cv::destroyAllWindows();

  auto finish = std::chrono::high_resolution_clock::now();
  cout << "during " << (std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count()) << " ms" << endl;
  cout << "during " << (std::chrono::duration_cast<std::chrono::seconds>(finish-start).count()) << " s" << endl;
  cout << frame_count<< endl;
}

int main(int argc, char *argv[])
try
{
  if (argc == 1)
  {
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
