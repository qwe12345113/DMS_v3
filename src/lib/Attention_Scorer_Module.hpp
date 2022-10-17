/*gettime6.c */
#include <iostream>
#include "cmath"
using namespace std;
using namespace dlib;

class AttentionScorer
{
public:
    full_object_detection keypoints;
    float ear_tresh=0.15, gaze_tresh, perclos_tresh=0.2, ear_time_tresh=4.0, pose_time_tresh=4.0, gaze_time_tresh=4.0, yaw_tresh=0.8;
    int pitch_tresh=35, roll_tresh, head_basic=0;
    int ear_counter = 0, eye_closure_counter = 0, gaze_counter = 0, pose_counter = 0, mear_counter = 0, yaw_counter = 0;
    int perclos_time_period = 60;
    bool verbose = false, is_tired, is_asleep, is_looking_away, is_distracted, is_yawn, is_lower_head;
    float fps, perclos_score, prev_time;
    float delta_time_frame, ear_act_tresh, mear_act_tresh, gaze_act_tresh, pose_act_tresh, yaw_act_tresh;
    float mear_tresh=2.0, mear_time_tresh=2.0, yaw_time_tresh=2.0, avg_pitch=0;

    void init(float capture_fps, float a, int b, float c, int d, int e, float f, float g, float h, int i, float j, int k, int l){
        
        ear_tresh = a;
        ear_time_tresh = b;
        gaze_tresh = c;
        gaze_time_tresh = d;
        pitch_tresh = e;
        yaw_tresh = f;
        pose_time_tresh = g;
        mear_tresh = h;
        
        mear_time_tresh = i;
        yaw_time_tresh = j;

        head_basic = k;
        avg_pitch = l;


        fps = capture_fps;
        delta_time_frame = (1.0 / capture_fps);
        ear_act_tresh = ear_time_tresh / delta_time_frame; // 33
        mear_act_tresh = mear_time_tresh / delta_time_frame; // 22
        gaze_act_tresh = gaze_time_tresh / delta_time_frame; // 33
        pose_act_tresh = pose_time_tresh / delta_time_frame; // 27.5
        yaw_act_tresh = yaw_time_tresh / delta_time_frame;  // 16.5

        // cout << "ear_act_tresh" << ear_act_tresh << endl;
        // cout << "mear_act_tresh" << mear_act_tresh << endl;
        // cout << "gaze_act_tresh" << gaze_act_tresh << endl;
        // cout << "pose_act_tresh" << pose_act_tresh << endl;
        // cout << "yaw_act_tresh" << yaw_act_tresh << endl;
        
    }

    void get_PERCLOS(float ear_score){
        bool tired = false;

        if (ear_score <= ear_tresh)
            eye_closure_counter++;

        time_t timep;
        float delta = time(&timep);

        float closure_time = eye_closure_counter * delta_time_frame;
        perclos_score = closure_time / perclos_time_period;

        
        if (perclos_score >= perclos_tresh)
            tired = true;

        if (verbose)
            cout << "Closure Time: " << closure_time << "/" << perclos_time_period << endl << "PERCLOS: " << perclos_score << endl;

        if (delta >= perclos_time_period){
            eye_closure_counter = 0;
            prev_time = time(&timep);
        }
        is_tired = tired;
    }

    void eval_scores(float ear_score, float mear_score, float head_pitch, float head_yaw, int head_y){
        bool asleep = false;
        // bool looking_away = false;
        bool distracted = false;
        bool yawn = false;
        bool lower_head = false;

        if (ear_counter >= ear_act_tresh)
            asleep = true;

        // if (gaze_counter >= gaze_act_tresh)
        //     looking_away = true;

        if (pose_counter >= pose_act_tresh)
            distracted = true;

        if (mear_counter >= mear_act_tresh)
            yawn = true;

        if (yaw_counter >= yaw_act_tresh)
            lower_head = true;
        

        // close eye
        if(ear_score <= ear_tresh){
            if(!asleep)
                ear_counter += 1;
        }
        else if (ear_counter > 0)
            ear_counter -= 1;
        
        // yawn
        if (mear_score >= mear_tresh){
            if(!yawn)
                mear_counter += 1;
        }
        else if (mear_counter > 0)
            mear_counter -= 1;
        

        // if (gaze_score >= gaze_tresh){
        //     if(!looking_away)
        //         gaze_counter += 1;
        // }
        // else if (gaze_counter > 0)
        //     gaze_counter -= 1;
        
        // distracted (turn arround head)
        // if((abs(head_pitch) > pitch_tresh) || (abs(head_yaw) > yaw_tresh)){
        
        // trun arround head
        // cout <<  abs(head_pitch-avg_pitch) << endl;
        if((abs(head_pitch-avg_pitch)> pitch_tresh)){
            if(!distracted)
                pose_counter += 1;
        }
        else if (pose_counter > 0)
            pose_counter -= 1;

        // lower head
        // cout << head_y<< endl;
        // if(abs(head_yaw > yaw_tresh) || (head_y - head_basic) > 50){
        if((head_y - head_basic) > 45){
            if(!lower_head)
                yaw_counter += 1;
        }
        else if (yaw_counter > 0)
            yaw_counter -= 1;


        is_asleep = asleep;
        // is_looking_away = looking_away;
        is_distracted = distracted;
        is_yawn = yawn;
        is_lower_head = lower_head;
    }
};