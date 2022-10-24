/*gettime6.c */
#include <iostream>
#include "cmath"
using namespace std;
using namespace dlib;

class AttentionScorer
{
public:
    full_object_detection keypoints;
    bool verbose = false, is_asleep = false, is_distracted = false, is_yawn = false, is_lower_head = false;
    int ear_counter = 0, pitch_counter = 0, mear_counter = 0, yaw_counter = 0;
    int head_basic=0, head_moveY=0;
    float avg_pitch=0;
    
    float ear_tresh=0.15, yaw_tresh=0.8, mear_tresh=2.0; 
    int pitch_tresh=35;    
    
    float ear_time_tresh=4.0, pitch_time_tresh=4.0, mear_time_tresh=2.0, yaw_time_tresh=2.0;
    float ear_act_tresh=0, mear_act_tresh=0, pitch_act_tresh=0, yaw_act_tresh=0;
    

    void init(float capture_fps, float tresh1, float tresh2, float tresh3, int tresh4, int tresh5, float tresh6, int tresh7, float tresh8, float tresh9, float tresh10, float tresh11)
    {
        mear_tresh = tresh1;
        ear_tresh = tresh2;
        yaw_tresh = tresh3;
        head_basic = tresh4;
        head_moveY = tresh5;
        avg_pitch = tresh6;
        pitch_tresh = tresh7;
        mear_time_tresh = tresh8;
        ear_time_tresh = tresh9;
        yaw_time_tresh = tresh10;
        pitch_time_tresh = tresh11;

        ear_act_tresh = ear_time_tresh / (1.0 / capture_fps); // 33
        mear_act_tresh = mear_time_tresh / (1.0 / capture_fps); // 22
        pitch_act_tresh = pitch_time_tresh / (1.0 / capture_fps); // 27.5
        yaw_act_tresh = yaw_time_tresh / (1.0 / capture_fps);  // 16.5        
    }

    void eval_scores(float ear_score, float mear_score, float head_pitch, float head_yaw, int head_y){
        
        bool asleep = false, distracted = false, yawn = false, lower_head = false;

        if (ear_counter >= ear_act_tresh)
            asleep = true;

        if (pitch_counter >= pitch_act_tresh)
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
        
        // distract
        if((abs(head_pitch-avg_pitch) > pitch_tresh)){
            if(!distracted)
                pitch_counter += 1;
        }
        else if (pitch_counter > 0)
            pitch_counter -= 1;

        // lower head
        if((head_yaw > yaw_tresh) || (head_y - head_basic) > head_moveY)
        {
            // if((head_y - head_basic) > head_moveY){
            if(!lower_head)
                yaw_counter += 1;
        }
        else if (yaw_counter > 0)
            yaw_counter -= 1;

        is_asleep = asleep;
        is_distracted = distracted;
        is_yawn = yawn;
        is_lower_head = lower_head;
    }
};