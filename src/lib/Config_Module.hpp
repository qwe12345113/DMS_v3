/*gettime6.c */
#include <iostream>
#include <map>
#include <cstring>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace dlib;
using namespace std;

class Config
{
public:
    int pitch_tresh, mear_time_tresh, ear_time_tresh, yaw_time_tresh, pitch_time_tresh;
    float ear_act_tresh, mear_act_tresh, pitch_act_tresh, yaw_act_tresh;
    float ear_tresh, mear_tresh, avg_pitch=0, avg_yaw=0;
    
    int head_basic, fps_lim, head_moveY, LAG;
    int INPUT_COL, INPUT_ROW, ROI_COL, ROI_ROW;
    int ROI_Xl, ROI_Xm, ROI_Xr, ROI_Y;
    bool record;
    cv::Rect myROI;
    cv::Size mySize;

    void init(string file_name)
    {
        std::map<std::string, float> data = loadcfg(file_name);

        fps_lim = data["fps_lim"];
        head_moveY = data["head_moveY"];
        pitch_tresh = data["pitch_tresh"];
        mear_time_tresh = data["mear_time_tresh"];
        ear_time_tresh = data["ear_time_tresh"];
        yaw_time_tresh = data["yaw_time_tresh"];
        pitch_time_tresh = data["pitch_time_tresh"];
        head_moveY = data["head_moveY"];
        LAG = data["LAG"];
        record = data["record"];

        INPUT_COL = data["INPUT_COL"];
        INPUT_ROW = data["INPUT_ROW"];
        ROI_COL = (int)(INPUT_COL / 2); 
        ROI_ROW = INPUT_ROW;

        ROI_Xl = data["ROI_Xl"];
        ROI_Xm = (int)(INPUT_COL / 4);
        ROI_Xr = (int)(INPUT_COL / 2);
        ROI_Y = data["ROI_Y"];
        
        mySize = cv::Size(INPUT_COL, INPUT_ROW);
        myROI = cv::Rect(ROI_Xm, ROI_Y, ROI_COL, ROI_ROW);
    }

    void cal_head_tresh(float pitch, float yaw, bool lag)
    {
        avg_pitch = avg_pitch + abs(pitch);
        avg_yaw = avg_yaw + yaw;
        if(lag)
        {
            avg_pitch = avg_pitch / LAG;
            avg_yaw = avg_yaw / LAG;
        }
    }

    void cal_normal_state(std::vector<full_object_detection> &shapes)
    {
        int size = shapes.size();
        float a[size], b[size], d[size];

        for (int i = 0; i < size; i++)
        {
            a[i] = eye_aspect_ratio(shapes.at(i));
            b[i] = mouth_aspect_ratio(shapes.at(i));
            d[i] = shapes.at(i).part(30).y();
        }
        ear_tresh = mean(a, size) - 0.03;
        mear_tresh = mean(b, size) + 0.5;
        head_basic = mean(d, size);
    }

private:
    std::map<std::string, float> loadcfg(string file_name)
    {
        std::map<std::string, float> data;
        const static size_t BUFSIZE = 4096;
        string s;
        float val;
        s.resize(BUFSIZE);

        ifstream fin(file_name);

        while (fin >> s)
        {
            fin >> val;
            data[s] = val;
            fin.getline(&s[0], BUFSIZE, '\n');
        }
        return data;
    }

    float mean(float data[], int len)
    {
        float sum = 0.0, mean = 0.0;

        for (int i = 0; i < len; ++i){
            sum += data[i];
        }

        mean = sum / len;
        return mean;
    }

    float distanceCalculate(point &p1, point &p2)
    {
        float x = p1.x() - p2.x(); // calculating number to square in next step
        float y = p1.y() - p2.y();
        return sqrt(pow(x, 2) + pow(y, 2)); // calculating Euclidean distance
    }

    float eye_aspect_ratio(full_object_detection &shape)
    {
        float r_ear = (distanceCalculate(shape.part(43), shape.part(47)) + distanceCalculate(shape.part(44), shape.part(46))) / (2 * distanceCalculate(shape.part(42), shape.part(45)));
        float l_ear = (distanceCalculate(shape.part(37), shape.part(41)) + distanceCalculate(shape.part(38), shape.part(40))) / (2 * distanceCalculate(shape.part(36), shape.part(39)));
        return (l_ear + r_ear) / 2;
    }

    float mouth_aspect_ratio(full_object_detection &shape)
    {
        float a = distanceCalculate(shape.part(61), shape.part(67));
        float b = distanceCalculate(shape.part(62), shape.part(66));
        float c = distanceCalculate(shape.part(63), shape.part(65));
        float d = distanceCalculate(shape.part(60), shape.part(64));

        return (a + b + c) / (3 * d);
    }
};