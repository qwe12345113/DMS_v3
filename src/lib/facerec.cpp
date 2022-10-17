#include "facerec.h"

using namespace dlib;
using namespace std;

std::string dirPath = "../database";

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
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

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel> &img);
// ----------------------------------------------------------------------------------------

void getDirLists(string path, std::vector<std::string> &files)
{
    DIR *pDir;
    DIR *qDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
        return;
    // cout << "Folder : " << path << endl;
    while ((ptr = readdir(pDir)) != 0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            string tempString = path + "/" + ptr->d_name;
            if ((qDir = opendir(tempString.c_str())) != NULL)
            {
                files.push_back(ptr->d_name);
                // cout << "DIR " << ptr->d_name << endl;
            }
            closedir(qDir);
        }
    }
    closedir(pDir);
}

void getFileLists(string path, std::vector<std::string> &files)
{
    DIR *pDir;
    DIR *qDir;
    char *p;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
        return;
    // cout << "Folder : " << path << endl;
    while ((ptr = readdir(pDir)) != 0)
    {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            string tempString = path + "/" + ptr->d_name;
            if (!(qDir = opendir(tempString.c_str())) && (p = strstr(ptr->d_name, ".txt")))
            {
                files.push_back(ptr->d_name);
                // cout << "JPG " << ptr->d_name << endl;
            }
            closedir(qDir);
        }
    }
    closedir(pDir);
}

string checkFaceRecognition(string filename, string avoid, anet_type net)
{
    std::vector<std::string> photoNames, photoLists;
    string bestAccount = "unknow", bestPhoto = "unknow";
    float bestDistance = BEST_THRESHOLD;
    // struct timeval time_now;
    // gettimeofday(&time_now, NULL);
    // long milli_time_step1 = time_now.tv_sec * 1000 + time_now.tv_usec / 1000;
    // anet_type net;
    // deserialize("../Model/dlib_face_recognition_resnet_model_v1.dat") >> net;

    matrix<rgb_pixel> img_source;
    load_image(img_source, filename);
    matrix<float, 0, 1> face_source = net(img_source);
    photoNames.clear();
    getDirLists(dirPath, photoNames);
    // cout << "DIR size " << photoNames.size() << endl;
    if (photoNames.size() > 0)
    {
        
        for (size_t i = 0; i < photoNames.size(); i++)
        {
            // cout << "2" << endl;
            if ((strcmp(avoid.c_str(), "ALL_RECOGNITION") == 0) || (strcmp(photoNames[i].c_str(), avoid.c_str()) != 0))
            {
                
                photoLists.clear();
                string accountDir = dirPath + "/" + photoNames[i];
                getFileLists(accountDir, photoLists);
                // cout << "JPG file in "<< dirPath + "/" + photoNames[i] << " size " << photoLists.size() << endl;
                // cout << photoLists.size() << endl;
                if (photoLists.size() > 0)
                {
                    // cout << "4" << endl;
                    for (size_t j = 0; j < photoLists.size(); j++)
                    {
                        // cout << "5" << endl;
                        string accountPhoto = accountDir + "/" + photoLists[j];
                        // cout << accountPhoto << endl;
                        ifstream vector(accountPhoto);
                        matrix<float, 0, 1> face_rec;
                        face_rec.set_size(128);
                        if (vector.is_open())
                        {
                            for (int k = 0; k < 128; k++)
                            {
                                vector >> face_rec(0, k);
                            }
                            auto distance = length(face_source - face_rec);
                            // cout << "Photo " << accountPhoto << " distance " << distance << endl;
                            if (distance < bestDistance && distance > 0.01)
                            {
                                bestAccount = photoNames[i];
                                bestPhoto = photoLists[j];
                                bestDistance = distance;
                            }
                        }
                    }
                }
            }
        }
    }
    // gettimeofday(&time_now, NULL);
    // long milli_time_step2 = time_now.tv_sec * 1000 + time_now.tv_usec / 1000;
    // cout << "Face recognition account -> " << bestAccount << " ; photo -> " << bestPhoto << " ; distance -> " << bestDistance << endl;
    // cout << "during " << milli_time_step2 - milli_time_step1 << " ms" << endl;
    return bestAccount;
}