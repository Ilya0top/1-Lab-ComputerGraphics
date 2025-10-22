#include <iostream>
#include <string>
#include <exception>
#include "TestRunner.h"

using namespace std;

int main()
{
    setlocale(LC_ALL, "RUSSIAN");
    string imagePath = "C:\\Users\\ilabe\\OneDrive\\�����������\\CG.png";
    cv::Mat image = cv::imread(imagePath);

    if (image.empty())
        throw exception("�� ������� ��������� �����������");



    // 1. Comprehensive ������������
    TestRunner::runComprehensiveTest(image);

    // 2. ���������������� ����
    //TestRunner::runOptimizedTest(image);

}