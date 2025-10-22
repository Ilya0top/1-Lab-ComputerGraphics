#include <iostream>
#include <string>
#include <exception>
#include "TestRunner.h"

using namespace std;

int main()
{
    setlocale(LC_ALL, "RUSSIAN");
    string imagePath = "Image\\original.jpg";
    cv::Mat image = cv::imread(imagePath);

    if (image.empty())
        throw exception("Не удалось загрузить изображение");



    // 1. Comprehensive тестирование
    TestRunner::runComprehensiveTest(image);

    // 2. Оптимизированный тест
    //TestRunner::runOptimizedTest(image);

}