#pragma once

#include <opencv.hpp>
#include <vector>
#include <cmath>

/// <summary>
/// Конвертер между BRG и Lab цветовыми пространствами
/// </summary>
class ColorConverter
{
public:

	/// <summary>
	/// Преобразует BGR изображение в Lab пространство 
	/// </summary>
	/// <param name="bgrImage">Входное BGR bpj,hf;tybt</param>
	/// <returns>Lab изображение в формате OpenCV</returns>
	static cv::Mat BGR2Lab(const cv::Mat& bgrImage)
	{
		cv::Mat labImage(bgrImage.size(), CV_32FC3);

		for (int y = 0; y < bgrImage.rows; ++y)
			for (int x = 0; x < bgrImage.cols; ++x)
			{
				cv::Vec3b bgr = bgrImage.at<cv::Vec3b>(y, x);

				float b = bgr[0] / 255.0f, g = bgr[1] / 255.0f, r = bgr[2] / 255.0f, x_xyz, y_xyz, z_xyz;
				RGB2XYZ(r, g, b, x_xyz, y_xyz, z_xyz);

				cv::Vec3f lab = XYZ2Lab(x_xyz, y_xyz, z_xyz);
				labImage.at<cv::Vec3f>(y, x) = lab;
			}
		return labImage;
	}

	/// <summary>
	/// Преобразует Lab изображение в BGR пространство
	/// </summary>
	/// <param name="labImage">Входное Lab изображение</param>
	/// <returns>BGR изображение в формате OpenCV</returns>
	static cv::Mat Lab2BGR(const cv::Mat& labImage)
	{
		cv::Mat bgrImage(labImage.size(), CV_8UC3);

		for (int y = 0; y < labImage.rows; y++)
			for (int x = 0; x < labImage.cols; x++)
			{
				cv::Vec3f lab = labImage.at<cv::Vec3f>(y, x);

				float x_xyz, y_xyz, z_xyz;
				Lab2XYZ(lab[0], lab[1], lab[2], x_xyz, y_xyz, z_xyz);

				float r, g, b;
				XYZ2RGB(x_xyz, y_xyz, z_xyz, r, g, b);

				bgrImage.at<cv::Vec3b>(y, x) = cv::Vec3b(saturate_cast(b * 255), saturate_cast(g * 255), saturate_cast(r * 255));
			}
		return bgrImage;
	}

private:

	/// <summary>
	/// Преобразует RGB в XYZ цветовое пространство
	/// </summary>
	/// <param name="r">Красный канал</param>
	/// <param name="g">Зеленый канал</param>
	/// <param name="b">Синий канал</param>
	/// <param name="x">Выходной X канал</param>
	/// <param name="y">Выходной Y канал</param>
	/// <param name="z">Выходной Z канал</param>
	static void RGB2XYZ(float r, float g, float b, float& x, float& y, float& z)
	{
		r = (r > 0.04045f) ? pow((r + 0.055f) / 1.055f, 2.4f) : (r / 12.92f);
		g = (g > 0.04045f) ? pow((g + 0.055f) / 1.055f, 2.4f) : (g / 12.92f);
		b = (b > 0.04045f) ? pow((b + 0.055f) / 1.055f, 2.4f) : (b / 12.92f);

		x = r * 0.4124564f + g * 0.3575761f + b * 0.1804375f;
		y = r * 0.2126729f + g * 0.7151522f + b * 0.0721750f;
		z = r * 0.0193339f + g * 0.1191920f + b * 0.9503041f;

		x /= 0.95047f;
		y /= 1.00000f;
		z /= 1.08883f;
	}

	/// <summary>
	/// Преобразует XYZ в Lab цветовое пространство
	/// </summary>
	/// <param name="x">X канал XYZ</param>
	/// <param name="y">Y канал XYZ</param>
	/// <param name="z">Z канал XYZ</param>
	/// <returns>Вектор Lab значений</returns>
	static cv::Vec3f XYZ2Lab(float x, float y, float z)
	{
		auto f = [](float t) -> float
			{
				const float delta = 6.0f / 29.0f;

				if (t > pow(delta, 3))
					return pow(t, 1.0f / 3.0f);
				else
					return t / (3 * delta * delta) + 4.0f / 29.0f;
			};

		const float xn = 0.95047f;
		const float yn = 1.00000f;
		const float zn = 1.08883f;

		float l = 116.0f * f(y / yn) - 16.0f;
		float a = 500.0f * (f(x / xn) - f(y / yn));
		float b = 200.0f * (f(y / yn) - f(z / zn));

		l = l * 255.0f / 100.0f;
		a = a + 128.0f;
		b = b + 128.0f;

		return cv::Vec3f(l, a, b);
	}

	/// <summary>
	/// Преобразует Lab в XYZ цветовое пространство
	/// </summary>
	/// <param name="l">L канал Lab</param>
	/// <param name="a">a канал Lab</param>
	/// <param name="b">b канал Lab</param>
	/// <param name="x">Выходной X канал</param>
	/// <param name="y">Выходной Y канал</param>
	/// <param name="z">Выходной Z канал</param>
	static void Lab2XYZ(float l, float a, float b, float& x, float& y, float& z)
	{
		l = l * 100.0f / 255.0f;
		a = a - 128.0f;
		b = b - 128.0f;

		auto f_inv = [](float t) -> float
			{
				const float delta = 6.0f / 29.0f;
				if (t > delta)
					return t * t * t;
				else
					return 3 * delta * delta * (t - 4.0f / 29.0f);
			};

		const float xn = 0.95047f;
		const float yn = 1.00000f;
		const float zn = 1.08883f;

		float fy = (l + 16.0f) / 116.0f;
		float fx = fy + (a / 500.0f);
		float fz = fy - (b / 200.0f);

		x = xn * f_inv(fx);
		y = yn * f_inv(fy);
		z = zn * f_inv(fz);
	}

	/// <summary>
	/// Преобразует XYZ в RGB цветовое пространство
	/// </summary>
	/// <param name="x">X канал XYZ</param>
	/// <param name="y">Y канал XYZ</param>
	/// <param name="z">Z канал XYZ</param>
	/// <param name="r">Выходной красный канал</param>
	/// <param name="g">Выходной зеленый канал</param>
	/// <param name="b">Выходной синий канал</param>
	static void XYZ2RGB(float x, float y, float z, float& r, float& g, float& b)
	{
		x *= 0.95047f;
		y *= 1.00000f;
		z *= 1.08883f;

		r = x * 3.2404542f + y * -1.5371385f + z * -0.4985314f;
		g = x * -0.9692660f + y * 1.8760108f + z * 0.0415560f;
		b = x * 0.0556434f + y * -0.2040259f + z * 1.0572252f;

		r = (r > 0.0031308f) ? (1.055f * pow(r, 1.0f / 2.4f) - 0.055f) : (12.92f * r);
		g = (g > 0.0031308f) ? (1.055f * pow(g, 1.0f / 2.4f) - 0.055f) : (12.92f * g);
		b = (b > 0.0031308f) ? (1.055f * pow(b, 1.0f / 2.4f) - 0.055f) : (12.92f * b);

		r = cv::max(0.0f, cv::min(1.0f, r));
		g = cv::max(0.0f, cv::min(1.0f, g));
		b = cv::max(0.0f, cv::min(1.0f, b));
	}

	/// <summary>
	/// Безопасное приведение float значения к uchar с обрезкой
	/// </summary>
	/// <param name="value">Входное значение float</param>
	/// <returns>Значение uchar в диапазоне 0-255</returns>
	static uchar saturate_cast(float value)
	{
		int ivalue = static_cast<int>(value);
		return (ivalue < 0) ? 0 : (ivalue > 255) ? 255 : ivalue;
	}
};