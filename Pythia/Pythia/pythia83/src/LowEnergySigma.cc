// LowEnergySigma.cc is a part of the PYTHIA event generator.
// Copyright (C) 2021 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Function definitions (not found in the header) for the LowEnergySigma class.
// Many cross sections and methods used here are based on UrQMD; see
// https://doi.org/10.1016/S0146-6410(98)00058-1

#include "Pythia8/MathTools.h"
#include "Pythia8/LowEnergySigma.h"

namespace Pythia8 {

//==========================================================================

// Tables for interpolation.

//--------------------------------------------------------------------------

// Important cross sections based on data interpolation.
// http://pdg.lbl.gov/2018/hadronic-xsections/hadron.html

static const LinearInterpolator ppTotalData(1.88, 5.0, {
  272.648, 40.1762, 24.645, 23.5487, 23.8546, 24.0346, 26.2545, 28.4582,
  33.1757, 38.4119, 43.3832, 46.0882, 47.1465, 47.4971, 47.8746, 47.4705,
  47.434, 47.339, 47.3373, 47.1153, 46.8987, 46.5044, 46.1466, 45.8345,
  45.511, 45.2259, 45.0478, 44.7954, 44.4798, 44.2947, 44.0139, 43.6164,
  43.378, 43.0795, 43.1889, 42.8338, 42.5581, 42.343, 42.2069, 42.1636,
  42.1013, 41.8453, 41.9952, 41.5904, 41.4213, 41.3825, 41.2637, 41.1657,
  41.1684, 41.1655, 41.0521, 40.9387, 40.8723, 40.8602, 40.8481, 40.9184,
  40.6, 40.6, 40.6, 40.6, 40.6, 40.6, 40.6, 40.6,
  40.5405, 40.4559, 40.3713, 40.2868, 40.2022, 40.1176, 40, 40.0422,
  40.1045, 40.1122, 40.1199, 40.1276, 40.1353, 40.143, 40.1507, 40.1584,
  40.1661, 40.1739, 40.1816, 40.1893, 40.197, 40.0306, 39.9698, 39.936,
  39.9021, 39.8682, 39.8343, 39.8005, 39.7666, 39.7327, 39.6988, 39.6649,
  39.6311, 39.5964, 39.5534, 39.5103,
});

static const LinearInterpolator pnTotalData(1.88, 5.0, {
  142.64, 114.626, 56.8113, 41.9946, 37.7012, 33.7894, 32.415, 32.9202,
  33.7572, 35.0413, 37.2901, 38.3799, 37.8479, 38.1157, 38.199, 38.7035,
  39.1832, 39.6465, 40.0639, 40.3346, 40.5307, 40.6096, 40.6775, 40.7091,
  40.7408, 40.7521, 40.7552, 40.7582, 40.7682, 40.7889, 40.8095, 40.8335,
  41.1568, 41.4801, 41.8033, 42.1266, 42.4499, 42.7732, 43.0965, 43.0604,
  43.0204, 42.9803, 42.9403, 42.9002, 42.8602, 42.8202, 42.7801, 42.7401,
  42.7, 42.66, 42.62, 42.5799, 42.5399, 42.4994, 42.3413, 42.1832,
  42.0251, 41.867, 41.7089, 41.5508, 41.3928, 41.2347, 41.0766, 40.9185,
  40.7604, 40.6023, 40.4442, 40.2861, 40.128, 39.97, 39.8119, 39.6957,
  39.6811, 39.6665, 39.6519, 39.6373, 39.6227, 39.6081, 39.5935, 39.5789,
  39.5643, 39.5497, 39.5351, 39.5205, 39.5059, 39.4906, 39.475, 39.4593,
  39.4437, 39.428, 39.4124, 39.3966, 39.3803, 39.364, 39.3477, 39.3314,
  39.3151, 39.2962, 39.2454, 39.1946,
});

static const LinearInterpolator NNElasticData(2.1, 5.0, {
  25.8, 25.4777, 25.2635, 24.738, 24.3857, 24.1983, 24.2628, 24.7068,
  23.8799, 22.6501, 22.4714, 22.15, 21.432, 20.6408, 19.8587, 19.7608,
  19.6628, 19.5648, 19.4668, 19.3689, 19.2709, 18.8591, 17.9313, 17.1393,
  16.8528, 16.5662, 16.2797, 15.9932, 15.7066, 15.4201, 15.3088, 14.7717,
  14.2346, 13.6974, 13.448, 13.3659, 13.2837, 13.2015, 13.1193, 13.0372,
  12.955, 12.8728, 12.7906, 12.7085, 12.5667, 12.418, 12.2694, 12.1208,
  11.9762, 11.861, 11.7459, 11.6308, 11.5157, 11.495, 11.4891, 11.4833,
  11.4774, 11.4716, 11.42, 11.35, 11.3132, 11.2642, 11.19721, 11.0205,
  10.9438, 10.8671, 10.7904, 10.6137, 10.537, 10.4603, 10.3422, 9.66659,
  9.60099, 9.57099, 9.8932, 9.8799, 9.7429, 9.6975, 9.622, 9.62194,
  9.56262, 9.50331, 9.944, 9.7, 9.7993, 9.8534, 9.6074, 9.7615,
  9.9155, 9.7173, 9.89788, 9.89375, 9.88963, 9.8855, 9.88138, 9.87726,
  9.87313, 9.91815, 10.1181, 10.3181,
});

static const LinearInterpolator ppiplusElData(1.75, 4.0, {
  14.9884, 13.1613, 17.5848, 17.1952, 14.932, 13.0259, 11.8972, 9.86683,
  9.39563, 8.70279, 9.05966, 7.03901, 7.47933, 8.05845, 7.78532, 7.00014,
  6.97142, 6.94208, 7.03342, 6.6845, 6.39362, 6.29438, 6.1932, 6.09007,
  5.98499, 5.87796, 5.72224, 5.62947, 5.53506, 5.43903, 5.34137, 5.36345,
  5.40184, 5.44085, 5.44156, 5.33228, 5.22132, 5.10868, 4.99435, 4.90088,
});

static const LinearInterpolator ppiminusElData(1.75, 4.0, {
  18.7143, 13.5115, 12.7121, 11.7892, 9.83201, 10.4961, 10.9317, 7.64778,
  9.47316, 8.90622, 8.28179, 7.9987, 7.70873, 7.7451, 7.53732, 6.25046,
  7.2379, 7.10072, 6.96453, 6.88106, 6.52894, 5.81146, 6.04954, 4.95788,
  4.73199, 6.09415, 5.63338, 5.53531, 5.46113, 5.38567, 5.30893, 5.26751,
  5.23021, 5.19231, 5.1538, 4.7455, 4.73382, 4.72196, 4.70993, 5.05733,
});


//--------------------------------------------------------------------------

// Parameterisation of cross sections for pi pi, from Pelàez et al.
// https://doi.org/10.1140/epjc/s10052-019-7509-6

static const LinearInterpolator PelaezpippimData(0.27914, 1.42, {
  0., 9.438, 10.6387, 11.9123, 13.2528, 14.6531, 16.1051, 17.5997,
  19.1267, 20.6754, 22.234, 23.7909, 25.3345, 26.8535, 28.3379,
  29.7789, 31.1701, 32.5071, 33.7888, 35.0171, 36.1979, 37.3411,
  38.461, 39.5772, 40.7148, 41.9052, 43.1878, 44.611, 46.2344, 48.132,
  50.3953, 53.1381, 56.501, 60.6557, 65.8059, 72.1792, 79.9974,
  89.4075, 100.345, 112.319, 124.18, 134.079, 139.906, 140.23, 135.099,
  125.944, 114.732, 103.138, 92.209, 82.4338, 73.9345, 66.6381,
  60.3804, 54.9732, 50.2289, 45.9671, 42.0124, 38.1876, 34.3082,
  30.1856, 25.6749, 20.9198, 18.0155, 24.4665, 23.7801, 23.3174,
  23.0671, 23.0193, 23.1631, 23.4885, 23.9846, 24.6426, 25.3704,
  26.203, 27.2053, 28.3814, 29.7409, 31.295, 33.0523, 35.014, 37.1648,
  39.4605, 41.8131, 44.0776, 46.0493, 47.4853, 48.1553, 47.9114,
  46.7423, 44.7782, 42.2424, 39.3806, 36.4036, 33.4611, 30.6419,
  27.9869, 25.505, 23.1868, 21.0128, 18.9585, 16.9936 }
);

static const LinearInterpolator Pelaezpippi0Data(0.27914, 1.42, {
  0., 0.582125, 0.711825, 0.856636, 1.01562, 1.18828, 1.37453, 1.57464,
  1.78922, 2.01923, 2.26594, 2.53105, 2.81662, 3.1252, 3.45986,
  3.82432, 4.22302, 4.66131, 5.14562, 5.68371, 6.28496, 6.96081,
  7.7252, 8.59529, 9.59224, 10.7424, 12.0785, 13.6419, 15.4846,
  17.6725, 20.2891, 23.4401, 27.2586, 31.9088, 37.5878, 44.5166,
  52.9115, 62.914, 74.4548, 87.0394, 99.5153, 110.029, 116.468,
  117.401, 112.871, 104.312, 93.687, 82.6714, 72.3139, 63.1025,
  55.1613, 48.4257, 42.7537, 37.9837, 33.964, 30.5625, 27.6687, 25.192,
  23.0588, 21.2093, 19.5951, 18.1764, 16.9204, 15.8036, 14.7875,
  13.8567, 13.0046, 12.2253, 11.514, 10.8678, 10.2798, 9.74475,
  9.25798, 8.81533, 8.41296, 8.04736, 7.71532, 7.41386, 7.14027,
  6.89203, 6.66685, 6.46261, 6.27736, 6.10933, 5.95688, 5.81849,
  5.69279, 5.57851, 5.47445, 5.37952, 5.2927, 5.213, 5.13949, 5.07123,
  5.00728, 4.94662, 4.88814, 4.83048, 4.77189, 4.70984, 4.64022 }
);

static const LinearInterpolator Pelaezpi0pi0Data(0.27914, 1.42, {
  0., 10.0054, 11.2925, 12.6399, 14.0407, 15.4871, 16.9701, 18.4796,
  20.0042, 21.5317, 23.0489, 24.542, 25.9971, 27.4003, 28.7384,
  29.9989, 31.1705, 32.2438, 33.2109, 34.0661, 34.8057, 35.4283,
  35.9342, 36.3257, 36.6068, 36.7826, 36.8596, 36.8448, 36.746,
  36.5712, 36.3283, 36.0254, 35.67, 35.2696, 34.8309, 34.3602, 33.8634,
  33.3456, 32.8114, 32.265, 31.7099, 31.1492, 30.5853, 30.0204, 29.456,
  28.8933, 28.3329, 27.7752, 27.2196, 26.6654, 26.1108, 25.5498,
  24.9632, 24.3243, 23.5973, 22.7338, 21.6687, 20.3155, 18.5633,
  16.2832, 13.3786, 10.033, 8.37422, 15.9303, 16.2469, 16.7007,
  17.2866, 18.0008, 18.8389, 19.7985, 20.8717, 22.0535, 23.2558,
  24.5172, 25.906, 27.4297, 29.101, 30.9338, 32.9395, 35.1217, 37.4672,
  39.934, 42.436, 44.8299, 46.9127, 48.4428, 49.1914, 49.0118, 47.8937,
  45.9682, 43.4596, 40.6144, 37.644, 34.6989, 31.8682, 29.1932,
  26.6834, 24.3292, 22.1114, 20.005, 17.979 }
);

static const LinearInterpolator PelaezpimpimData(0.27914, 1.42, {
  0., 1.14955, 1.36568, 1.58422, 1.80352, 2.02229, 2.23956, 2.45456,
  2.66672, 2.87558, 3.08079, 3.28206, 3.4792, 3.67203, 3.86041,
  4.04425, 4.22345, 4.39796, 4.56771, 4.73267, 4.89278, 5.04802,
  5.19837, 5.34378, 5.48425, 5.61975, 5.75026, 5.87577, 5.99624,
  6.11168, 6.22207, 6.32738, 6.42762, 6.52275, 6.61278, 6.69769,
  6.77746, 6.85208, 6.92153, 6.9858, 7.04486, 7.09868, 7.14723,
  7.19046, 7.22833, 7.26076, 7.28766, 7.30894, 7.32447, 7.33409,
  7.33762, 7.33743, 7.33651, 7.33484, 7.33239, 7.32911, 7.32497,
  7.31992, 7.31391, 7.30689, 7.29878, 7.28954, 7.2791, 7.2674, 7.25437,
  7.23997, 7.22413, 7.20679, 7.18972, 7.17776, 7.16691, 7.1557,
  7.14338, 7.12947, 7.11365, 7.09569, 7.07542, 7.05272, 7.0275,
  6.99969, 6.96925, 6.93612, 6.90027, 6.86167, 6.82028, 6.77606,
  6.72895, 6.67888, 6.62576, 6.56948, 6.50989, 6.4468, 6.37995,
  6.30902, 6.23356, 6.153, 6.0665, 5.97291, 5.87046, 5.75634, 5.62556 }
);

// pipi -> f_0(500) cross section, without Clebsh-Gordan coefficients.
static const LinearInterpolator pipiTof0500Data(0.27915, 1., {
  8.15994, 9.53565, 11.0102, 12.5738, 14.2131, 15.9117, 17.6494, 19.4033,
  21.1478, 22.8556, 24.4988, 26.0502, 27.4844, 28.7791, 29.9161, 30.8821,
  31.669, 32.2741, 32.6997, 32.9524, 33.0426, 32.9833, 32.7894, 32.4769,
  32.0621, 31.5607, 30.9881, 30.3583, 29.6841, 28.9768, 28.2464, 27.5015,
  26.749, 25.995, 25.244, 24.4994, 23.7637, 23.0379, 22.3219, 21.6142,
  20.9102, 20.182, 19.3792, 18.4299, 17.2295, 15.6266, 13.4101, 10.3267,
  6.29745, 3.39533, }
);

// Elastic contribution to pipi scattering, below 1.42 GeV.
static const LinearInterpolator pipidWaveData(0.27914, 1.42, {
  0., 1.36571, 1.8036, 2.23967, 2.66687, 3.08096, 3.47941, 3.86064,
  4.2237, 4.56798, 4.89306, 5.19865, 5.48454, 5.75055, 5.99653,
  6.22235, 6.42789, 6.61304, 6.7777, 6.92175, 7.04505, 7.1474, 7.22846,
  7.28776, 7.32452, 7.33762, 7.33651, 7.33237, 7.32495, 7.31388,
  7.29873, 7.27904, 7.2543, 7.22403, 7.18964, 7.16684, 7.1433, 7.11354,
  7.07528, 7.02732, 6.96902, 6.9, 6.81997, 6.72858, 6.62534, 6.50941,
  6.37939, 6.23292, 6.06575, 5.86953, 5.62432 }
);

//--------------------------------------------------------------------------

// Parameterisation of total cross sections for pi K, from Pelàez et al.
// This disregards Clebsch-Gordan coefficients.
// https://doi.org/10.1103/PhysRevD.93.074025

static const LinearInterpolator PelaezpiK12TotData(0.64527, 1.800, {
  12.8347, 13.4543, 14.0733, 14.691, 15.3069, 15.9207, 16.5323,
  17.1416, 17.7488, 18.3545, 18.9594, 19.5647, 20.1719, 20.7829,
  21.4004, 22.0274, 22.668, 23.3271, 24.0107, 24.7262, 25.4827,
  26.2916, 27.1668, 28.1256, 29.1899, 30.387, 31.7519, 33.3291,
  35.1764, 37.3692, 40.0068, 43.2209, 47.1879, 52.1452, 58.411,
  66.4084, 76.6817, 89.8813, 106.652, 127.296, 151.028, 174.909,
  193.406, 200.496, 193.953, 177.122, 155.853, 134.776, 116.185,
  100.692, 88.0982, 77.9397, 69.7307, 63.0521, 57.5679, 53.0177,
  49.2015, 45.9667, 43.1962, 40.7997, 38.7069, 36.8628, 35.224,
  33.7556, 32.4297, 31.2237, 30.1187, 29.0994, 28.1526, 27.1998,
  26.4297, 25.8274, 25.2935, 24.8079, 24.3608, 23.9459, 23.5589,
  23.1967, 22.8569, 22.5375, 22.237, 21.9542, 21.6882, 21.438, 21.2032,
  20.9832, 20.7777, 20.5866, 20.4095, 20.2466, 20.0979, 19.9634,
  19.8434, 19.7382, 19.6481, 19.5735, 19.5149, 19.4729, 19.448,
  19.4409, 19.4525, 19.4833, 19.5344, 19.6065, 19.7007, 19.818,
  19.9595, 20.1262, 20.3194, 20.5405, 20.7909, 21.0722, 21.3862,
  21.7351, 22.1215, 22.5483, 23.0191, 23.5384, 24.1113, 24.7441,
  25.4439, 26.2189, 27.0775, 28.0281, 29.078, 30.2311, 31.4861,
  32.8328, 34.2485, 35.6941, 37.1118, 38.4247, 39.5421, 40.3687,
  40.8195, 40.8346, 40.3919, 39.5111, 38.2492, 36.6887, 34.9232,
  33.0443, 31.1324, 29.2521, 27.4508, 25.7605, 24.2, 22.7783, 21.4973,
  20.3539, 19.3421, 18.4542, 17.6815, 17.0154, 16.4476, 15.9701,
  15.5758, 15.2581, 15.0111, 14.8293, 14.7081, 14.6429, 14.6297,
  14.6644, 14.7431, 14.8616, 15.0158, 15.2007, 15.4114, 15.6419,
  15.8861, 16.1373, 16.3885, 16.6328, 16.8638, 17.0759, 17.2647,
  17.4278, 17.5647, 17.6768, 17.7677, 17.8424, 17.9067, 17.967, 18.029,
  18.098, 18.1777, 18.2704, 18.3766, 18.4956, 18.6245, 18.7591, 18.893,
  19.0179, 19.1233, 19.1971, 19.227, 19.2025, 19.1176, 18.9719, 19.02 }
);

static const LinearInterpolator PelaezpiK32TotData(0.64527, 1.800, {
  7.37595, 9.30047, 11.1782, 12.9339, 14.5161, 15.896, 17.063, 18.0201,
  18.7794, 19.3582, 19.7763, 20.0538, 20.2102, 20.2636, 20.2301,
  20.1239, 19.9575, 19.7412, 19.4842, 19.1941, 18.8771, 18.5387,
  18.1832, 17.8142, 17.4347, 17.0473, 16.6537, 16.2558, 15.8546,
  15.4511, 15.0461, 14.64, 14.2332, 13.8258, 13.4177, 13.0089, 12.599,
  12.1876, 11.774, 11.3574, 10.937, 10.5113, 10.0788, 9.6376, 9.18503,
  8.71768, 8.23082, 7.71761, 7.16748, 6.56219, 7.67088 }
);

static const LinearInterpolator PelaezpiK32ElData(0.64527, 1.800, {
  0.64527, 1.800, 4.0221, 5.08067, 6.11084, 7.06899, 7.92639, 8.66777, 9.28859,
  9.79193, 10.1857, 10.4807, 10.6885, 10.8209, 10.889, 10.903, 10.8717,
  10.8031, 10.7037, 10.5793, 10.4347, 10.2738, 10.1, 9.91604, 9.72414,
  9.52616, 9.32358, 9.11763, 8.90926, 8.69924, 8.48814, 8.27641,
  8.06436, 7.85219, 7.64002, 7.42787, 7.21568, 7.00333, 6.79061,
  6.57724, 6.36287, 6.14703, 5.92916, 5.70857, 5.48438, 5.2555,
  5.02051, 4.77753, 4.52398, 4.25615, 3.96826, 3.6504, 3.28237 }
);

//--------------------------------------------------------------------------

// Definitions of static variables used for SaS/DL diffractive cross sections.

// Type of the two incoming hadrons as function of the process number:
// = 0 : p/n ; = 1 : pi/rho/omega; = 2 : phi.
static constexpr int IHADATABLE[] = { 0, 0, 1, 1, 1, 2, 1, 1, 2};
static constexpr int IHADBTABLE[] = { 0, 0, 0, 0, 0, 0, 1, 2, 2};

// Hadron-Pomeron coupling beta(t) = beta(0) * exp(b*t).
static constexpr double BETA0[] = { 4.658, 2.926, 2.149};
static constexpr double BHAD[]  = {   2.3,   1.4,   1.4};
static constexpr double XPROC[] = { 21.70, 21.70, 13.63, 13.63, 13.63,
  10.01, 8.56, 6.29, 4.62};

// Pomeron trajectory alpha(t) = 1 + epsilon + alpha' * t.
static constexpr double ALPHAPRIME = 0.25;
static constexpr double ALP2       = 2. * ALPHAPRIME;
static constexpr double SZERO      = 1. / ALPHAPRIME;

// Conversion coefficients = 1/(16pi) * (mb <-> GeV^2) * (G_3P)^n,
// with n = 0 elastic, n = 1 single and n = 2 double diffractive.
static constexpr double CONVERTSD = 0.0336;
static constexpr double CONVERTDD = 0.0084;

// Minimum threshold below which no cross sections will be defined.
static constexpr double MMIN  = 0.5;

// Lowest energy for parametrized diffractive cross sections.
static constexpr double ECMMIN = 10.;

// Diffractive mass spectrum starts at m + MMIN0 and has a low-mass
// enhancement, factor cRes, up to around m + mRes0.
static constexpr double MMIN0 = 0.28;
static constexpr double CRES  = 2.0;
static constexpr double MRES0 = 1.062;

// Parameters and coefficients for single diffractive scattering.
static constexpr int ISDTABLE[] = { 0, 0, 1, 1, 1, 2, 3, 4, 5};
static constexpr double CSD[6][8] = {
  { 0.213, 0.0, -0.47, 150., 0.213, 0.0, -0.47, 150., } ,
  { 0.213, 0.0, -0.47, 150., 0.267, 0.0, -0.47, 100., } ,
  { 0.213, 0.0, -0.47, 150., 0.232, 0.0, -0.47, 110., } ,
  { 0.267, 0.0, -0.46,  75., 0.267, 0.0, -0.46,  75., } ,
  { 0.232, 0.0, -0.46,  85., 0.267, 0.0, -0.48, 100., } ,
  { 0.232, 0.0, -0.48, 110., 0.232, 0.0, -0.48, 110., } };

// Parameters and coefficients for double diffractive scattering.
static constexpr int IDDTABLE[] = { 0, 0, 1, 1, 1, 2, 3, 4, 5};
static constexpr double CDD[6][9] = {
  { 3.11, -7.34,  9.71, 0.068, -0.42, 1.31, -1.37,  35.0,  118., } ,
  { 3.11, -7.10,  10.6, 0.073, -0.41, 1.17, -1.41,  31.6,   95., } ,
  { 3.12, -7.43,  9.21, 0.067, -0.44, 1.41, -1.35,  36.5,  132., } ,
  { 3.11, -6.90,  11.4, 0.078, -0.40, 1.05, -1.40,  28.4,   78., } ,
  { 3.11, -7.13,  10.0, 0.071, -0.41, 1.23, -1.34,  33.1,  105., } ,
  { 3.11, -7.39,  8.22, 0.065, -0.44, 1.45, -1.36,  38.1,  148., } };

//==========================================================================

// The LowEnergySigma class.

//--------------------------------------------------------------------------

// Initialize.

void LowEnergySigma::init(NucleonExcitations* nucleonExcitationsPtrIn) {

  // Flag to allow or suppress inelastic processes.
  doInelastic    = flag("Rescattering:inelastic");

  // Mode for calculating total cross sections for pi pi and pi K.
  useSummedResonances = flag("LowEnergyQCD:useSummedResonances");

  // Suppression factors in Additive Quark Model (AQM).
  sEffAQM        = parm("LowEnergyQCD:sEffAQM");
  cEffAQM        = parm("LowEnergyQCD:cEffAQM");
  bEffAQM        = parm("LowEnergyQCD:bEffAQM");

  // Mixing for eta and eta'; specifically fraction of s sbar.
  double theta   = parm("StringFlav:thetaPS");
  double alpha   = (theta + 54.7) * M_PI / 180.;
  fracEtass      = pow2(sin(alpha));
  fracEtaPss     = 1. - fracEtass;

  // Some standard masses.
  mp             = particleDataPtr->m0(2212);
  sp             = mp * mp;
  s4p            = 4. * sp;
  mpi            = particleDataPtr->m0(211);
  mK             = particleDataPtr->m0(321);

  // Store pointer
  nucleonExcitationsPtr = nucleonExcitationsPtrIn;

}

//--------------------------------------------------------------------------

// Get the total cross section for the specified collision

double LowEnergySigma::sigmaTotal(int idAIn, int idBIn, double eCMIn,
  double mAIn, double mBIn) {

  // Energy cannot be less than the hadron masses.
  if (eCMIn <= mAIn + mBIn) {
    infoPtr->errorMsg("Error in LowEnergySigma::sigmaTotal: nominal masses "
      "are higher than total energy", "for " + to_string(idAIn) + " "
      + to_string(idBIn) + " @ " + to_string(eCMIn));
    return 0.;
  }

  // For K0S/K0L, take average of K0 and K0bar.
  if (idAIn == 310 || idAIn == 130)
    return 0.5 * (sigmaTotal( 311, idBIn, eCMIn, mAIn, mBIn)
                + sigmaTotal(-311, idBIn, eCMIn, mAIn, mBIn));
  if (idBIn == 310 || idBIn == 130)
    return 0.5 * (sigmaTotal(idAIn,  311, eCMIn, mAIn, mBIn)
                + sigmaTotal(idAIn, -311, eCMIn, mAIn, mBIn));

  // Fix particle ordering.
  setConfig(idAIn, idBIn, eCMIn, mAIn, mBIn);

  // Special handling for pi pi and pi K cross sections
  if (!useSummedResonances && eCM < 1.42) {
    if (idA == 211 && idB == -211)
      return PelaezpippimData(eCM);
    else if (idA == 211 && idB == 111)
      return Pelaezpippi0Data(eCM);
    else if (idA == 111 && idB == 111)
      return Pelaezpi0pi0Data(eCM);
    else if (idA == 211 && idB == 211)
      return PelaezpimpimData(eCM);
  }
  if (!useSummedResonances && eCM < 1.8) {
    if ((idA == 321 && idB == 211) || (idA == 311 && idB == -211))
      return PelaezpiK32TotData(eCM);
    else if ((idA == 321 || idA == 311) && (abs(idB) == 211 || idB == 111))
      return PelaezpiK12TotData(eCM) * (idB == 111 ? 1./3. : 2./3.);
  }

  // Calculate total.
  calcTot();

  return sigTot;
}

//--------------------------------------------------------------------------

// Gets the partial cross section for the specified process.

double LowEnergySigma::sigmaPartial(int idAIn, int idBIn, double eCMIn,
  double mAIn, double mBIn, int proc) {

  // Energy cannot be less than the hadron masses.
  if (eCMIn <= mAIn + mBIn) {
    infoPtr->errorMsg("Error in LowEnergySigma::sigmaPartial: nominal masses "
      "are higher than total energy", "for " + to_string(idAIn) + " "
      + to_string(idBIn) + " @ " + to_string(eCMIn));
    return 0.;
  }

  // For K0S/K0L, take average of K0 and K0bar.
  if (idAIn == 310 || idAIn == 130)
    return 0.5 * (sigmaPartial( 311, idBIn, eCMIn, mAIn, mBIn, proc)
                + sigmaPartial(-311, idBIn, eCMIn, mAIn, mBIn, proc));
  if (idBIn == 310 || idBIn == 130)
    return 0.5 * (sigmaPartial(idAIn,  311, eCMIn, mAIn, mBIn, proc)
                + sigmaPartial(idAIn, -311, eCMIn, mAIn, mBIn, proc));

  // Total cross section.
  if (proc == 0) return sigmaTotal(idAIn, idBIn, eCMIn, mAIn, mBIn);

  // Get all partial cross sections.
  vector<int> procs;
  vector<double> sigmas;
  if (!sigmaPartial(idAIn, idBIn, eCMIn, mAIn, mBIn, procs, sigmas))
    return 0.;

  if (proc == 9)
    return sigResTot;

  // Return partial cross section.
  for (size_t i = 0; i < procs.size(); ++i)
    if (procs[i] == proc) return sigmas[i];

  return 0.;
}

//--------------------------------------------------------------------------

// Gets all partial cross sections for the specified collision.
// Returns whether any processes have positive cross sections.

bool LowEnergySigma::sigmaPartial(int idAIn, int idBIn, double eCMIn,
  double mAIn, double mBIn, vector<int>& procsOut, vector<double>& sigmasOut) {

  // No cross sections below threshold.
  if (eCMIn <= mAIn + mBIn) return false;

  // If K_S/K_L + X then take the average K0 and K0bar.
  if (idAIn == 130 || idAIn == 310) {
    vector<int> procsK, procsKbar;
    vector<double> sigmasK, sigmasKbar;
    if ( !sigmaPartial( 311, idBIn, eCMIn, mAIn, mBIn, procsK, sigmasK)
      || !sigmaPartial(-311, idBIn, eCMIn, mAIn, mBIn, procsKbar, sigmasKbar))
      return false;
    for (size_t i = 0; i < procsK.size(); ++i) {
      procsOut.push_back(procsK[i]);
      sigmasOut.push_back(0.5 * sigmasK[i]);
    }
    for (size_t iKbar = 0; iKbar < procsKbar.size(); ++iKbar) {
      auto iter = std::find(procsOut.begin(), procsOut.end(),
        procsKbar[iKbar]);
      if (iter == procsOut.end()) {
        procsOut.push_back(procsKbar[iKbar]);
        sigmasOut.push_back(0.5 * sigmasKbar[iKbar]);
      } else {
        int iK = std::distance(procsOut.begin(), iter);
        sigmasOut[iK] += 0.5 * sigmasKbar[iKbar];
      }
    }
    return true;
  }

  // If X + K_S/K_L then take the average K0 and K0bar.
  if (idBIn == 130 || idBIn == 310) {
    vector<int> procsK, procsKbar;
    vector<double> sigmasK, sigmasKbar;
    if ( !sigmaPartial(idAIn,  311, eCMIn, mAIn, mBIn, procsK, sigmasK)
      || !sigmaPartial(idAIn, -311, eCMIn, mAIn, mBIn, procsKbar, sigmasKbar))
      return false;
    for (size_t i = 0; i < procsK.size(); ++i) {
      procsOut.push_back(procsK[i]);
      sigmasOut.push_back(0.5 * sigmasK[i]);
    }
    for (size_t iKbar = 0; iKbar < procsKbar.size(); ++iKbar) {
      auto iter = std::find(procsOut.begin(), procsOut.end(),
        procsKbar[iKbar]);
      if (iter == procsOut.end()) {
        procsOut.push_back(procsKbar[iKbar]);
        sigmasOut.push_back(0.5 * sigmasKbar[iKbar]);
      } else {
        int iK = std::distance(procsOut.begin(), iter);
        sigmasOut[iK] += 0.5 * sigmasKbar[iKbar];
      }
    }
    return true;
  }

  // Store current configuration.
  setConfig(idAIn, idBIn, eCMIn, mAIn, mBIn);

  // Calculate total, annihilation and resonant cross sections (if needed).
  calcTot();

  // Abort early if total cross section vanishes.
  if (sigTot == 0.)
    return false;

  // If inelastic is off, return only elastic cros ssection.
  if (!doInelastic) {
    procsOut.push_back(2);
    sigmasOut.push_back(sigTot);
    return true;
  }

  // Calculate diffractive cross sections.
  calcDiff();

  // Calculate elastic cross sections.
  calcEla();

  // Calculate excitation cross sections for NN.
  calcEx();

  // Calculate nondiffractive as difference between total and the others.
  sigND = sigTot - sigEl - sigXB - sigAX - sigXX - sigEx - sigAnn - sigResTot;

  // Give warning if sigmaND is very negative.
  if (sigND < -0.1) {
    infoPtr->errorMsg("Warning in LowEnergySigma::sigmaPartial: sum of "
      "partial sigmas is larger than total sigma", " for " + to_string(idA)
      + " + " + to_string(idB) + " @ " + to_string(eCM) + " GeV");
  }

  // Rescale pi pi and pi K cross sections to match Pelàez total cross
  // section.
  if (!useSummedResonances) {
    // Rescale for pi pi below 1.42 GeV, and for pi K below 1.8 GeV
    if ( (eCM < 1.42 && (abs(idA) == 211 || idA == 111)
                     && (abs(idB) == 211 || idB == 111))
      || (eCM < 1.8 && (idA == 321 || idA == 311)
                    && (abs(idB) == 211 || idB == 111))) {

      // Get cross section from parameterization.
      double sigPelaez;
      if (idA == 211 && idB == -211)
        sigPelaez = PelaezpippimData(eCM);
      else if (idA == 211 && idB == 111)
        sigPelaez = Pelaezpippi0Data(eCM);
      else if (idA == 111 && idB == 111)
        sigPelaez = Pelaezpi0pi0Data(eCM);
      else if (idA == 211 && idB == 211)
        sigPelaez = PelaezpimpimData(eCM);
      else if ( (idA == 321 && idB == 211) || (idA == 311 && idB == -211) )
        sigPelaez = PelaezpiK32TotData(eCM);
      else if ( (idA == 321 && idB == -211) || (idA == 311 && idB == 211) )
        sigPelaez = 2./3. * PelaezpiK12TotData(eCM);
      else if ((idA == 321 || idA == 311) && idB == 111)
        sigPelaez = 1./3. * PelaezpiK12TotData(eCM);
      else
        sigPelaez = sigTot;

      double scale = sigPelaez / sigTot;
      sigTot *= scale;
      sigEl  *= scale;
      sigXB  *= scale;
      sigAX  *= scale;
      sigXX  *= scale;
      sigND  *= scale;
      sigResTot *= scale;
      for (auto& res : sigRes)
        res.second *= scale;
    }
  }

  // Write results to output arrays.
  bool gotAny = false;
  procsOut.clear();
  sigmasOut.clear();

  if (sigND > TINYSIGMA) {
    procsOut.push_back(1); sigmasOut.push_back(sigND); gotAny = true; }
  if (sigEl > TINYSIGMA) {
    procsOut.push_back(2); sigmasOut.push_back(sigEl); gotAny = true; }
  if (sigXB > TINYSIGMA) {
    procsOut.push_back(3); sigmasOut.push_back(sigXB); gotAny = true; }
  if (sigAX > TINYSIGMA) {
    procsOut.push_back(4); sigmasOut.push_back(sigAX); gotAny = true; }
  if (sigXX > TINYSIGMA) {
    procsOut.push_back(5); sigmasOut.push_back(sigXX); gotAny = true; }
  if (sigEx > TINYSIGMA) {
    procsOut.push_back(7); sigmasOut.push_back(sigEx); gotAny = true; }
  if (sigAnn > TINYSIGMA) {
    procsOut.push_back(8); sigmasOut.push_back(sigAnn); gotAny = true; }

  for (auto resonance : sigRes) {
    procsOut.push_back(resonance.first);
    sigmasOut.push_back(resonance.second);
    gotAny = true;
  }

  return gotAny;

}

//--------------------------------------------------------------------------

// Picks a process randomly according to their partial cross sections.

int LowEnergySigma::pickProcess(int idAIn, int idBIn, double eCMIn,
  double mAIn, double mBIn) {

  vector<int> processes;
  vector<double> sigmas;
  if (!sigmaPartial(idAIn, idBIn, eCMIn, mAIn, mBIn, processes, sigmas))
    return 0;
  return processes[rndmPtr->pick(sigmas)];
}

//--------------------------------------------------------------------------

// Picks a resonance according to their partial cross sections.

int LowEnergySigma::pickResonance(int idAIn, int idBIn, double eCMIn) {

  // Set canonical ordering.
  setConfig(idAIn, idBIn, eCMIn,
    particleDataPtr->m0(idAIn), particleDataPtr->m0(idBIn));

  // Fail if no resonances exist.
  if (!hasExplicitResonances()) return 0;

  // Calculate cross section for each resonance.
  calcRes();

  // Return zero if no resonances are available.
  if (sigResTot == 0.)
    return 0;

  vector<int> ids;
  vector<double> sigmas;
  for (auto resonance : sigRes) {
    if (resonance.second != 0.) {
      ids.push_back(resonance.first);
      sigmas.push_back(resonance.second);
    }
  }

  // Pick resonance at random.
  int resPick = ids[rndmPtr->pick(sigmas)];
  // Change to antiparticle if the canonical ordering changed signs.
  return (didFlipSign) ? particleDataPtr->antiId(resPick) : resPick;

}

//--------------------------------------------------------------------------

// Calculate total cross section. This will also calculate annihilation and
// resonance cross sections, if necessary.

void LowEnergySigma::calcTot() {

  // pipi special case.
  if ((idA == 211 || idA == 111) && (abs(idB) == 211 || idB == 111)) {

    // Resonances are available for all cases except pi+pi+
    if (!(idA == 211 && idB == 211))
      calcRes();

    if (eCM < 1.42) {
      // Below threshold, use resonances + d-wave
      double dWaveFactor = (idA == 211 && idB == -211) ? 1./6.
                         : (idA == 211 && idB ==  111) ? 1./2.
                         : (idA == 111 && idB ==  111) ? 2./3.
                         : 1.;
      sigTot = sigResTot + dWaveFactor * pipidWaveData(eCM);
    }
    else {
      // Above threshold, use parameterisation based on Pelaez et al.
      double sCM = eCM * eCM;
      double h = pow2(2. * M_PI) * GEVSQINV2MB
               / (eCM * sqrt(sCM - 4. * mpi * mpi));
      double sCM1 = pow(sCM, 0.53), sCM2 = pow(sCM, 0.06);

      if (idA == 211 && idB == -211)
        sigTot = h * (0.83 * sCM + 1.01 * sCM1 + 0.013 * sCM2);
      else if (idA == 211 && idB == 111)
        sigTot = h * (0.83 * sCM + 0.267 * sCM1 - 0.0267 * sCM2);
      else if (idA == 111 && idB == 111)
        sigTot = h * (0.83 * sCM + 0.267 * sCM1 + 0.053 * sCM2);
      else // if (idA == 211 && idB == 211)
        sigTot = h * (0.83 * sCM - 0.473 * sCM1 + 0.013 * sCM2);
    }
  }

  // piK special case. idA must be K+ or K0bar due to canonical ordering.
  else if ((idA == 321 || idA == 311) && (abs(idB) == 211 || idB == 111)) {

    // 2 * isospin. Resonances are available if |I| = 1/2.
    int isoType = abs( (idA == 321 ? 1 : -1)
                     + (idB == 211 ? 2 : idB == -211 ? -2 : 0) );

    if (isoType == 1)
      calcRes();

    double cg2 = isoType == 3 ? 1. : (idB == 111) ? 1./3. : 2./3.;

    if (eCM < 1.8) {
      sigTot = (isoType == 1) ? sigResTot : PelaezpiK32TotData(eCM);
    }
    else {
      double sCM = eCM * eCM;
      double pCoefficient = (isoType == 1 ? 12.3189 : -5.76786);
      sigTot = cg2 * (10.3548 * sCM + pCoefficient * pow(sCM, 0.53))
             / sqrt((sCM - pow2(mpi + mK)) * (sCM - pow2(mpi - mK)));
    }
  }

  // Npi special case.
  else if ((idA == 2212 || idA == 2112) && (abs(idB) == 211 || idB == 111)) {
    calcRes();
    if (eCM < meltpoint(idA, idB))
      sigTot = sigResTot;
    else {
      // Npi-.
      if (idB == -211)
        sigTot = HPR1R2(18.75, 9.56, 1.767, mA, mB, eCM * eCM);
      // Npi+, Npi0.
      else
        sigTot = HPR1R2(18.75, 9.56, -1.767, mA, mB, eCM * eCM);
    }
  }

  // NKbar special case.
  else if ((idA == 2212 || idA == 2112) && (idB == -321 || idB == -311)) {

    calcRes();

    if (eCM < 2.16) {

      sigTot = sigResTot;

      // Add non-diffractive contribution based on description by UrQMD.
      static constexpr double e0 = 1.433;
      if (eCM < 1.4738188)
        sigTot += 5.93763355 / pow2(eCM - 1.251377);
      else if (eCM < 1.485215)
        sigTot += -1.296457765e7 * pow4(eCM - e0)
                + 2.160975431e4 * pow2(eCM - e0) + 120.;
      else if (eCM < 1.977)
        sigTot += 3. + 1.0777e6 * exp(-6.4463 * eCM)
                - 10. * exp(-pow2(eCM - 1.644) / 0.004)
                + 10. * exp(-pow2(eCM - 1.977) / 0.004);
      else
        sigTot += 1.0777e6 * exp(-6.44463 * eCM) + 12.5;
    }

    // Use HPR1R2 parametrisations above meltpoint.
    // pK-, pKbar0.
    else if (idA == 2212 && (idB == -321 || idB == -311))
      sigTot = HPR1R2(16.36, 4.29, 3.408, mA, mB, eCM * eCM);
    // nK-, nKbar0.
    else if (idA == 2112 && (idB == -321 || idB == -311))
      sigTot = HPR1R2(16.31, 3.70, 1.826, mA, mB, eCM * eCM);
  }

  // NK+ and NK0: use a very simple fit to data.
  else if ((idA == 2212 || idA == 2112) && (idB == 321 || idB == 311)) {
    double t = clamp((eCM - 1.65) / (1.9 - 1.65), 0, 1);
    sigTot = 12.5 * (1 - t) + 17.5 * t;
  }

  // pp/nn: use parametrisation.
  else if ((idA == 2212 && idB == 2212) || (idA == 2112 && idB == 2112))
    sigTot = (eCM < 5.) ? ppTotalData(eCM)
                        : HPR1R2(34.41, 13.07, -7.394, mA, mB, eCM * eCM);

  // pn: use parametrisation.
  else if (idA == 2212 && idB == 2112)
    sigTot = (eCM < 5.) ? pnTotalData(eCM)
                        : HPR1R2(34.71, 12.52, -6.66, mA, mB, eCM * eCM);

  // Other BB: use AQM.
  else if (collType == 1)
    sigTot = totalAQM();

  // BBbar: use generic parametrisation.
  else if (collType == 2) {
    // Calculate effective energy, i.e. eCM of protons with the same momenta.
    double sBB = pow2(eCM);
    double sNN = s4p + (sBB - pow2(mA + mB)) * (sBB - pow2(mA - mB)) / sBB;
    double pLab = sqrt(sNN * (sNN - s4p)) / (2. * mp);

    // Calculate ppbar cross section based on UrQMD parameterization.
    double sigmaTotNN
      = (pLab < 0.3) ? 271.6 * exp(-1.1 * pLab * pLab)
      : (pLab < 6.5) ? 75.0 + 43.1 / pLab + 2.6 / pow2(pLab) - 3.9 * pLab
      :                HPR1R2(34.41, 13.07, 7.394, mA, mB, sNN);

    // Scale pp cross section scaled by AQM factor.
    double aqmFactor = factorAQM();
    sigTot = sigmaTotNN * aqmFactor;

    // Calculate annihilation sigma. If quarks can annihilate, store it;
    // otherwise, subtract it from the total cross section.
    double sigmaAnnNN;
    if (sNN < pow2(2.1)) {
      calcEla();
      sigmaAnnNN = sigTot - sigEl;
    }
    else {
      static constexpr double sigma0 = 120., A = 0.050, B = 0.6;
      sigmaAnnNN = sigma0 * s4p / sNN
                 * ((A * A * s4p) / (pow2(sNN - s4p) + A * A * s4p) + B);
    }

    // Check that particles can annihilate. Overrun not possible.
    vector<int> countA(5), countB(5);
    for (int quarksA = ( idA / 10) % 1000; quarksA > 0; quarksA /= 10)
      if (quarksA % 10 > 1 && quarksA % 10 < 6) countA[quarksA % 10 - 1] += 1;
    for (int quarksB = (-idB / 10) % 1000; quarksB > 0; quarksB /= 10)
      if (quarksB % 10 > 1 && quarksB % 10 < 6) countB[quarksB % 10 - 1] += 1;
    int nMutual = 0;
    for (int i = 0; i < 5; ++i) nMutual += min(countA[i], countB[i]);

    // At least one mutual quark pair is needed for baryon number annihilation.
    if (nMutual >= 1)
      sigAnn = sigmaAnnNN * aqmFactor;
    else
      sigTot -= sigmaAnnNN * aqmFactor;
  }

  // If explicit resonances are available.
  else if (hasExplicitResonances()) {
    // Calculate resonances.
    calcRes();
    if (eCM < meltpoint(idA, idB))
      sigTot = sigResTot + elasticAQM();
    else
      sigTot = totalAQM();
  }

  // Last resort: use AQM.
  else
    sigTot = totalAQM();

}

//--------------------------------------------------------------------------

// Calculate all resonance cross sections.

void LowEnergySigma::calcRes() {
  for (auto idR : hadronWidthsPtr->possibleResonances(idA, idB)) {
    double sigResNow = calcRes(idR);
    if (sigResNow > 0.) {
      if (didFlipSign) idR = particleDataPtr->antiId(idR);
      sigResTot += sigResNow;
      sigRes.push_back(make_pair(idR, sigResNow));
    }
  }
}

//--------------------------------------------------------------------------

// Calculate and return cross sections for forming the specified resonance.

double LowEnergySigma::calcRes(int idR) const {

  // Do special case for f0(500).
  if (idR == 9000221) {
    if ((idA == 211 && idB == -211) || (idA == 111 && idB == 111))
      return pipiTof0500Data(eCM);
    else
      return 0.;
  }

  double gammaR = hadronWidthsPtr->width(idR, eCM);
  double brR    = hadronWidthsPtr->br(idR, idA, idB, eCM);

  if (gammaR == 0. || brR == 0.)
    return 0.;

    // Find particle entries
  auto entryR = particleDataPtr->findParticle(idR);
  auto entryA = particleDataPtr->findParticle(idA);
  auto entryB = particleDataPtr->findParticle(idB);

  if (entryR == nullptr || entryA == nullptr || entryB == nullptr) {
    infoPtr->errorMsg("Error in HadronWidths::sigmaResonant: particle does "
      "not exist", to_string(idR) + " --> " + to_string(idA) + " "
      + to_string(idB));
    return 0.;
   }

  // Calculate the resonance sigma
  double s = pow2(eCM), mA0 = entryA->m0(), mB0 = entryB->m0();
  double pCMS2 = 1 / (4 * s) * (s - pow2(mA0 + mB0)) * (s - pow2(mA0 - mB0));

  return GEVSQINV2MB * M_PI / pCMS2
    * entryR->spinType() / (entryA->spinType() * entryB->spinType())
    * brR * pow2(gammaR) / (pow2(entryR->m0() - eCM) + 0.25 * pow2(gammaR));
}

//--------------------------------------------------------------------------

// Calculate elastic cross section. This may be dependent on the sigTot.

void LowEnergySigma::calcEla() {

  double sCM = eCM * eCM;

  // Special case for pipi.
  if ((abs(idA) == 211 || idA == 111) && (abs(idB) == 211 || idB == 111)) {
    if (eCM < 1.42) {
      double dWaveFactor = (idA == 211 && idB == -211) ? 1./6.
                         : (idA == 211 && idB ==  111) ? 1./2.
                         : (idA == 111 && idB ==  111) ? 2./3.
                         : 1.;
      sigEl = dWaveFactor * pipidWaveData(eCM);
    }
    else
      sigEl = 4.;
  }

  // Special case for K pi.
  else if ((idA == 321 || idA == 311) && (abs(idB) == 211 || idB == 111)) {
    // For isospin = 1/2, no direct elastic scattering at low energies.
    // For isospin = 3/2 (K+pi+ or K0pi-), use parameterization.
    if (eCM <= 1.8 && ((idA == 321 && idB == 211)
                    || (idA == 311 && idB == -211)))
      sigEl = PelaezpiK32ElData(eCM);
    // Above threshold, both cases can be approximated by a constant.
    else if (eCM > 1.8)
      sigEl = 1.5;
  }

  // Special cases for Npi.
  else if ((idA == 2212 || idA == 2112) && (abs(idB) == 211 || idB == 111)) {
    // Below meltpoint, resonance scattering dominates.
    if (eCM < meltpoint(idA, idB))
      sigEl = 0.;
    else if (eCM < 4.0) {
      // Data exists for ppi+ and for ppi-.
      // For general Npi, pick the case that has the corresponding isospin.
      double sigData
        = ((idA == 2212 && idB == 211) || (idA == 2112 && idB == -211))
        ? ppiplusElData(eCM) : ppiminusElData(eCM);

      double sigResEla = 0.;
      for (auto resonance : sigRes)
        sigResEla += resonance.second
                   * hadronWidthsPtr->br(resonance.first, idA, idB, eCM);

      sigEl = clamp(sigData - sigResEla, 0., sigTot - sigResTot);
    } else {
      double pLab = sqrt((sCM - pow2(mA + mB)) * (sCM - pow2(mA - mB)))
        / (2. * mA);
      sigEl = HERAFit(0., 11.4, -0.4, 0.079, 0., pLab);
    }
  }

  // Special case for NK- and NKbar0.
  else if ((idA == 2212 || idA == 2112) && (idB == -321 || idB == -311)) {
    // Use ad hoc parameterization from UrQMD.
    static constexpr double e0 = 1.433;
    if (eCM < 1.67)
      sigEl = 1.93763355 / pow2(eCM - 1.251377);
    else if (eCM < 1.485215)
      sigEl = -1.296457765e7 * pow4(eCM - e0)
              + 2.160975431e4 * pow2(eCM - e0) + 120.;
    else if (eCM < 1.825)
      sigEl = 1.1777e6 * exp(-6.4463 * eCM)
              - 12. * exp(-pow2(eCM - 1.646) / 0.004)
              + 10. * exp(-pow2(eCM - 1.937) / 0.004);
    else
      sigEl = 5.0 + 5.5777e5 * exp(-6.44463 * eCM);
  }

  // Special case for NK+ and NK0.
  else if ((idA == 2212 || idA == 2112) && (idB == 321 || idB == 311)) {
    double t = clamp((eCM - 1.7) / (2.5 - 1.7), 0., 1.);
    sigEl = 12.5 * (1 - t) + 4.0 * t;
  }

  // Special case for pp/nn/pn
  else if ((idA == 2112 || idA == 2212) && (idB == 2112 || idB == 2212)) {
    // Below 2.1 GeV, only elastic scattering can occur.
    // Below 5.0 GeV, fit to data.
    // Above 5.0 GeV, use HERA fit.
    if (eCM < 2.1)
      sigEl = sigTot;
    else if (eCM < 5.0)
      sigEl = NNElasticData(eCM);
    else {
      double pLab = sqrt((sCM - pow2(mA + mB)) * (sCM - pow2(mA - mB)))
        / (2. * mA);
      sigEl = HERAFit(11.9, 26.9, -1.21, 0.169, -1.85, pLab);
    }
  }

  // Other BB.
  else if (collType == 1)
    sigEl = eCM < mA + mB + 2. * mpi ? totalAQM() : elasticAQM();

  // BBbar.
  else if (collType == 2) {
    double sNN = s4p + (sCM - pow2(mA + mB)) * (sCM - pow2(mA - mB)) / sCM;
    double pLab = sqrt(sNN * (sNN - s4p)) / (2. * mp);

    // Get elastic cross section for ppbar, based on UrQMD parameterization.
    double sigmaElNN
      = (pLab < 0.3) ? 78.6
      : (pLab < 5.)  ? 31.6 + 18.3 / pLab - 1.1 / pow2(pLab) - 3.8 * pLab
                    : HERAFit(10.2, 52.7, -1.16, 0.125, -1.28, pLab);

    // Scale by AQM factor.
    sigEl = sigmaElNN * factorAQM();
  }

  // For mesons at low energy with no resonances, all interactions are elastic.
  else if ((eCM < mA + mB + 2. * mpi) && !hasExplicitResonances())
    sigEl = totalAQM();

  // For other baryon+meson/meson+meson, use AQM.
  else
    sigEl =  elasticAQM();

}

//--------------------------------------------------------------------------

void LowEnergySigma::calcDiff() {

  int idANow = idA, idBNow = idB;
  double eCMNow = eCM, mANow = mA, mBNow = mB;
  bool withEnhancement = !hasExcitation(idA, idB);

  // Real variables. Reset some to zero or unity to begin with.
  double sCM, bA, bB, scaleA, scaleB, scaleC, mMinXBsave, mMinAXsave,
    mResXBsave, mResAXsave, sum1, sum2, sum3, sum4, sMinXB, sMaxXB, sResXB,
    sRMavgXB, sRMlogXB, BcorrXB, sMinAX, sMaxAX, sResAX, sRMavgAX, sRMlogAX,
    BcorrAX, y0min, sLog, Delta0, sMaxXX, sLogUp, sLogDn, BcorrXX;
  sCM = bA = bB = sum1 = sum2 = sum3 = sum4 = 0.;
  scaleA = scaleB = scaleC = 1.;
  double eCMsave = eCMNow;

  // Quark content, stripped of spin and radial and orbital excitations.
  int  idqA     = (abs(idANow) / 10) % 1000;
  int  idqB     = (abs(idBNow) / 10) % 1000;
  bool sameSign = (idANow > 0 && idBNow > 0) || (idANow < 0 && idBNow < 0);

  // Order flavour of incoming hadrons: idAbsA < idAbsB (restore later).
  bool swapped = false;
  if (idqA > idqB) {
    swap( idANow , idBNow );
    swap( idqA, idqB);
    swap( mANow  , mBNow  );
    swapped = true;
  }

  // Check that (well) above the mass threshold.
  // For pseudoscalar mesons use the corresponding vector meson masses.
  if (abs(idANow) < 400 && abs(idANow) % 10 == 1)
    mANow  = particleDataPtr->m0(abs(idANow) + 2);
  if (idANow == 130 || idANow == 310) mANow = particleDataPtr->m0(313);
  if (abs(idBNow) < 400 && abs(idBNow) % 10 == 1)
    mBNow  = particleDataPtr->m0(abs(idBNow) + 2);
  if (idBNow == 130 || idBNow == 310) mBNow = particleDataPtr->m0(313);
  if (eCMNow < mANow + mBNow + MMIN) return;

  // Convert all baryons to protons, with appropriate scale factor,
  // and all mesons to pi0 or phi0, again with scale factor.
  // Exception: heavy hadrons turned to protons, to go for heavier object.
  int idqAtmp = idqA;
  bool heavyA = false;
  if (idqAtmp == 11 || idqAtmp == 22) {
    idqA = 11;
    if (idANow == 221) scaleA = 1. - fracEtass + sEffAQM * fracEtass;
  } else if (idqAtmp == 33) {
    if (idANow == 331) scaleA = fracEtaPss + (1. - fracEtaPss) / sEffAQM;
  } else {
    int nq[10] = {};
    ++nq[idqA%10];
    ++nq[(idqA/10)%10];
    ++nq[(idqA/100)%10];
    heavyA = (nq[4] > 0) || (nq[5] > 0);
    idqA = (idqAtmp < 100 && !heavyA) ? 11 : 221;
    double nqA = (idqAtmp < 100 && !heavyA) ? 2. : 3.;
    scaleA = (nq[1] + nq[2] + sEffAQM * nq[3] + cEffAQM * nq[4]
      + bEffAQM * nq[5]) / nqA;
  }
  int idqBtmp = idqB;
  bool heavyB = false;
  if (idqBtmp == 11 || idqBtmp == 22) {
    idqB = 11;
    if (idBNow == 221) scaleB = 1. - fracEtass + sEffAQM * fracEtass;
  } else if (idqBtmp == 33) {
    if (idBNow == 331) scaleB = fracEtaPss + (1. - fracEtaPss) / sEffAQM;
  } else {
    int nq[10] = {};
    ++nq[idqB%10];
    ++nq[(idqB/10)%10];
    ++nq[(idqB/100)%10];
    heavyB = (nq[4] > 0) || (nq[5] > 0);
    idqB = (idqBtmp < 100 && !heavyB) ? 11 : 221;
    double nqB = (idqBtmp < 100 && !heavyB) ? 2. : 3.;
    scaleB = (nq[1] + nq[2] + sEffAQM * nq[3] + cEffAQM * nq[4]
      + bEffAQM * nq[5]) / nqB;
  }
  scaleC = scaleA * scaleB;

  // Find process number.
  int iProc               = -1;
  if (idqA > 100) {
    iProc                 = (sameSign) ? 0 : 1;
  } else if (idqB > 100) {
    iProc                 = (sameSign) ? 2 : 3;
    if (idqA == 11) iProc = 4;
    if (idqA == 33) iProc = 5;
  } else if (idqA > 10) {
    iProc                 = 6;
    if (idqB == 33) iProc = 7;
    if (idqA == 33) iProc = 8;
  }
  if (iProc == -1) return;

  // Collision energy, rescaled to same p_CM as for proton if heavy flavour.
  if (!heavyA && !heavyB) {
    sCM = eCMNow * eCMNow;
  } else {
    double sAB = eCMNow * eCMNow;
    sCM = s4p + (sAB - pow2(mANow + mBNow))
              * (sAB - pow2(mANow - mBNow)) / sAB;
    eCMNow = sqrt(sCM);
    if (heavyA) mANow = mp;
    if (heavyB) mBNow = mp;
    if (eCMNow < mANow + mBNow + MMIN) return;
    eCMsave = eCMNow;
  }

  // Slope of hadron form factors.
  int iHadA   = IHADATABLE[iProc];
  int iHadB   = IHADBTABLE[iProc];
  bA          = BHAD[iHadA];
  bB          = BHAD[iHadB];

  // Smooth interpolation of diffractive cross sections at low energies.
  bool lowE   = (eCMNow < ECMMIN);
  if (lowE) {
    eCMNow       = ECMMIN;
    sCM       = eCMNow * eCMNow;
  }

  // Lookup coefficients for single and double diffraction.
  int iSD     = ISDTABLE[iProc];
  int iDD     = IDDTABLE[iProc];

  // Single diffractive scattering A + B -> X + B cross section.
  mMinXBsave  = mANow + MMIN0;
  sMinXB      = pow2(mMinXBsave);
  sMaxXB      = CSD[iSD][0] * sCM + CSD[iSD][1];
  sum1        = log( (2.*bB + ALP2 * log(sCM/sMinXB))
              / (2.*bB + ALP2 * log(sCM/sMaxXB)) ) / ALP2;
  if (withEnhancement) {
    mResXBsave = mANow + MRES0;
    sResXB    = pow2(mResXBsave);
    sRMavgXB  = mResXBsave * mMinXBsave;
    sRMlogXB  = log1p(sResXB/sMinXB);
    BcorrXB   = CSD[iSD][2] + CSD[iSD][3] / sCM;
    sum2      = CRES * sRMlogXB / (2.*bB + ALP2 * log(sCM/sRMavgXB) + BcorrXB);
  }
  if (lowE) {
    double ratio = max( 0., eCMsave - mMinXBsave - mBNow)
                 / (ECMMIN  - mMinXBsave - mBNow);
    double ratio03 = pow(ratio, 0.3);
    double ratio06 = pow2(ratio03);
    sum1     *= ratio06;
    sum2     *= ratio03;
  }
  sigXB       = CONVERTSD * scaleC * XPROC[iProc] * BETA0[iHadB]
              * max( 0., sum1 + sum2);

  // Single diffractive scattering A + B -> A + X cross section.
  mMinAXsave  = mBNow + MMIN0;
  sMinAX      = pow2(mMinAXsave);
  sMaxAX      = CSD[iSD][4] * sCM + CSD[iSD][5];
  sum1        = log( (2.*bA + ALP2 * log(sCM/sMinAX))
              / (2.*bA + ALP2 * log(sCM/sMaxAX)) ) / ALP2;
  if (withEnhancement) {
    mResAXsave = mBNow + MRES0;
    sResAX    = pow2(mResAXsave);
    sRMavgAX  = mResAXsave * mMinAXsave;
    sRMlogAX  = log1p(sResAX/sMinAX);
    BcorrAX   = CSD[iSD][6] + CSD[iSD][7] / sCM;
    sum2      = CRES * sRMlogAX / (2.*bA + ALP2 * log(sCM/sRMavgAX) + BcorrAX);
  }
  if (lowE) {
    double ratio = max( 0., eCMsave - mANow - mMinAXsave)
                 / (ECMMIN  - mANow - mMinAXsave);
    double ratio03 = pow(ratio, 0.3);
    double ratio06 = pow2(ratio03);
    sum1     *= ratio06;
    sum2     *= ratio03;
  }
  sigAX       = CONVERTSD * scaleC * XPROC[iProc] * BETA0[iHadA]
              * max( 0., sum1 + sum2);

  // Double diffractive scattering A + B -> X1 + X2 cross section.
  y0min       = log( sCM * sp / (sMinXB * sMinAX) ) ;
  sLog        = log(sCM);
  Delta0      = CDD[iDD][0] + CDD[iDD][1] / sLog + CDD[iDD][2] / pow2(sLog);
  sum1        = (y0min * (log( max( 1e-10, y0min/Delta0) ) - 1.) + Delta0)
              / ALP2;
  if (y0min < 0.) sum1 = 0.;
  if (withEnhancement) {
    sMaxXX    = sCM * ( CDD[iDD][3] + CDD[iDD][4] / sLog
              + CDD[iDD][5] / pow2(sLog) );
    sLogUp    = log( max( 1.1, sCM * SZERO / (sMinXB * sRMavgAX) ));
    sLogDn    = log( max( 1.1, sCM * SZERO / (sMaxXX * sRMavgAX) ));
    sum2      = CRES * log( sLogUp / sLogDn ) * sRMlogAX / ALP2;
    sLogUp    = log( max( 1.1, sCM * SZERO / (sMinAX * sRMavgXB) ));
    sLogDn    = log( max( 1.1, sCM * SZERO / (sMaxXX * sRMavgXB) ));
    sum3      = CRES * log(sLogUp / sLogDn) * sRMlogXB / ALP2;
    BcorrXX   =  CDD[iDD][6] + CDD[iDD][7] / eCMNow + CDD[iDD][8] / sCM;
    sum4      = pow2(CRES) * sRMlogAX * sRMlogXB / max( 0.1,
                ALP2 * log( sCM * SZERO / (sRMavgAX * sRMavgXB) ) + BcorrXX);
  }
  if (lowE) {
    double ratio = max( 0., eCMsave - mMinXBsave - mMinAXsave)
                 / (ECMMIN  - mMinXBsave - mMinAXsave);
    double ratio05 = sqrt(ratio);
    double ratio025 = sqrt(ratio05);
    sum1     *= ratio * ratio05;
    sum2     *= ratio * ratio025;
    sum3     *= ratio * ratio025;
    sum4     *= ratio;
  }
  sigXX       = CONVERTDD * scaleC * XPROC[iProc]
              * max( 0., sum1 + sum2 + sum3 + sum4);
  if (eCMsave < mMinXBsave + mMinAXsave) sigXX = 0.;

  // Restore original order, and order single diffractive correctly.
  if (swapped) {
    swap( idBNow, idANow);
    swap( mBNow,  mANow);
    swap( bB,  bA);
    swap( sigXB, sigAX);
    swap( mMinXBsave, mMinAXsave);
    swap( mResXBsave, mResAXsave);
   }

  if (didSwapIds)
    swap(sigXB, sigAX);

}

//--------------------------------------------------------------------------

// Calculate excitation cross section. This may be dependent on total,
// elastic and diffractive cross sections.

void LowEnergySigma::calcEx() {

  // Excitations are available only for NN collisions
  if ( (abs(idA) == 2212 || abs(idA) == 2112)
    && (abs(idB) == 2212 || abs(idB) == 2112) ) {
    if (eCM < 3.)
      sigEx = sigTot - sigEl - sigXB - sigAX - sigXX - sigAnn;
    else
      sigEx = min(nucleonExcitationsPtr->sigmaExTotal(eCM),
                sigTot - sigEl - sigXB - sigAX - sigXX - sigAnn);
  }
  else
    sigEx = 0.;
}

//--------------------------------------------------------------------------

// Sets the configuration of colliding particles, ensuring canonical ordering.
//
// The canonical ordering of A and B is defined as follows:
//   1) If one is a baryon and the other is meson, then A must be the baryon.
//      Otherwise, select them such that |A| >= |B|.
//   2) A must be positive.
//
// Implications:
//  - In BB, the antiparticle cross sections are the same as the particle ones,
//    so BbarBbar is replaced by BB.
//  - In BBbar, A is always the particle and B is the antiparticle.
//    Again signs are flipped if necessary.
//  - In XM, X is a baryon or meson and B is always a meson.
//
// This sets collType for the overall process type:
//  1: BB;
//  2: BBbar;
//  3: XM;
// This also sets didFlipSign if both particles were replaced by their
// antiparticles, and didSwapIds if the particle ids were swapped.
//
void LowEnergySigma::setConfig(int idAIn, int idBIn, double eCMIn,
  double mAIn, double mBIn) {

  // Store input and clear cross sections.
  idA = idAIn;
  idB = idBIn;
  eCM = eCMIn;
  mA  = mAIn;
  mB  = mBIn;

  sigTot = sigND = sigEl = sigXB = sigAX = sigXX
         = sigAnn = sigEx = sigResTot = 0.;
  sigRes.clear();

  // Ensure |A| >= |B|.
  bool mesA = particleDataPtr->isMeson(idA);
  bool mesB = particleDataPtr->isMeson(idB);
  didSwapIds = (mesA && !mesB) || (mesA == mesB && abs(idA) < abs(idB));
  if (didSwapIds) {
    swap(idA, idB);
    swap(mA, mB);
    swap(mesA, mesB);
  }

  // Ensure A > 0.
  didFlipSign = idA < 0;
  if (didFlipSign) {
    idA = -idA;
    idB = particleDataPtr->antiId(idB);
  }

  // Get type of overall collision: XM, BBbar, BB.
  if (mesB) collType = 3;
  else if (idB < 0) collType = 2;
  else collType = 1;

}

//--------------------------------------------------------------------------

// HPR1R2 fit for parameterizing certain total cross sections.
// http://doi.org/10.1088/1674-1137/40/10/100001 p. 591

double LowEnergySigma::HPR1R2(double p, double r1, double r2,
  double mAIn, double mBIn, double s) const {

  static constexpr double H    = 0.2720;
  static constexpr double M    = 2.1206;
  static constexpr double eta1 = 0.4473;
  static constexpr double eta2 = 0.5486;

  double ss = s / pow2(mAIn + mBIn + M);
  return p + H * pow2(log(ss)) + r1 * pow(ss, -eta1) + r2 * pow(ss, -eta2);

}

//--------------------------------------------------------------------------

// HERA/CERN fit for parameterizing certain elastic cross sections.
// https://doi.org/10.1103/PhysRevD.50.1173 p. 1335

double LowEnergySigma::HERAFit(double a, double b, double n, double c,
  double d, double p) const {
  return a + b * pow(p, n) + c * pow2(log(p)) + d * log(p);
}

//--------------------------------------------------------------------------

// Additive quark model for generic collisions and for scale factors.

double LowEnergySigma::nqEffAQM(int id) const {

  // ssbar mixing into eta and eta' handled as special cases.
  if (id == 221) return 2. * (1. - fracEtass  + sEffAQM * fracEtass);
  if (id == 331) return 2. * (1. - fracEtaPss + sEffAQM * fracEtaPss);

  // Count up number of quarks of each kind for hadron and combine.
  int idAbs = abs(id);
  int nq[10] = {};
  ++nq[(idAbs/10)%10];
  ++nq[(idAbs/100)%10];
  ++nq[(idAbs/1000)%10];
  return nq[1] + nq[2] + sEffAQM * nq[3] + cEffAQM * nq[4] + bEffAQM * nq[5];

}

double LowEnergySigma::factorAQM() const {
  return nqEffAQM(idA) * nqEffAQM(idB) / 9.;
}

double LowEnergySigma::totalAQM() const {
  return 40. * factorAQM();
}

double LowEnergySigma::elasticAQM() const {
  double aqmT = totalAQM();
  return 0.039 * sqrt(aqmT) * aqmT;
}

//--------------------------------------------------------------------------

// Check which cross sections contain explicit resonances.

bool LowEnergySigma::hasExplicitResonances() const {

  // N has explicit resonances with pi, Kbar, eta and omega.
  if (idA == 2212 || idA == 2112)
    return abs(idB) == 211 || idB == 111 || idB == -321 || idB == -311
        || idB == 221 || idB == 223;

  // pi+pi0, pi+pi-, pi0pi0 and pi0pi-.
  if (idA == 211 && (idB == 111 || idB == -211))
    return true;
  if (idA == 111 && idB == 111)
    return true;

  // K pi and K Kbar.
  if (idA == 321)
    return idB == 111 || idB == -211 || idB == -321 || idB == -311;
  if (idA == 311)
    return idB == 111 || idB == 211  || idB == -321 || idB == -311;

  // Sigma+ and Sigma- can have resonances with pi and Kbar,
  // and always have resonances with K.
  if (idA == 3222)
    return idB == 111 || idB == -211 || idB == -321
        || idB == 321 || idB == 311;
  if (idA == 3112)
    return idB == 111 || idB == 211  || idB == -311
        || idB == 321 || idB == 311;

  // Sigma0/Lambda have resonances with all pi, K and Kbar.
  if (idA == 3212 || idA == 3122)
    return idB == 211 || idB == 111 || idB == -211
        || idB == 321 || idB == 311 || idB == -321 || idB == -311;

  // Xi0 and Xi- can have resonances with pi.
  // They can in principle have resonances with K/Kbar, but
  //   1) Omega resonances are not implemented, and
  //   2) Sigma* -> Xi+K branching ratios are very small.
  if (idA == 3322)
    return idB == 111 || idB == -211;
  if (idA == 3312)
    return idB == 111 || idB == 211;

  // No further combinations have implemented resonances.
  return false;

}


//--------------------------------------------------------------------------

// Give meltpoint, below which sigma(total) = sum sigma(res) + sigma(elastic).

double LowEnergySigma::meltpoint(int idX, int idM) const {

  // p+M.
  if (idX == 2212)
    return idM == -211 ? 1.75
         : idM ==  211 ? 2.05
         : idM ==  111 ? 2.00
         : idM == -321 ? 2.10
         : idM == -311 ? 2.10
         : idM ==  221 ? 1.75
         : idM ==  223 ? 1.95
         : 0.;

  // n0M.
  if (idX == 2112)
    return idM == -211 ? 2.00
         : idM ==  211 ? 1.90
         : idM ==  111 ? 2.00
         : idM == -321 ? 2.10
         : idM == -311 ? 2.10
         : idM ==  221 ? 1.75
         : idM ==  223 ? 1.95
         : 0.;

  // LambdaM.
  if (idX == 3122)
    return abs(idM) == 211 || idM == 111 ? 2.05
         : abs(idM) == 321 || abs(idM) == 311 ? 2.00
         : 0.;

  // SigmaM.
  if (idX == 3222 || idX == 3212 || idX == 3112)
    return abs(idM) == 211 || idM == 111 ? 2.0
         : abs(idM) == 321 || abs(idM) == 311 ? 2.05
         : 0.;

  // XiM.
  if (idX == 3322 || idX == 3312) {
    if (abs(idM) == 211 || idM == 111)
      return 1.6;
    else
      return 0.;
  }

  // pipi.
  if ((abs(idX) == 211 || idX == 111) && (abs(idM) == 211 || idM == 111))
    return 1.42;

  // Kpi.
  if ((abs(idX) == 321 || abs(idX) == 311)
    && (abs(idM) == 211 || abs(idM) == 111))
    return 1.60;

  // KK.
  if ((abs(idX) == 321 || abs(idX) == 311)
   && (abs(idM) == 321 || abs(idM) == 311))
   return 1.65;

  // No further combinations are defined with resonances.
  return 0.;
}

//==========================================================================

}
