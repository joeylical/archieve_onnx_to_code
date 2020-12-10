// Auto generated by dump.py

#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

#define float_IS_ZERO(value) ((value) > -0.0001f && (value) < 0.0001f)
#define uint8_t_IS_ZERO(value) (!(value))


// layer: conv1.0.bias shape: (4,)
const float layer_conv1_0_bias[4] = 
{0.012068589, -0.81733847, -0.5753527, -1.644566};

// layer: conv1.0.weight shape: (4, 1, 3, 3)
const float layer_conv1_0_weight[4][1][3][3] = 
{
  {
    {
      {0.16419885, 0.11747783, 0.11801928},
      {-0.4849659, -0.362503, 0.58499604},
      {0.05605864, -0.006774251, 0.23456974},
    },
  },
  {
    {
      {0.5239138, -0.2638289, -0.80555236},
      {0.34485415, 1.1390537, 0.08342136},
      {-0.40685377, 0.010646204, 0.67581314},
    },
  },
  {
    {
      {0.41384387, 0.4295634, 0.34900865},
      {-0.46017396, 0.030486876, 0.22405931},
      {-0.36303368, -0.4412347, -0.54201746},
    },
  },
  {
    {
      {-0.85155225, -0.23894325, 0.052182566},
      {-0.051337454, -0.50477415, -0.30772248},
      {0.019891376, -0.12900066, -0.5758395},
    },
  },
};

// layer: conv1.3.bias shape: (8,)
const float layer_conv1_3_bias[8] = 
{-0.07484604, -0.20836836, 0.160943, 0.11757406, 0.34018373, 0.40397254, -0.17087589, -0.15587983};

// layer: conv1.3.weight shape: (8, 4, 3, 3)
const float layer_conv1_3_weight[8][4][3][3] = 
{
  {
    {
      {-0.11202956, -0.025301544, -0.17648079},
      {-0.14206828, 0.015419483, 0.07207043},
      {-0.047219556, -0.11540466, -0.03090424},
    },
    {
      {-0.19729339, 0.03170303, -0.030127317},
      {0.073213845, -0.1773196, -0.09131042},
      {-0.12463301, -0.14881541, 0.011604252},
    },
    {
      {0.03035529, -0.05999551, -0.14705133},
      {-0.047581974, 0.020858943, -0.07042067},
      {0.015046844, -0.06389357, 0.03410557},
    },
    {
      {-0.073470965, -0.14395547, -0.15326081},
      {0.09201635, 0.047164094, -0.055052012},
      {-0.1122834, -0.12532608, -0.08122643},
    },
  },
  {
    {
      {-0.029837485, 0.034926128, -0.70838505},
      {-0.68230337, 0.2699418, -0.08563392},
      {0.2340018, -0.3564074, 0.04791378},
    },
    {
      {-0.51969177, -1.297105, -0.5333749},
      {0.9136432, 0.6875402, 0.63402545},
      {0.9446356, 0.77222663, 0.5643755},
    },
    {
      {-0.8472527, 0.72667503, 0.28192264},
      {0.12431303, -0.22619818, -0.07743142},
      {-0.30908245, 0.28353366, 0.3932131},
    },
    {
      {-0.76204914, -0.1458092, 0.82769567},
      {-0.5391324, 0.036237013, -0.05480827},
      {-0.4557027, -0.2504538, -0.43847808},
    },
  },
  {
    {
      {-0.37788823, -0.67455256, 0.7695674},
      {-0.01252505, 0.9455811, -0.0086480975},
      {0.17098051, -0.8415877, 0.17564107},
    },
    {
      {-0.29190385, 0.2634518, 0.025184654},
      {0.113923565, -0.40929022, -0.24781136},
      {0.36319318, -0.1954842, -0.021984477},
    },
    {
      {0.18240802, 0.7263403, -0.50452507},
      {-0.07646485, -0.59983027, 0.6208134},
      {-0.28148928, 0.48615423, 0.5380413},
    },
    {
      {0.44694307, 0.6758118, 0.37139916},
      {0.17063542, -1.712543, -0.8531068},
      {-1.4237199, -2.2576373, 0.23136352},
    },
  },
  {
    {
      {-0.41806677, 0.19338916, 0.7210292},
      {-1.1168019, 0.29434273, 0.47914666},
      {-0.15346389, -0.0014043894, 0.96185076},
    },
    {
      {0.1052417, 0.059571907, -0.07916966},
      {0.33900467, -0.13103992, 0.0028551733},
      {0.5424758, 0.32133397, 0.10622093},
    },
    {
      {0.7185143, 0.07530356, -0.10786665},
      {0.21055207, -0.36754248, -0.15693396},
      {-0.11289137, -0.0805898, -0.36324775},
    },
    {
      {-0.2878888, -0.040162727, -0.95119226},
      {0.34507555, 0.21144295, -1.9825566},
      {0.41448364, 0.26092908, -1.6338197},
    },
  },
  {
    {
      {0.119749285, 0.29326788, 0.16049428},
      {0.011645174, -0.17187801, 0.37761542},
      {0.30328166, 0.16932575, -0.09323172},
    },
    {
      {0.50631934, 0.057176698, 0.243693},
      {0.5832873, -0.061555035, -0.17955475},
      {0.0067738993, -0.5636833, -0.10054652},
    },
    {
      {0.5975972, 0.82052505, 0.42471468},
      {0.8703062, 0.07112165, -0.067453615},
      {-0.5581427, -0.37801787, 0.10058052},
    },
    {
      {-0.9912375, -0.9787445, -1.0267788},
      {0.14283559, -0.18559462, -1.3229394},
      {0.13101691, 0.18749052, 0.17398447},
    },
  },
  {
    {
      {-0.17744742, 0.20618553, 0.32028723},
      {0.21298796, 0.42030218, -0.033125937},
      {0.6126319, 0.53914154, -0.6552938},
    },
    {
      {0.7137572, 0.28974384, 0.4494163},
      {0.4219316, -0.01231901, -0.112061284},
      {-1.3722147, -1.2514924, -0.4923923},
    },
    {
      {-0.023830319, -0.25093937, 0.03854663},
      {0.27084827, -0.5113978, -0.5270049},
      {-0.42962894, -1.412131, -0.005108264},
    },
    {
      {0.3079039, 0.096927255, 0.034806427},
      {-0.03554102, 0.13913676, -0.55372775},
      {-0.12869702, -1.4113474, -0.15354943},
    },
  },
  {
    {
      {0.024023142, -0.13694327, -0.042277027},
      {-0.09852959, -0.0030480588, -0.066670835},
      {-0.038438294, -0.08064624, -0.14155279},
    },
    {
      {0.07142026, -0.13085806, 0.029441219},
      {-0.1938793, -0.21029192, 0.07112549},
      {-0.16187312, -0.008900931, -0.15763612},
    },
    {
      {-0.05061995, -0.11178616, 0.07425963},
      {-0.13670725, -0.052546874, -0.1400803},
      {-0.061068516, 0.03218676, -0.17113654},
    },
    {
      {-0.19594961, -0.00446475, 0.026089942},
      {-0.18216386, 0.0014344822, -0.17168787},
      {-0.12901147, -0.029122189, -0.017154947},
    },
  },
  {
    {
      {-0.11289377, -1.9660454, -1.8663223},
      {-1.8206156, -0.43670288, -0.20464592},
      {0.8729789, 0.32225978, 0.104307845},
    },
    {
      {0.0057572727, -1.152341, -1.6089212},
      {-0.59835166, -0.6877395, 0.50860626},
      {-0.9409917, 0.83827454, 0.50464076},
    },
    {
      {1.1479511, 1.4642093, 0.51633817},
      {-0.32252988, -0.6713133, -0.4167896},
      {-1.3097835, -0.049955357, 0.64709353},
    },
    {
      {0.29039568, 0.4796663, 0.40008336},
      {-0.00073365885, -0.13686337, -0.21370165},
      {0.035004787, 0.061421864, -0.08342222},
    },
  },
};

// layer: conv1.6.bias shape: (16,)
const float layer_conv1_6_bias[16] = 
{0.38790166, -0.75463784, 0.881266, -1.7277125, -0.21923609, -1.5932575, -2.3842187, -0.13875142, -0.9669386, -1.5123605, -0.20788756, 1.085496, -1.6663865, -0.07899732, 0.70292026, 1.3622745};

// layer: conv1.6.weight shape: (16, 8, 3, 3)
const float layer_conv1_6_weight[16][8][3][3] = 
{
  {
    {
      {0.041477058, -0.061477304, -0.10539342},
      {-0.1483118, -0.08815705, 0.12587251},
      {0.05812701, -0.04763501, 0.1816229},
    },
    {
      {-0.15976542, -0.43413, 0.24060965},
      {-1.971683, -2.3689487, -0.50042135},
      {-1.2070591, -1.0082846, 0.1464736},
    },
    {
      {0.59993875, -0.37743673, 0.046105478},
      {0.5055645, 0.45339227, -1.9499904},
      {0.07612228, 0.35790634, 0.12099037},
    },
    {
      {-0.5662607, -0.054709665, -0.41671193},
      {0.6404933, -0.6078585, 0.5969097},
      {0.7177423, -0.9267034, 0.23298809},
    },
    {
      {0.17911556, 0.20322387, 0.61222386},
      {0.5344517, 0.087469496, 0.29445818},
      {-0.78309727, -1.2404772, -1.3882283},
    },
    {
      {0.5424776, 0.3688147, -0.2810884},
      {0.7543714, -0.32131657, -1.6329116},
      {0.32702157, -1.210125, 0.120306075},
    },
    {
      {-0.065364674, -0.05320794, 0.0708571},
      {0.020085432, -0.092967294, -0.03345222},
      {-0.10248325, -0.071494624, -0.0494653},
    },
    {
      {-0.34090653, -0.46169677, -2.0689173},
      {0.4308698, 0.03961827, 0.7039946},
      {0.040122557, -0.027466048, 0.007734289},
    },
  },
  {
    {
      {0.020334829, -0.046534237, -0.053604145},
      {-0.005655898, -0.044411615, -0.12656467},
      {0.008632601, -0.10429068, 0.06475538},
    },
    {
      {-0.029151654, 0.1378881, -0.9113251},
      {0.96909016, 0.88868237, 0.15585764},
      {-0.3914773, -0.72049534, -0.52825874},
    },
    {
      {1.287366, 0.11352009, -1.2015505},
      {1.0956559, 0.32733738, -1.4503545},
      {1.4128278, 1.262895, -0.26480654},
    },
    {
      {-0.90764755, -2.130898, -1.8666717},
      {-0.7244527, -0.9467641, -2.1138542},
      {0.67781866, 0.16625781, -1.8918508},
    },
    {
      {0.22773354, 0.8951766, 0.20696852},
      {-0.019651813, 0.35212106, 0.0770586},
      {-0.5010301, -0.36563703, -1.379504},
    },
    {
      {-0.020430978, -1.9495082, -2.4951088},
      {0.41732723, 0.8151337, 0.34631217},
      {0.729788, 0.41217694, 0.9261675},
    },
    {
      {0.044345114, -0.06462806, -0.10328839},
      {0.020174498, -0.012066161, 0.02804624},
      {-0.113663346, 0.022563593, 0.009990931},
    },
    {
      {-0.25566587, 0.106494814, -1.0862525},
      {0.3428003, -0.7915176, -0.54832417},
      {-0.3681501, -1.0968201, -1.4410552},
    },
  },
  {
    {
      {-0.00044420577, -0.019966278, -0.051817227},
      {-0.09529699, -0.16144705, -0.0050575756},
      {-0.015034507, 0.013718906, 0.016005648},
    },
    {
      {0.5696549, -0.11636314, -2.9228425},
      {0.4844161, 0.8178156, -0.28769216},
      {-1.3775449, 0.27650356, 0.10848807},
    },
    {
      {-0.37712964, -0.19695874, -0.45622426},
      {-0.5896002, 0.69574666, 0.48975343},
      {0.05044927, 0.34898457, 0.35437995},
    },
    {
      {0.046367325, -0.11116957, -1.3162756},
      {-0.2664304, -0.9637676, 0.33373716},
      {-1.0263281, -0.37194127, -0.92753595},
    },
    {
      {-0.6534007, -0.6385955, -1.8905976},
      {-0.8342477, -0.8123504, -0.6436884},
      {0.46392828, 0.6511558, 0.53825366},
    },
    {
      {0.13778217, -0.3654691, -0.32077086},
      {0.8925559, -1.1815228, 0.48304027},
      {-0.5979897, -0.66635823, 0.41894087},
    },
    {
      {-0.09331001, 0.05140324, 0.05009001},
      {0.033270348, 0.09327521, -0.091519475},
      {-0.02726473, -0.010794199, -0.10322813},
    },
    {
      {0.5438642, -0.5747471, -0.30048257},
      {-0.6743633, -0.796304, -0.115181416},
      {-0.26399583, 0.117994465, -0.33616757},
    },
  },
  {
    {
      {0.0045458125, -0.039345548, -0.010174504},
      {0.020917017, 0.14578782, 0.03980959},
      {0.018702641, 0.015966173, 0.12656778},
    },
    {
      {-0.31565544, -0.042285223, 0.49712145},
      {-0.28469175, -1.0140462, -0.31144428},
      {-1.8230988, 0.53499895, 0.08232706},
    },
    {
      {0.04072448, -0.067792006, -0.57742107},
      {-0.10896561, 0.8219671, 0.2214989},
      {-0.9950892, -0.26784292, -0.16338494},
    },
    {
      {-0.8673073, -0.52354795, 0.12064058},
      {-0.24410045, 0.5202407, 0.9917521},
      {-0.39635327, -0.19584298, -0.8867344},
    },
    {
      {0.48597682, -0.31773502, 0.42832047},
      {0.32173464, 0.86181456, 0.39841363},
      {0.29307428, -0.12048599, -0.69492656},
    },
    {
      {0.12389966, -0.15524742, -0.18793914},
      {-0.46186006, 0.2413619, 0.7760601},
      {-0.921675, -0.6700827, -0.6069884},
    },
    {
      {0.0599078, 0.05216782, 0.07349159},
      {-0.051831104, 0.1564756, 0.05901229},
      {-0.037274577, 0.065800235, -0.059455555},
    },
    {
      {0.85435236, 0.82479256, -1.5428681},
      {0.5589704, 0.4273422, -0.66798913},
      {0.5399093, -0.41004264, 0.22211665},
    },
  },
  {
    {
      {0.043398924, -0.17618209, 0.014623906},
      {0.007065507, -0.028862124, -0.11145739},
      {-0.051465206, 0.036370724, -0.10349647},
    },
    {
      {-2.1862159, -0.8021219, -0.5787368},
      {-0.90912867, -0.07771569, 0.074170865},
      {-0.5714084, 1.1706525, 0.6351366},
    },
    {
      {-2.498152, 0.404786, 0.5841276},
      {-1.335306, -1.9915066, 0.53136116},
      {-2.9515593, 0.45088738, -0.049324483},
    },
    {
      {-0.46077546, -0.010511985, 0.6208044},
      {-2.0862186, -0.56168586, 0.14244433},
      {0.58371043, -0.32035005, 0.53315204},
    },
    {
      {-1.5403398, -0.19602305, 0.48814842},
      {-1.3285067, 0.19655383, 0.2987977},
      {-0.03650192, -0.36416367, 0.27544907},
    },
    {
      {-3.8135855, -1.0173492, 0.3264404},
      {-1.2726126, -1.7103051, -0.31565034},
      {-0.62992054, -0.8595713, -0.20900528},
    },
    {
      {0.016264804, -0.05156035, 0.059065424},
      {-0.09108816, -0.047130093, -0.061827585},
      {-0.0117673725, 0.046714164, 0.019754283},
    },
    {
      {-2.2893953, -0.4621493, 0.6639773},
      {0.7349123, 0.9061855, 0.9595667},
      {1.0826591, 0.20539032, 0.13753782},
    },
  },
  {
    {
      {-0.034394752, 0.07548754, -0.10362431},
      {0.031872936, -0.009268506, -0.034352023},
      {0.13683482, 0.058473114, -0.09152521},
    },
    {
      {0.4035884, -0.2027866, 0.26900375},
      {-0.54718703, 1.0056664, -0.45509103},
      {-0.7773151, 0.64908457, -0.2758272},
    },
    {
      {0.54178524, 2.581347, 0.13579398},
      {-1.7503961, -0.04219778, 1.3792531},
      {-0.61415017, -0.038685914, 0.30274132},
    },
    {
      {-0.074209936, 0.4105979, -0.44846702},
      {0.3469366, 0.16431527, 0.14913142},
      {-0.3095421, 0.077234164, -0.44056472},
    },
    {
      {-0.07339054, 0.002075293, -0.09427749},
      {-0.83631724, 0.116779804, -0.8149232},
      {-0.3740629, 0.8354958, 0.3691797},
    },
    {
      {0.5508304, 0.28682026, -0.18493198},
      {-0.5041355, -0.5623371, 0.21546778},
      {0.85926133, -0.15512231, -0.37083322},
    },
    {
      {-0.11301504, 0.057002388, -0.08330915},
      {-0.09441328, -0.004510707, -0.021818327},
      {-0.10349765, 0.021482604, -0.00900932},
    },
    {
      {-1.3121244, 0.51437557, 0.17813691},
      {-2.312501, -0.8630508, 0.16187066},
      {-0.70537245, -0.0028732691, 0.045554},
    },
  },
  {
    {
      {0.09787468, 0.05051965, 0.012673751},
      {0.030031273, 0.116445385, -0.0034510365},
      {0.025514593, -0.071711615, 0.016312562},
    },
    {
      {0.16344814, 0.6445928, -0.4247116},
      {0.54602474, 0.20397359, 0.47580785},
      {0.15964094, -0.6026541, -0.37673777},
    },
    {
      {0.021842295, 0.5929474, 0.54657716},
      {-0.004224982, -0.20754759, -1.1471187},
      {0.36055407, -0.0076129185, -0.20600289},
    },
    {
      {0.1685975, -0.40186882, -1.6550242},
      {0.43913728, -0.3754981, -1.0839335},
      {0.03207112, 0.56561446, 0.78656894},
    },
    {
      {-0.80940396, 1.3378263, 0.75154775},
      {-0.5515863, 0.3401127, 0.554428},
      {-0.117324874, -0.2527304, 0.30641153},
    },
    {
      {0.22417682, -0.8690158, -3.154354},
      {0.3789928, -0.3572638, -0.4689317},
      {0.34475207, 0.11448292, 0.87194085},
    },
    {
      {-0.0031043328, -0.06952539, -0.087252684},
      {0.03520924, 0.0078701265, -0.014044906},
      {0.04528375, 0.06262041, -0.012857931},
    },
    {
      {-1.7633039, 0.0066532805, 1.3783885},
      {-0.5634661, 0.45446646, 0.7106268},
      {-0.22614159, 0.29809725, -0.51051503},
    },
  },
  {
    {
      {-0.14013776, -0.07434739, 0.004451693},
      {-0.015530815, -0.11589429, 0.0012267248},
      {0.02035713, 0.0070862337, -0.098116145},
    },
    {
      {-0.06357188, -0.17272788, -0.077380024},
      {-0.014038108, -0.10873856, -0.07456055},
      {-0.059092406, -0.19208421, -0.04501719},
    },
    {
      {0.05860881, -0.09562432, 0.055427518},
      {-0.3914473, -0.28173947, 0.056404836},
      {0.06652581, -0.110118195, -0.045440733},
    },
    {
      {-0.0714915, -0.0076153823, -0.00830783},
      {-0.078891546, -0.1328585, -0.11510477},
      {-0.0015858427, -0.10007558, 0.043602556},
    },
    {
      {-0.080984004, 0.04768054, -0.08327446},
      {-0.14784168, 0.04301209, -0.03757355},
      {-0.17212225, -0.010268605, 0.02021031},
    },
    {
      {-0.051696733, -0.07339118, 0.013917192},
      {-0.0032710573, -0.16145192, -0.018258909},
      {-0.05511747, 0.019984327, 0.0020835288},
    },
    {
      {0.0056792116, 0.051979687, 0.087257445},
      {-0.020804927, -0.02399071, 0.14271764},
      {0.035039596, 0.13357951, 0.0996038},
    },
    {
      {-0.12906644, -0.14971761, 0.016318217},
      {-0.12131818, -0.20135419, -0.11266084},
      {-0.028503533, -0.12859553, -0.1393755},
    },
  },
  {
    {
      {-0.04528178, 0.010652129, 0.029797515},
      {-0.085708104, 0.021477228, 0.061962724},
      {-0.008014502, -0.15323204, -0.13296942},
    },
    {
      {-0.34208703, 0.8271547, 0.13070035},
      {-0.26788977, -0.43875918, -0.18128562},
      {0.33337176, -2.1242247, -2.282343},
    },
    {
      {0.2605853, 0.04742249, -0.8632482},
      {0.12483515, 0.8363413, 0.11784326},
      {1.8490258, 0.7226911, -0.88244015},
    },
    {
      {0.22689103, 0.2287205, -0.6696167},
      {0.43501842, 0.44842595, -1.3650478},
      {0.21058157, 0.28895757, -0.34471437},
    },
    {
      {0.32237557, 1.1267371, 0.7660526},
      {0.4202893, 0.046972074, -0.92862564},
      {0.5004388, -0.24240524, -1.7597506},
    },
    {
      {0.24059129, -0.6309139, -0.85030204},
      {-0.31306267, 0.45011994, 0.41704136},
      {0.23717867, 0.6345604, -0.5591571},
    },
    {
      {0.04251756, -0.030652208, 0.15748237},
      {-0.08659518, -0.17440414, -0.023012778},
      {0.012896991, 0.17669529, 0.16147998},
    },
    {
      {-0.60826206, -0.3886991, -2.0070772},
      {-0.34463662, -0.65338624, -0.9210038},
      {-0.036667082, 0.01198647, -0.42005908},
    },
  },
  {
    {
      {-0.05021293, 0.048787717, -0.14578103},
      {-0.1112357, -0.07810591, -0.062040642},
      {-0.123887114, -0.02262581, -0.081980094},
    },
    {
      {-0.3069972, 0.21542895, -0.23370188},
      {-0.25015065, 0.096112475, -0.26812562},
      {-0.30573696, 0.7002829, 0.82621485},
    },
    {
      {0.13754882, -0.42962343, -0.05609243},
      {1.3817748, 1.0501049, 0.7295645},
      {0.7951948, 0.8209757, 0.06429514},
    },
    {
      {0.3679709, 0.36204425, 1.1496934},
      {-0.3263297, 0.10902468, 0.23373161},
      {0.6635513, 0.10965291, -0.90318936},
    },
    {
      {0.31094146, 0.80275255, -0.024770014},
      {0.45686513, -0.24833003, -0.32928985},
      {-0.8680892, -0.3500039, 0.121463545},
    },
    {
      {-0.5632973, -0.08231384, 0.19818737},
      {0.30792612, -0.36816147, -0.17453429},
      {1.7246583, 0.4100849, 0.09332195},
    },
    {
      {0.0029793303, -0.056435842, -0.02215413},
      {0.08696561, -0.15626276, 0.09433073},
      {-0.061312586, -0.11528291, -0.036428887},
    },
    {
      {-0.20950557, -0.6340599, -2.0567672},
      {-0.27309132, -0.69450194, -0.25298348},
      {-0.47322148, -0.52933234, -0.2699834},
    },
  },
  {
    {
      {-0.06551315, 0.06802327, 0.07431958},
      {0.10630829, 0.09846925, 0.040800296},
      {0.032226063, 0.0913711, -0.019245349},
    },
    {
      {-0.21723853, -0.2588372, 0.13834931},
      {-2.61783, -2.646392, -0.43608668},
      {-3.266681, -2.1924422, -0.70323735},
    },
    {
      {0.67847353, 0.9054964, 0.76755536},
      {0.3312587, 0.94581753, 0.0025963886},
      {0.01523791, -1.2347271, -0.4761202},
    },
    {
      {-1.2641325, 0.3281313, 0.40580127},
      {0.11247961, 0.9214033, -1.2183869},
      {0.47150764, 0.6990099, 0.105653375},
    },
    {
      {-0.28641164, -0.20537071, -1.068351},
      {-0.069361776, -0.16730611, 0.17757855},
      {-0.44011405, -0.6833002, -0.49916548},
    },
    {
      {0.1033462, 0.57353044, 0.8983822},
      {-0.32730925, 0.39466047, -0.4951276},
      {0.23659573, -0.30901068, -1.3015414},
    },
    {
      {-0.10819426, 0.0073852465, -0.05094585},
      {-0.056398172, -0.1145512, -0.017653182},
      {-0.022302128, 0.06822284, 0.014835916},
    },
    {
      {0.33782485, -0.108769186, -0.47842023},
      {-0.62278605, -0.16499619, -0.76437557},
      {-0.41559675, -1.0929801, -0.5834979},
    },
  },
  {
    {
      {0.052244864, -0.16240767, 0.09140562},
      {0.0065472745, -0.1465178, -0.13007157},
      {-0.18281928, -0.15938604, -0.10731855},
    },
    {
      {0.43567806, 0.5767165, -2.0749817},
      {0.12365223, -0.3330605, -1.3079398},
      {-0.9693622, 0.4235835, -0.9621337},
    },
    {
      {-0.28088635, 0.74395794, -0.118687354},
      {-0.7035689, 0.6721443, -0.41274205},
      {-2.0635772, -0.29278928, -2.1167886},
    },
    {
      {0.023180794, 0.94164646, 0.38629398},
      {-1.8544528, -0.04450217, 0.9787486},
      {0.4466459, 0.152667, 0.38499644},
    },
    {
      {0.38295785, -0.74742657, -0.23012447},
      {0.6584439, 0.49924114, 0.77093416},
      {-0.07002597, 0.115665466, -0.47781533},
    },
    {
      {0.9434108, -1.7966329, -0.5544203},
      {-0.051892683, -1.092612, -0.18891558},
      {-0.20825516, -0.43835726, -1.7083126},
    },
    {
      {-0.09131168, 0.041837994, 0.01115523},
      {-0.094893664, -0.015523039, -0.078785814},
      {0.015818914, 0.033076733, 0.009495561},
    },
    {
      {-1.6153016, 0.16678125, 0.2963014},
      {0.42714548, 0.5962134, -2.2120337},
      {-0.47123298, 1.4500811, -2.266016},
    },
  },
  {
    {
      {-0.03334265, -0.026343329, 0.14012435},
      {-0.031095762, 0.16556504, 0.025790842},
      {0.014554285, 0.015228232, 0.034206197},
    },
    {
      {-0.7309127, -0.045337543, -0.36603785},
      {-0.055394597, 0.4366571, 0.6794541},
      {0.36636764, 0.06703204, 0.21192479},
    },
    {
      {-0.27584767, 0.49716076, 0.59753346},
      {0.5022728, 0.57440275, 0.25287786},
      {0.087186344, 0.75421774, 0.13945654},
    },
    {
      {0.70896333, 0.6544234, 0.13109489},
      {-0.027587164, -0.1403848, 0.8098018},
      {0.043988906, 0.01767935, 0.41389176},
    },
    {
      {-2.1634827, -2.6534748, -2.122776},
      {-1.4201715, 0.37507102, -0.6793118},
      {0.66076285, 0.3438767, 0.5131874},
    },
    {
      {0.73980105, 0.54882115, 0.3177642},
      {-0.11715583, 0.036723617, -0.014871959},
      {0.19479686, 0.5948467, 0.14402336},
    },
    {
      {-0.0758862, -0.1483353, -0.025331331},
      {-0.109987795, 0.108686216, -0.14309008},
      {-0.035518106, -0.051089376, 0.044176966},
    },
    {
      {-2.0536497, -0.96356624, 0.48925233},
      {-1.1527416, -0.07979518, -0.6044105},
      {-0.6472761, -0.027735198, 0.08825184},
    },
  },
  {
    {
      {-0.15738341, -0.076025285, -0.06489653},
      {0.03258573, -0.15222852, 0.043187983},
      {-0.12316223, -0.0052017383, -0.017146},
    },
    {
      {-0.062303174, -0.10356044, -0.11602676},
      {-0.13527513, 0.011797731, -0.122232035},
      {-0.10942069, -0.032494813, 0.03079278},
    },
    {
      {0.0052292976, -0.17410545, 0.053433355},
      {0.096899405, 0.029638853, -0.019923812},
      {-0.030619325, -0.112685025, -0.050641358},
    },
    {
      {-0.081270166, -0.10985796, -0.0016012405},
      {-0.040066227, 0.04222485, 0.020818384},
      {0.004056797, -0.13934526, -0.067939974},
    },
    {
      {-0.1741076, -0.00092461647, -0.12686554},
      {-0.09102721, -0.020163713, -0.057735126},
      {-0.08096379, -0.1126783, -0.12624429},
    },
    {
      {-0.02882789, -0.09378701, -0.019519301},
      {-0.06593195, -0.027021827, -0.063319676},
      {-0.09896382, 0.0037728748, -0.16462377},
    },
    {
      {-0.012718575, 0.08602142, -0.12099323},
      {-0.14686182, -0.055027395, 0.06163322},
      {0.007905651, 0.013247843, 0.029217424},
    },
    {
      {-0.09557627, -0.023922672, 0.017475313},
      {-0.028070217, -0.12514211, 0.01931844},
      {-0.044420883, -0.17348658, -0.022996537},
    },
  },
  {
    {
      {-0.016306281, -0.070811726, 0.039067425},
      {-0.007073134, -0.14281204, 0.032686114},
      {-0.093916975, -0.07784844, -0.073179595},
    },
    {
      {0.18334572, -0.18675637, 0.105644},
      {-0.3416585, 0.4116987, -0.2674309},
      {-0.3894028, 0.21331984, 0.6038931},
    },
    {
      {0.35055596, 0.34903714, 0.25026605},
      {-0.52647936, -0.8210941, 0.6108414},
      {-1.5658861, 0.7263749, -0.10412867},
    },
    {
      {0.51517916, 0.37593144, 1.0698744},
      {0.23653473, -0.03942847, -0.57895714},
      {-1.7162212, -3.0515585, -2.8691144},
    },
    {
      {-1.1827091, -1.5018979, -0.53740287},
      {0.02687646, -0.022881651, 0.060434636},
      {-0.13016441, 0.46200874, 0.40879813},
    },
    {
      {0.572142, -0.023886658, 0.82482475},
      {-0.24907796, -0.5570497, -0.36160973},
      {-1.4069473, -0.6424684, -1.4625928},
    },
    {
      {0.04766226, 0.15578021, -0.06829595},
      {-0.13073742, -0.041143704, -0.05103979},
      {0.089181826, 0.116049804, -0.02654643},
    },
    {
      {0.038045857, 0.5941791, 0.37435624},
      {-2.0525658, -0.62554586, 0.4238389},
      {0.020667728, 0.5725068, 0.85469365},
    },
  },
  {
    {
      {-0.13772863, -0.11543185, 0.017735587},
      {-0.06646856, -0.0273007, -0.18193837},
      {-0.15762125, 0.010125359, 0.00452571},
    },
    {
      {-2.0478806, -0.95917434, 0.68993556},
      {-0.5377497, 0.23030567, -1.6521256},
      {-0.5445247, 1.2347009, -0.8380385},
    },
    {
      {-1.3674693, -0.5460718, -0.6216194},
      {-1.535619, 0.28110608, 0.81228286},
      {-1.3787482, -0.58884376, -0.046455324},
    },
    {
      {-1.6346948, 0.058840383, 1.029506},
      {-0.88267034, -0.18755561, 0.8660854},
      {0.7127874, -0.50421864, -0.64450586},
    },
    {
      {-2.4115052, -0.75747377, -0.9594221},
      {-2.051875, -0.06352762, -0.73765796},
      {-0.17519164, 0.4500408, 0.73005885},
    },
    {
      {-3.3392792, -1.8783273, 0.6314611},
      {-0.65486246, 0.4302962, 0.45189276},
      {0.14788422, 0.32806394, -0.7020606},
    },
    {
      {0.08020976, -0.012542159, 0.10605225},
      {-0.060020003, 0.09260063, -0.045804854},
      {-0.1356939, -0.15614887, 0.037295025},
    },
    {
      {-1.1921823, -0.16788375, -1.3687866},
      {0.4945951, -0.8019217, -0.7741072},
      {0.54034084, 0.39985076, -0.23538391},
    },
  },
};

// layer: conv1.9.bias shape: (10,)
const float layer_conv1_9_bias[10] = 
{1.635937, 1.3585476, -1.5044124, -0.043877054, -0.24402545, -0.27277052, -0.65751183, -0.16072558, 1.3628707, -0.4504084};

// layer: conv1.9.weight shape: (10, 16, 1, 1)
const float layer_conv1_9_weight[10][16][1][1] = 
{
  {
    {
      {-0.0007726809},
    },
    {
      {-0.6890479},
    },
    {
      {0.29504982},
    },
    {
      {-0.70021725},
    },
    {
      {-2.841937},
    },
    {
      {-0.008525819},
    },
    {
      {0.08660539},
    },
    {
      {-0.041469384},
    },
    {
      {1.0988278},
    },
    {
      {-0.166131},
    },
    {
      {0.20378353},
    },
    {
      {-0.91974556},
    },
    {
      {0.012608927},
    },
    {
      {-0.011716407},
    },
    {
      {0.7442016},
    },
    {
      {-0.33607802},
    },
  },
  {
    {
      {0.1400464},
    },
    {
      {-1.0834117},
    },
    {
      {0.7334597},
    },
    {
      {-0.56177956},
    },
    {
      {0.3417223},
    },
    {
      {-1.2813773},
    },
    {
      {-1.2437397},
    },
    {
      {-0.04732399},
    },
    {
      {-1.162658},
    },
    {
      {0.25098756},
    },
    {
      {0.8453641},
    },
    {
      {0.60859025},
    },
    {
      {-0.5292112},
    },
    {
      {-0.11600442},
    },
    {
      {0.10682236},
    },
    {
      {1.027645},
    },
  },
  {
    {
      {-0.058562152},
    },
    {
      {-0.319459},
    },
    {
      {0.29035306},
    },
    {
      {0.54098177},
    },
    {
      {0.23866813},
    },
    {
      {-0.24226514},
    },
    {
      {-0.51045954},
    },
    {
      {0.052126355},
    },
    {
      {-0.6627958},
    },
    {
      {0.757708},
    },
    {
      {-0.21600774},
    },
    {
      {-0.5315207},
    },
    {
      {-0.25956073},
    },
    {
      {-0.16313884},
    },
    {
      {0.666272},
    },
    {
      {-0.46292982},
    },
  },
  {
    {
      {-1.013294},
    },
    {
      {0.8307728},
    },
    {
      {-0.13651977},
    },
    {
      {0.7330992},
    },
    {
      {0.92328537},
    },
    {
      {-0.3893416},
    },
    {
      {-0.69720787},
    },
    {
      {0.04806419},
    },
    {
      {0.2143122},
    },
    {
      {0.040386498},
    },
    {
      {-0.5842199},
    },
    {
      {-1.0547205},
    },
    {
      {-0.71835834},
    },
    {
      {0.17894901},
    },
    {
      {-0.19145612},
    },
    {
      {0.4794652},
    },
  },
  {
    {
      {0.41509053},
    },
    {
      {-1.7489077},
    },
    {
      {-0.23519675},
    },
    {
      {0.35086876},
    },
    {
      {-0.06343731},
    },
    {
      {-0.26303846},
    },
    {
      {-0.23683049},
    },
    {
      {-0.24107099},
    },
    {
      {-0.51353794},
    },
    {
      {-0.050534915},
    },
    {
      {-0.3144966},
    },
    {
      {0.79175},
    },
    {
      {1.0585338},
    },
    {
      {-0.18500565},
    },
    {
      {-0.58291626},
    },
    {
      {-0.91867036},
    },
  },
  {
    {
      {-0.39109972},
    },
    {
      {0.63016313},
    },
    {
      {-0.1702778},
    },
    {
      {-0.339324},
    },
    {
      {0.4542387},
    },
    {
      {-0.068264626},
    },
    {
      {1.024609},
    },
    {
      {-0.18740849},
    },
    {
      {0.33042562},
    },
    {
      {-0.7404849},
    },
    {
      {-0.34017485},
    },
    {
      {-0.9127901},
    },
    {
      {0.19544968},
    },
    {
      {0.14405927},
    },
    {
      {0.18968241},
    },
    {
      {0.42310935},
    },
  },
  {
    {
      {-1.0155139},
    },
    {
      {0.6951837},
    },
    {
      {-0.48185852},
    },
    {
      {-0.4805641},
    },
    {
      {-1.6585269},
    },
    {
      {0.28114715},
    },
    {
      {0.008851547},
    },
    {
      {0.016964577},
    },
    {
      {-0.025230207},
    },
    {
      {-0.8492932},
    },
    {
      {0.44519478},
    },
    {
      {-0.59820104},
    },
    {
      {0.8400524},
    },
    {
      {0.035781257},
    },
    {
      {0.96038234},
    },
    {
      {-0.19423965},
    },
  },
  {
    {
      {0.4515879},
    },
    {
      {-0.6244088},
    },
    {
      {0.47349435},
    },
    {
      {0.57136387},
    },
    {
      {0.029510131},
    },
    {
      {-1.228337},
    },
    {
      {0.34296843},
    },
    {
      {-0.059977453},
    },
    {
      {-0.20913304},
    },
    {
      {0.4561087},
    },
    {
      {0.8905934},
    },
    {
      {0.51868844},
    },
    {
      {-0.8247084},
    },
    {
      {0.0021572807},
    },
    {
      {-0.8042256},
    },
    {
      {-0.9308715},
    },
  },
  {
    {
      {-0.85091394},
    },
    {
      {-0.08760379},
    },
    {
      {-0.56777215},
    },
    {
      {-0.06323898},
    },
    {
      {-0.38923463},
    },
    {
      {0.87345713},
    },
    {
      {0.087282404},
    },
    {
      {0.061110336},
    },
    {
      {-0.3442349},
    },
    {
      {0.44703183},
    },
    {
      {-0.45665473},
    },
    {
      {-0.5341861},
    },
    {
      {-0.23465094},
    },
    {
      {-0.080665044},
    },
    {
      {-0.5714847},
    },
    {
      {0.41440156},
    },
  },
  {
    {
      {0.50543624},
    },
    {
      {-0.09093448},
    },
    {
      {-0.9897013},
    },
    {
      {-0.018727234},
    },
    {
      {0.11395633},
    },
    {
      {0.56907284},
    },
    {
      {-0.02592644},
    },
    {
      {0.18902512},
    },
    {
      {0.63482255},
    },
    {
      {-0.30504087},
    },
    {
      {-0.17763911},
    },
    {
      {0.95896226},
    },
    {
      {-0.10771362},
    },
    {
      {-0.20984383},
    },
    {
      {-0.48080954},
    },
    {
      {-0.89393204},
    },
  },
};

float buf1[3136];
float buf2[3136];


//node

// 1x28x28 => 4x28x28
void op_Conv_0(void* in, void* out)
{
  float (*i)[1][28][28];
  i = (typeof(i))(in);
  float (*o)[4][28][28];
  o = (typeof(o))(out);
  
  {
    float *p = (float*)((*o));
    for(int c=0;c < 4;c++) {
      int cnt=0;
      while(cnt++ < 28*28) {
        *p++ = layer_conv1_0_bias[c];
      }
    }
  }
  
  for(int c_i=0;c_i < 1;c_i++) {
    for(int m=-1;m <= 1;m++) {
      for(int n=-1;n <= 1;n++) {
        for(int o_c=0;o_c < 4 ;o_c++) {
          float t = layer_conv1_0_weight[o_c][c_i][m-(-1)][n-(-1)];
          if(float_IS_ZERO(t))
            continue;
          for(int o_x=(m>=0?0:-m);o_x < 28 - (m<0?0:m) ;o_x += 1) {
            for(int o_y=(n>=0?0:-n);o_y < 28 - (n<0?0:n) ;o_y += 1) {
                (*o)[o_c][o_x][o_y] += (*i)[c_i][o_x+m][o_y+n] * t;
            } // o_y
          } // o_x
        } // o_c
      } // n
    } // m
  } // c_i
}


//node

void op_Relu_1(void* in, void* out)
{
  float* p = (float*)in;
  int i = 0;
  while(i++ < 3136) {
    if(*p<0)*p=0;
    p++;
  }
}


//node

void op_MaxPool_2(void* in, void* out)
{
  float (*i)[4][28][28];
  i = (typeof(i))(in);
  float (*o)[4][14][14];
  o = (typeof(o))(out);

  for(int c=0;c<4;c++) {
    for(int x=0, o_i=0;x<28;x+=2) {
      for(int y=0, o_j=0;y<28;y+=2) {
        float max=-FLT_MIN;
        for(int m=0;m<2;m++) {
          for(int n=0;n<2;n++) {
            if(max < (*i)[c][x+m][y+n]) {
              max = (*i)[c][x+m][y+n];
            }
          }
        }
        (*o)[c][o_i][o_j] = max;
        o_j++;
      }
      o_i++;
    }
  }
}


//node

// 4x14x14 => 8x12x12
void op_Conv_3(void* in, void* out)
{
  float (*i)[4][14][14];
  i = (typeof(i))(in);
  float (*o)[8][12][12];
  o = (typeof(o))(out);

  {
    float *p = (float*)((*o));
    for(int c=0;c < 8;c++) {
      int cnt=0;
      while(cnt++ < 12*12) {
        *p++ = layer_conv1_3_bias[c];
      }
    }
  }

  for(int c_i=0;c_i < 4;c_i++) {
    for(int m=0;m < 3;m++) {
      for(int n=0;n < 3;n++) {
        for(int c=0;c<8;c++) {
          float t = layer_conv1_3_weight[c][c_i][m][n];
          if(float_IS_ZERO(t))
            continue;
          for(int x=0;x<12;x++) {
            for(int y=0;y<12;y++) {
              (*o)[c][x][y] += (*i)[c_i][x+m][y+n] * t;
            }
          }
        }
      }
    }
  }
}


//node

void op_Relu_4(void* in, void* out)
{
  float* p = (float*)in;
  int i = 0;
  while(i++ < 1152) {
    if(*p<0)*p=0;
    p++;
  }
}


//node

void op_MaxPool_5(void* in, void* out)
{
  float (*i)[8][12][12];
  i = (typeof(i))(in);
  float (*o)[8][6][6];
  o = (typeof(o))(out);

  for(int c=0;c<8;c++) {
    for(int x=0, o_i=0;x<12;x+=2) {
      for(int y=0, o_j=0;y<12;y+=2) {
        float max=-FLT_MIN;
        for(int m=0;m<2;m++) {
          for(int n=0;n<2;n++) {
            if(max < (*i)[c][x+m][y+n]) {
              max = (*i)[c][x+m][y+n];
            }
          }
        }
        (*o)[c][o_i][o_j] = max;
        o_j++;
      }
      o_i++;
    }
  }
}


//node

// 8x6x6 => 16x4x4
void op_Conv_6(void* in, void* out)
{
  float (*i)[8][6][6];
  i = (typeof(i))(in);
  float (*o)[16][4][4];
  o = (typeof(o))(out);

  {
    float *p = (float*)((*o));
    for(int c=0;c < 16;c++) {
      int cnt=0;
      while(cnt++ < 4*4) {
        *p++ = layer_conv1_6_bias[c];
      }
    }
  }

  for(int c_i=0;c_i < 8;c_i++) {
    for(int m=0;m < 3;m++) {
      for(int n=0;n < 3;n++) {
        for(int c=0;c<16;c++) {
          float t = layer_conv1_6_weight[c][c_i][m][n];
          if(float_IS_ZERO(t))
            continue;
          for(int x=0;x<4;x++) {
            for(int y=0;y<4;y++) {
              (*o)[c][x][y] += (*i)[c_i][x+m][y+n] * t;
            }
          }
        }
      }
    }
  }
}


//node

void op_Relu_7(void* in, void* out)
{
  float* p = (float*)in;
  int i = 0;
  while(i++ < 256) {
    if(*p<0)*p=0;
    p++;
  }
}


//node

void op_MaxPool_8(void* in, void* out)
{
  float (*i)[16][4][4];
  i = (typeof(i))(in);
  float (*o)[16][2][2];
  o = (typeof(o))(out);

  for(int c=0;c<16;c++) {
    for(int x=0, o_i=0;x<4;x+=2) {
      for(int y=0, o_j=0;y<4;y+=2) {
        float max=-FLT_MIN;
        for(int m=0;m<2;m++) {
          for(int n=0;n<2;n++) {
            if(max < (*i)[c][x+m][y+n]) {
              max = (*i)[c][x+m][y+n];
            }
          }
        }
        (*o)[c][o_i][o_j] = max;
        o_j++;
      }
      o_i++;
    }
  }
}


//node

// 16x2x2 => 10x2x2
void op_Conv_9(void* in, void* out)
{
  float (*i)[16][2][2];
  i = (typeof(i))(in);
  float (*o)[10][2][2];
  o = (typeof(o))(out);

  {
    float *p = (float*)((*o));
    for(int c=0;c < 10;c++) {
      int cnt=0;
      while(cnt++ < 2*2) {
        *p++ = layer_conv1_9_bias[c];
      }
    }
  }

  for(int c_i=0;c_i < 16;c_i++) {
    for(int m=0;m < 1;m++) {
      for(int n=0;n < 1;n++) {
        for(int c=0;c<10;c++) {
          float t = layer_conv1_9_weight[c][c_i][m][n];
          if(float_IS_ZERO(t))
            continue;
          for(int x=0;x<2;x++) {
            for(int y=0;y<2;y++) {
              (*o)[c][x][y] += (*i)[c_i][x+m][y+n] * t;
            }
          }
        }
      }
    }
  }
}


//node

void op_AveragePool_11(void* in, void* out)
{
  float (*i)[10][2][2];
  i = (typeof(i))(in);
  float (*o)[10][1][1];
  o = (typeof(o))(out);

  for(int c=0;c<10;c++) {
    for(int x=0, o_i=0;x<2;x+=2) {
      for(int y=0, o_j=0;y<2;y+=2) {
        float result=0;
        for(int m=0;m<2;m++) {
          for(int n=0;n<2;n++) {
              result += (*i)[c][x+m][y+n];
          }
        }
        (*o)[c][o_i][o_j] = result/(2*2);
        o_j++;
      }
      o_i++;
    }
  }
}


void Model(void* input, void* output)
{
  op_Conv_0(input, buf1);
  op_Relu_1(buf1, NULL);
  op_MaxPool_2(buf1, buf2);
  op_Conv_3(buf2, buf1);
  op_Relu_4(buf1, NULL);
  op_MaxPool_5(buf1, buf2);
  op_Conv_6(buf2, buf1);
  op_Relu_7(buf1, NULL);
  op_MaxPool_8(buf1, buf2);
  op_Conv_9(buf2, buf1);
  op_AveragePool_11(buf1, output);
}

