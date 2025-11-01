from __future__ import division
import random
import math
from numpy import *
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd


from RFBNN_core import *
from PSO_RFBNN import *
from Training_Data import *
from Py_SARIMAX import *
from LSTM import *


import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


####### STARTING TRAINING CODE #########


Mins = [1995.00, -14.50, 70246000000.00, 53980.22, 64841710004.00, 15559900000.00, -46324000000.00]
Maxes = [2023.00, 8.90, 663000000000.00, 531175.00, 615433000000.00, 202128000000.00, 81883000000.00]

LSTMs = [] # Loading will happen later
windowSize = 5

ARIMA_coefficient = 0.0
RFBNN_coefficient = 0.0
LSTM_coefficient = 1.0

#### Trained models are stored here in their exported formats


def denorm(x):
    for i in range(0, len(x)):
        minV = Mins[i]
        maxV = Maxes[i]
        x[i] = (((x[i]+1.0)/2.0) * (maxV - minV)) + minV

    return x

def norm(x):
    for i in range(0, len(x)):
        minV = Mins[i]
        maxV = Maxes[i]
        x[i] = (((x[i]-minV)/(maxV-minV))*2.0)-1.0

    return x


FutureModelsEx = """[[12, [48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193, 48.27092036743193], [array([ 23.98334632, -11.02537892, -10.47065505, -10.13523447,
       -19.15983999, -24.33625693]), array([-26.98654483,   7.50225617,  12.71066261,   7.20453779,
        20.61777008, -21.10341547]), array([ 20.64296499,  14.47109804,  19.40868026,  14.56144517,
         5.9744765 , -24.66776722]), array([ 11.22053371, -20.12390804, -20.58066805, -19.99071517,
       -21.02897951,  -8.49633592]), array([ 22.6823443 ,   1.32016194,   6.37940933,   2.47320941,
       -12.11948428, -34.03738016]), array([ 14.88331827, -15.3493618 , -15.66794131, -15.98745317,
       -17.58029647,  24.51946997]), array([ 20.15224954, -18.12717066, -18.71110167, -17.78223393,
       -19.40191909,  -9.54019049]), array([ 21.40628164,  14.77103986,  17.89565145,  14.6415295 ,
        15.75498376, -20.22892961]), array([ 14.66725166,  12.54801627,  18.05643961,  12.41685847,
        16.88840965, -27.02697903]), array([ 24.42408787,   9.90726751,  14.99193348,  11.46616101,
        -0.93042588, -28.57767866]), array([ 15.88242142, -17.2586389 , -16.65092568, -17.38762978,
       -21.3580695 , -16.81095745]), array([ 22.85664095,  -7.54855244,  -4.43625418,  -6.71382079,
       -15.00390227, -31.6099801 ])], array([ 5.29652011e+10,  3.12715945e+02,  5.40889281e+06, -1.55646390e+11,
       -3.18164996e+03,  3.36881513e+02,  1.77360188e+11,  3.89695052e+06,
       -9.25904461e+06,  4.14204439e+03, -7.40402779e+10, -6.52157485e+08])], [18, [75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587, 75.551158514587], [array([-10.26386258, -13.73005234, -14.12760656, -13.27343318,
       -14.39806508,  -4.8022093 ]), array([  5.36591147,  -2.60057936,   0.57116145,  -1.50337752,
       -10.73174749, -27.3255779 ]), array([ 15.69420251,   6.94112861,  10.74970464,   6.86693901,
        10.36520216, -18.31295509]), array([ -6.3253537 , -11.09164201, -11.32185158, -11.55273487,
       -12.70374348,  17.71807759]), array([ 8.61226355e-16, -7.54023296e+00, -5.69898373e+00, -6.81614838e+00,
       -1.38700350e+01, -2.39129405e+01]), array([ 15.25239247,   9.58974247,  13.00860794,   9.91031084,
         3.84631088, -17.12218194]), array([ -6.39256635, -13.25388667, -12.60631966, -13.61199843,
       -15.44493377, -10.03582621]), array([ 14.72804407,   7.26037142,  10.98658185,   8.40277984,
        -0.68184668, -20.94266268]), array([ 11.33199247,   3.58757076,   7.55466165,   4.47513973,
        -5.64992748, -25.48909206]), array([ 15.9579082 ,   5.64414317,   9.56256331,   5.42016186,
        15.51128668, -15.87665039]), array([ 14.03957589,  10.42817018,  12.63410704,  10.33673748,
        11.12282232, -14.28137237]), array([ -3.93194568, -10.3660084 , -10.55851147,  -9.67898291,
       -16.99271013, -16.82761423]), array([ -8.53922905, -13.30978148, -13.7785838 , -13.3835564 ,
       -14.41282877,  -8.55761051]), array([ 13.38953683,   9.20817261,  12.47064705,   9.11360142,
        11.34561038, -16.39000357]), array([-11.12652486, -13.60429752, -13.69778061, -13.61963454,
       -13.86903953,  -5.06454709]), array([ 14.57967328,  10.20274268,  13.5593229 ,  10.04514666,
         4.30807168, -16.73372279]), array([  9.9564304 ,   1.21750937,   5.57003099,   1.32215568,
       -10.23521933, -25.7265254 ]), array([ -5.22489296, -11.84110537, -11.62172453, -11.64663467,
       -15.69922541, -14.67359732])], array([-1.68777068e+12,  7.25913048e+02,  1.73087053e+11,  1.52478586e+05,
       -2.74054345e+02,  1.56624572e+11,  2.37658758e+12, -3.16392390e+02,
        6.69544940e+04, -2.59311541e+11,  2.87834748e+11,  1.78356183e+11,
       -1.25766507e+12, -4.67614741e+11,  1.14911017e+12,  1.09381013e+11,
       -6.92328449e+04, -7.58618829e+11])], [18, [79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9], [array([ 30.61535461,  18.93364988,  20.96991034,  13.39563273,
        20.21984485, -35.72386769]), array([ -8.44174293,  31.92255728, -20.3358513 , -19.60359272,
       -29.81825107, -28.68294826]), array([-22.70365767,  17.22651068, -30.01829913, -29.27538429,
       -30.35373292, -10.40652428]), array([ 18.38964793,  31.28814127,  11.38270587,   5.11829111,
       -13.49914641, -44.10083008]), array([ 26.75864532, -34.04407802,  16.03476075,   9.08867171,
        26.00973846, -26.62239005]), array([-12.70273444,  21.59820674, -22.7368272 , -23.20049283,
       -25.51197729,  35.5818892 ]), array([ 28.22797968,  26.83285712,  25.22846619,  18.92776439,
         7.76595195, -32.06451561]), array([  3.90467828,  33.82221329,  -7.03942   ,  -9.12926795,
       -19.90885732, -43.72129697]), array([ 26.38038754,  28.39654202,  23.73950913,  19.42274772,
        20.8998025 , -26.83472354]), array([-13.62906872,  17.53670381, -26.87690475, -29.02103033,
       -32.92888211, -21.39656556]), array([  9.36971927,  29.60516339,   0.99733707,  -2.62513189,
       -18.73930677, -47.71472566]), array([-20.62584227,  30.15286428, -27.29796543, -25.24508669,
       -28.06764224,  -9.75159203]), array([-16.01902954,  28.07997719, -26.88508455, -26.36265176,
       -28.15493204, -18.55823795]), array([ 25.86064774,  31.42806384,  19.29109676,  14.7542558 ,
        -1.19723955, -36.7727595 ]), array([ -3.38057345,  33.37721114, -11.37967684, -11.95246827,
       -24.85432643, -39.80499233]), array([  0.        ,  30.7052191 , -10.8440697 , -14.26121914,
       -26.93341194, -40.13483054]), array([ 26.7997076 ,  18.27935238,  24.96051198,  18.24124734,
        22.7087049 , -32.80526494]), array([-19.31796838,  14.52711222, -30.09739473, -28.9775421 ,
       -31.44939961, -16.76792345])], array([ 1.34083787e+11, -1.26319713e+10,  1.29012182e+09, -1.07117499e+04,
       -3.83538579e+02, -3.82834973e+02, -9.62593989e+10,  2.08707183e+02,
        3.80399399e+10,  6.43839151e+09, -1.32046078e+01,  6.27392088e+09,
       -8.10410398e+09,  1.14421583e+04,  3.90232859e+08,  2.40848529e+01,
       -7.58581891e+10,  6.33889831e+09])], [8, [30.608065567693146, 30.608065567693146, 30.608065567693146, 30.608065567693146, 30.608065567693146, 30.608065567693146, 30.608065567693146, 30.608065567693146], [array([-10.57587938,  11.26440355, -14.60328976, -14.43671811,
       -15.40287887,  -6.75196973]), array([ 12.68444308,  16.54955803,   5.71164444,   6.39249882,
        -2.19605684, -20.74020564]), array([  1.95082143,  16.20220042,  -4.92917699,  -4.14772617,
       -11.22954085, -22.56831137]), array([ 14.02731605, -17.84645808,   4.96131317,   4.76442918,
        13.63472692, -13.95588883]), array([ 15.36954096,  11.09328204,   9.16356872,   9.05218733,
         9.44808647, -18.13244511]), array([ -6.52105563,  11.0876212 , -11.43480951, -11.91016825,
       -13.09678825,  18.26626227]), array([ -5.25497429,  14.04050674, -11.89075687, -11.752398  ,
       -15.95295202, -13.58280039]), array([ 13.85341881,  14.91218386,  10.28989836,  10.199678  ,
        10.97533979, -14.0920092 ])], array([ 9847.8319578 ,    49.84840112,    67.83921217,    41.27025401,
       -2133.96441925,    36.94343728, -9640.70365469,  2359.07049005])], [11, [17.93654249047897, 17.93654249047897, 17.93654249047897, 17.93654249047897, 17.93654249047897, 17.93654249047897, 17.93654249047897, 17.93654249047897, 17.93654249047897, 17.93654249047897, 17.93654249047897], [array([ 13.03055278,  15.83583864,   6.4235721 ,   9.72031548,
        -0.60325995, -18.52890107]), array([ -5.94802194,  11.80087552, -12.82342559, -12.37188563,
       -15.86936355, -12.49079161]), array([ 14.40134921,   9.32283396,   7.97580044,  11.47707779,
        10.73465177, -17.17895375]), array([ -6.38884198,  10.86282095, -11.20297006, -11.43549026,
       -12.83125237,  17.8959159 ]), array([ 13.36639309, -17.00558918,   4.72755172,   8.0096325 ,
        12.99230153, -13.2983313 ]), array([ -0.93450074,  16.74912862,  -7.2277951 ,  -6.11019714,
       -12.80209148, -19.10822654]), array([-10.20635125,   9.69465071, -14.35055431, -14.74250736,
       -15.14811508,  -7.00705494]), array([ 13.17793072,  14.18507074,   9.78816633,  11.85871914,
        10.44018587, -13.40488752]), array([ 14.1226856 ,  13.42469455,   9.41095773,  12.62200484,
         3.88536831, -16.04213543]), array([  6.91453532,  15.36409748,  -0.17303219,   3.0147374 ,
        -8.20061821, -23.2328429 ]), array([-10.20719688,  14.92187413, -13.20745766, -13.50905839,
       -13.88995157,  -4.82581115])], array([ 7.38359365e+00, -7.71828393e+09,  8.94238253e+04, -1.36157790e+03,
       -1.42800892e+03,  2.82006056e+06,  9.62685535e+09,  7.29493877e+04,
       -1.04345147e+05,  4.34129135e+02, -1.91945022e+09])], [18, [22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472, 22.077016507918472], [array([ 13.67578428,  13.72368545,   8.59847067,  11.6639351 ,
         8.88590255, -15.35229748]), array([ -9.84014348,   7.3997879 , -14.91817163, -15.33094354,
       -14.76051552,  -8.54120731]), array([  1.07302252,  17.16836038,  -6.80103147,  -4.71525391,
        -6.1658273 , -22.30674673]), array([ 14.62450212, -18.60623679,   5.17253156,   8.76353753,
         4.96726561, -14.55003402]), array([ -6.49240184,  11.03890173, -11.38456448, -11.62085371,
       -11.85783447,  18.18599955]), array([ 13.28810273,  14.30366278,   9.86999876,  11.95786209,
         9.78346002, -13.51695697]), array([ -3.71214303,  18.84626461,  -9.78653038,  -9.96827219,
        -9.13791082, -15.88692113]), array([  9.41212751,  16.01379081,   2.16632623,   5.82585808,
         2.61962647, -22.57153794]), array([-10.42281798,  15.23709005, -13.48645754, -13.7944294 ,
       -12.75705205,  -4.92775361]), array([ 13.62743407,  10.72520235,   9.44976955,  12.68345559,
         9.32932785, -16.18822186]), array([ -7.10230971,   9.13863627, -14.72541743, -14.00595341,
       -15.12328903, -11.15006743]), array([ -8.15690667,  14.29835387, -13.1189664 , -13.68991328,
       -13.42388996,  -9.44987425]), array([ 12.65039238,  15.3738353 ,   6.23616733,   9.43672973,
         7.2174188 , -17.98832887]), array([-11.44112884,   8.68101216, -14.82430205, -15.12722016,
       -14.75284064,  -5.24419399]), array([ 15.81087433,   9.7780203 ,   6.99272945,  10.82961871,
         6.91798832, -18.4490949 ]), array([  4.91597732,  15.53283591,  -2.38251959,   0.5232693 ,
        -1.37731862, -25.03431562]), array([ -1.80449034,  17.81616516,  -7.09106141,  -6.07427029,
        -6.38001623, -21.24720111]), array([ -5.26706458,  15.35816963, -11.93667836, -11.71552683,
       -11.74063802, -14.79203217])], array([-2.22564904e+13, -2.47433713e+13,  2.13800143e+02, -9.04114153e+01,
       -1.92899369e+02, -2.17846420e+12, -8.23512363e+11,  7.18998696e+07,
       -4.71637156e+13,  2.40878938e+13,  7.13324257e+13, -1.27212976e+13,
        3.00722220e+12,  3.54262957e+13, -2.66049414e+12, -7.79626697e+01,
        4.80312485e+09, -2.13126447e+13])], [8, [73.87853533042122, 73.87853533042122, 73.87853533042122, 73.87853533042122, 73.87853533042122, 73.87853533042122, 73.87853533042122, 73.87853533042122], [array([-19.70988343,  18.59775244, -26.42329769, -27.02397114,
       -25.9278299 , -27.65693509]), array([33.45298355, 21.65606892, 18.52703638, 26.66017531, 18.33338304,
       24.93558928]), array([ 12.24803135,  47.92857085, -10.08761708,  -1.95556803,
        -8.06284317, -31.34077408]), array([ 29.4067799 , -37.41320598,  10.4008667 ,  17.62162002,
         9.9881203 ,  28.58375845]), array([31.5932782 , 35.97842522, 17.43897278, 25.47681954, 18.76680063,
        1.42785844]), array([ -7.65796388,  36.69739234, -20.41511561, -19.66379043,
       -19.32495892, -31.90237213]), array([27.70561055, 29.82304688, 20.57888529, 24.93206719, 20.39845258,
       21.94970744]), array([-14.3762109 ,  23.16018305, -25.60026062, -25.78866937,
       -26.37918546, -28.99379663])], array([ 1.30966325e+06, -1.51156989e+06,  8.88041887e+01, -4.45839373e+02,
        1.47484488e+05,  1.13211793e+05,  1.37413833e+06, -1.41659953e+06])]]"""
MainModelEx = """[8, [34.20901903168867, 34.20901903168867, 34.20901903168867, 34.20901903168867, 34.20901903168867, 34.20901903168867, 34.20901903168867, 34.20901903168867], [array([ -8.30841013, -18.79995586, -18.39996271, -18.5812027 ,
       -25.22251503, -21.47517191]), array([ 23.38011177,  14.78585901,  20.1063505 ,  14.88222789,
         4.80132463, -28.09975098]), array([  3.48984596,  -8.81785903,  -5.01482067,  -7.41991305,
       -20.08864938, -40.37270094]), array([ 21.46827765,  18.80875936,  20.4332596 ,  18.74159613,
        19.32042255, -15.71449615]), array([ -9.89770434, -17.35583471, -17.71605903, -18.07733757,
       -19.87839779,  27.72466202]), array([-15.38719234, -21.24680325, -21.80704998, -21.00445272,
       -22.41015157,  -9.82366129]), array([ 23.63628458,  11.51284644,  17.27845496,  11.31101701,
        19.40454545, -26.63467544]), array([ 19.04290651,   5.87946885,  12.47026154,   7.67385748,
        -9.46827999, -38.76082732])], array([ 3.76992969e+05, -1.97860786e+04, -2.48700509e+02,  8.81092930e+04,
       -3.22783836e+02, -4.05047098e+05, -3.29279337e+04,  5.86185949e+02])]"""
GrowthModelEx = """[18, [79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9, 79.9], [array([  8.48591249,  15.72564202,   4.74736363,   1.12687951,
        -8.7235256 , -21.92683868]), array([ 13.3998538 ,   9.13967619,  12.48025599,   9.12062367,
        11.35435245, -16.40263247]), array([ -6.81453436,   8.7683519 , -13.43845238, -14.51051517,
       -16.46444105, -10.69828278]), array([ 13.37932266, -17.02203901,   8.01738037,   4.54433585,
        13.00486923, -13.31119503]), array([ -6.35136722,  10.79910337, -11.3684136 , -11.60024642,
       -12.75598865,  17.7909446 ]), array([ -0.93586472,  16.10741814,  -5.57564474,  -6.49847952,
       -12.90438939, -19.99584762]), array([-11.35182884,   8.61325534, -15.00914957, -14.63769215,
       -15.17686646,  -5.20326214]), array([13.524515  , 11.33354357, 13.524515  , 13.52472137, 13.52464398,
       -6.3159873 ]), array([ 13.19019377,  14.19827101,  11.86975456,   9.71137386,
        10.44990125, -13.41736177]), array([ 14.11398984,  13.41642856,  12.61423309,   9.46388219,
         3.88297598, -16.03225781]), array([ 12.93032387,  15.71403192,   9.64554838,   7.3771279 ,
        -0.59861977, -18.38637975]), array([-10.31292113,  15.07643214, -13.64898271, -12.62254334,
       -14.03382112,  -4.87579601]), array([ -4.22087146,  15.96127864, -10.16792565,  -9.80179636,
       -14.90912553, -14.34147413]), array([ -8.00951477,  14.0399886 , -13.44254227, -13.18132588,
       -14.07746602,  -9.27911898]), array([  3.82209747,  15.56522404,  -0.80371208,  -2.37560976,
        -9.60886045, -23.33443159]), array([ 15.30767731,   9.46682494,  10.48495517,   6.69781637,
        10.10992242, -17.86193385]), array([  9.78272554,  15.4550622 ,   6.52181703,   3.86331564,
        -4.87749088, -22.00432028]), array([ -9.65898419,   7.26355611, -15.04869737, -14.48877105,
       -15.7246998 ,  -8.38396172])], array([-9.27731377e+03,  1.99174744e+10, -2.80419237e+09,  5.05284169e+02,
        5.10263975e+02, -1.78929699e+08, -5.58581414e+07,  8.72294093e+09,
       -1.33812319e+10,  3.98963996e+10,  2.85832061e+01, -2.52084330e+09,
        4.98260306e+09,  3.16038504e+09, -2.62909168e+01, -5.51555845e+10,
        9.15429921e+03, -2.58317391e+09])]"""
ARIMAModelsEx = """[(2, 1, 2), (0, 2, 0), (2, 2, 1), (2, 0, 1), (0, 2, 1), (0, 2, 0), (1, 1, 2)]"""

####################################### BELOW not to be modified

FutureModels = [importRFBN(x) for x in eval(FutureModelsEx)]

if len(MainModelEx) + len(GrowthModelEx) != 0:
    MainModel = importRFBN(eval(MainModelEx))
    GrowthModel = importRFBN(eval(GrowthModelEx))
else:
    MainModel = None
    GrowthModel = None

ARIMAModels = eval(ARIMAModelsEx)

#######################################

def randX0(particles, length):
    x0list = []
    for i in range(0, particles):
      x0 = []
      for _ in range(0, length):
        x0.append(random.random())
      x0list.append(x0)
    return x0list

def fitnessFunction(position, keys, values):
    xtrain, xtest, ytrain, ytest = train_test_split(keys, values, train_size=0.90)
    model = fitness_model(position, xtrain, ytrain)
    return calculate_fit(model, xtest, ytest)

def train(keys, values):
    model = PSOModel(randX0(numParticles, dims), bounds, fitnessFunction, keys, values, maxiter=numIter)
    model.begin()
    FinalModel = fitness_model(model.posbestgroup, keys, values)
    return FinalModel


data = readData("Data.csv") ## In the file, this will always be de-normalized, but in the code it is always normalised
data = [norm(row) for row in data] ## Normalizing the data
fullData = readData("DataFull.csv")
fullData = [norm(row) for row in fullData] ## Normalizing any extensions
NaNValue = -20.0
delta = 0.01



def trainForParameter(i):
    keys = []
    values = []
    keys.append(data[0][:i] + data[0][(i+1):])
    for k in range(1, len(data)):
        values.append(data[k][i])
        keys.append(data[k][:i] + data[k][(i+1):])

    keys = keys[:-1]

    model = train(keys, values)
    return model

def GDPTrain():
    keys = [point[0:1] + point[2:] for point in data]
    values = [point[1] for point in data]
    return train(keys, values)

def GrowthTrain():
    keys = [point[:2] + point[3:] for point in data]
    values = [point[2] for point in data]
    return train(keys, values)

def trainCyclicModels(t_t_split=0.75):
    global FutureModels
    global MainModel
    global GrowthModel
    global ARIMAModels
    
    FutureModels = []
    ARIMAModels = []

    for i in range(0, len(data[0])):
        FutureModels.append(trainForParameter(i))
        dataSpecific = [row[i] for row in data]
        ARIMAModels.append(continueSeriesModel(dataSpecific))
        print("\n\n----------------------\n\nMARK: Finished training parameter", i+1, "\n\n----------------------\n\n")

    MainModel = GDPTrain()
    GrowthModel = GrowthTrain()


def getForecastOneYear(row, history=None): # This doesn't use ARIMA; history needs to be of length windowSize
    forecast = []

    for i in range(0, len(data[0])):
        inputVal = row[:i] + row[(i+1):]
        forecast.append(numpy.array(numpy.array(FutureModels[i].predict(inputVal)).flat)[0]) # time-step parameter prediction

    forecast[0] = row[0] + (data[1][0] - data[0][0]) # Update the year reading correctly

    trueForecastGDP = numpy.array(numpy.array(MainModel.predict(forecast[0:1] + forecast[2:])).flat)[0] # predict true GDP
    trueForecastGrowth = numpy.array(numpy.array(GrowthModel.predict(forecast[:2] + forecast[3:])).flat)[0] # predict true growth

    forecast[-1] = trueForecastGDP # set true GDP


    if history != None:
        rawLSTM = []
        for indx in range(0, len(data[-1])):
            rawLSTM.append(makePredictions(LSTMs[indx], array([[[x[indx]] for x in history[-windowSize:]]])))#array([[[[row[indx]] for row in history]]])).flatten()[0])

        return [((LSTM_coefficient*a) + (RFBNN_coefficient*b))/(LSTM_coefficient+RFBNN_coefficient) for a, b in zip(rawLSTM, forecast)]
    
    else:
        return forecast

def replaceFullData():
    global data
    global NaNValue
    global fullData

    if len(numpy.array(fullData).flatten()) != 0:
        mainD = data[:]
        forecasted = [] # This is the replaced version

        for row in range(0, len(fullData)):
            f = getForecastOneYear(mainD[-1])
            # Fix the version here by combining
            f = [(point if (abs(fullData[row][index] - NaNValue) <= delta) else fullData[row][index]) for index, point in enumerate(f)]
            
            forecasted.append(f)
            mainD.append(f)
            #trainCyclicModels()
            
        fullData = forecasted
        data = data + fullData


def getForecastNYears(mainD, n, autoDenorm=False): #ensure window size is smaller
    # Where it should continue from - data from the textfields
    forecasted = []
    

    for _ in range(0, n):
        f = getForecastOneYear(mainD[-1], history=mainD[-windowSize:])
        forecasted.append(f)
        mainD.append(f)
        #trainCyclicModels()
        
    ## Now, forecast using ARIMA
    rawARIMA = []
    for indx in range(0, len(mainD[-1])):
        rawARIMA.append(continueSeries(ARIMAModels[indx], [row[indx] for row in mainD], n))
    ARIMA_pred = numpy.array(rawARIMA).T # rows are the inner dimension


    
    if autoDenorm:
        return [denorm(x) for x in list(numpy.add(ARIMA_pred*ARIMA_coefficient, numpy.array(forecasted)*(RFBNN_coefficient+LSTM_coefficient)))]
    else:
        return list(numpy.add(ARIMA_pred*ARIMA_coefficient, numpy.array(forecasted)*(RFBNN_coefficient+LSTM_coefficient)))



## Running getForecastNYears(n) returns a tuple consisting of all required datapoints in the order given in the excel sheet



def predictForPast(plot=False, extra=10, ind=2):
    predicted = []
    true = []
    
    for k in range (windowSize, len(data)-1): # K is the index to start from
        predicted.append(getForecastNYears(data[0:(k+1)], 1)[0][ind]) # k+1 elements before
        true.append(data[k+1][ind])

    extra_pred = getForecastNYears(data[:], extra)
    predicted += [l[ind] for l in extra_pred]

    if plot:
        y_pred = pd.Series(predicted)
        y_true = pd.Series(true)
        y_pred.plot()
        y_true.plot()
        plt.show()
        

    return predicted, true








####### ENDING TRAINING CODE #########


print("Training will commence if no model is found.")

if MainModel == None or GrowthModel == None:
    print("No model found, so training now")
    trainCyclicModels()
    replaceFullData()
    print("Fixed Missing Data")
    trainCyclicModels()
    print("Please store the values of each array in the training file")
    print("FutureModelsEx:", [x.export() for x in FutureModels])
    print("MainModelEx:", MainModel.export())
    print("GrowthModelEx:", GrowthModel.export())
    print("ARIMAModels:", ARIMAModels)
    numpy.savetxt("Data.csv", numpy.array([denorm(row) for row in data]), delimiter=",") # Denormalizing the repaired data for storage
    numpy.savetxt("DataFull.csv", numpy.array([[]]), delimiter=",")

    for indx in range(0, len(data[-1])):
        trainModelAndStore(numpy.array([row[indx] for row in data[:]]), windowSize, filePath=("model"+str(indx)))
    
    
    # All 4 key ones, and data
else:
    LSTMs = [retrieveModel("model"+str(i)) for i in range(0, len(Mins))]
    print("Model found, skipping training")

originalData = data[:]


def nowCastPrepare(newRow): # Missing entries are to be denoted by -20
    global data
    global fullData
    global originalData
    
    data = originalData[:]
    fullData = [newRow[:]]
    replaceFullData()


def fullForecast(row, n): # Use -20 for unknown variables
    global data
    
    nowCastPrepare(norm(row))
    return getForecastNYears(data[:], n, autoDenorm=True)

# Call nowCastPrepare before running any nowCast - note that doing this will remove previous entries
# When calling forecase, use data[:], instead of directly using data




# DATA repairing to be fixed
