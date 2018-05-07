import numpy as np

class AlgCompare:
    def __init__(self):
        pgMean = {
          "x": [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120, 2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480, 2520, 2560, 2600, 2640, 2680, 2720, 2760, 2800, 2840, 2880, 2920, 2960, 3000, 3040, 3080, 3120, 3160, 3200, 3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3600, 3640, 3680, 3720, 3760, 3800, 3840, 3880, 3920, 3960, 4000, 4040, 4080, 4120, 4160, 4200, 4240, 4280, 4320, 4360, 4400, 4440, 4480, 4520, 4560, 4600, 4640, 4680, 4720, 4760, 4800, 4840, 4880, 4920, 4960, 5000, 5000, 4960, 4920, 4880, 4840, 4800, 4760, 4720, 4680, 4640, 4600, 4560, 4520, 4480, 4440, 4400, 4360, 4320, 4280, 4240, 4200, 4160, 4120, 4080, 4040, 4000, 3960, 3920, 3880, 3840, 3800, 3760, 3720, 3680, 3640, 3600, 3560, 3520, 3480, 3440, 3400, 3360, 3320, 3280, 3240, 3200, 3160, 3120, 3080, 3040, 3000, 2960, 2920, 2880, 2840, 2800, 2760, 2720, 2680, 2640, 2600, 2560, 2520, 2480, 2440, 2400, 2360, 2320, 2280, 2240, 2200, 2160, 2120, 2080, 2040, 2000, 1960, 1920, 1880, 1840, 1800, 1760, 1720, 1680, 1640, 1600, 1560, 1520, 1480, 1440, 1400, 1360, 1320, 1280, 1240, 1200, 1160, 1120, 1080, 1040, 1000, 960, 920, 880, 840, 800, 760, 720, 680, 640, 600, 560, 520, 480, 440, 400, 360, 320, 280, 240, 200, 160, 120, 80, 40, 0],
          "y": [5.2201706435, 2.49127416148, 2.86490054103, 3.24229542411, 3.45727368983, 3.83232489859, 4.38703127274, 4.73803847808, 5.333246251, 5.78027606576, 6.13863383279, 6.40168612707, 6.63135416253, 7.22719452123, 7.41807316598, 7.56719000463, 7.74870073072, 7.84935540565, 8.16554791289, 8.30807688754, 8.36527568252, 8.44149881179, 8.75179034797, 8.73272626291, 9.03537735267, 9.01631710918, 9.79757339102, 9.53635442131, 9.52290435136, 9.82320935274, 9.9877090948, 9.93026270374, 10.5361400001, 10.5110167658, 10.8262241159, 10.8986403713, 11.383312932, 11.5660223557, 11.5092449756, 11.3491545048, 11.805978601, 12.0555782803, 11.9234372841, 12.7699096519, 13.2580153042, 12.4078809101, 12.8731183604, 13.2872702952, 13.2187197715, 13.5655978441, 13.5335531384, 14.291450935, 14.6640975269, 14.1478448626, 14.9086741021, 15.9565205176, 14.9380345873, 15.5601564162, 16.6279665673, 17.2011942904, 16.8531925366, 17.4767521818, 16.9529046557, 17.5602221659, 16.9636457792, 17.3433972178, 18.0303611268, 17.3684080502, 18.4708377423, 18.9281860285, 19.9000165312, 20.6903979544, 20.5252380161, 21.6521350529, 22.2358523504, 22.2305828792, 23.0304114269, 23.3036612, 23.7980034511, 22.1941896608, 22.737120956, 22.5716842609, 22.7789836574, 24.4549109238, 25.1043502044, 25.1270306814, 25.825418397, 26.2444240938, 27.9273698662, 27.4135885059, 29.3377193585, 29.4130541753, 27.7159157073, 29.0816897947, 29.1954012033, 29.6406151284, 29.3129527003, 30.2134013564, 31.0934608512, 31.4365415242, 31.9255746986, 32.8566988153, 35.2269701433, 34.2385261927, 35.0657474455, 35.91336126, 35.1583407412, 35.1091826621, 34.8959756481, 35.035391118, 34.9187920864, 35.0265206893, 36.495058801, 38.3297294621, 39.9413140047, 39.9101696845, 40.6472853454, 42.0929115594, 40.047495826, 42.2916221518, 42.2020805518, 43.1483013264, 43.6658107106, 42.7093145121, 42.4947972811, 42.4256144514, 21.2483855486, 20.5317027189, 20.4236854879, 19.1611892894, 18.3376986736, 18.8009194482, 19.4868778482, 18.904504174, 19.2810884406, 18.9347146546, 18.1048303155, 19.5491859953, 18.2662705379, 18.372441199, 17.1609793107, 17.3017079136, 17.738108882, 17.1150243519, 16.0403173379, 16.0551592588, 15.52463874, 15.7617525545, 14.4524738073, 14.8505298567, 14.9498011847, 15.4024253014, 15.0424584758, 15.2245391488, 14.0200986436, 14.1120472997, 12.5888848716, 13.3725987967, 13.7083102053, 12.9415842927, 12.8279458247, 12.8547806415, 12.5769114941, 12.3291301338, 12.9270759062, 12.874081603, 12.1549693186, 12.9821497956, 12.0905890762, 11.7725163426, 10.9643157391, 11.203879044, 11.0023103392, 12.0264965489, 11.7208388, 10.6875885731, 10.8374171208, 10.5876476496, 11.2773649471, 10.4857619839, 10.5541020456, 10.3044834688, 10.5368139715, 9.88666225771, 10.1430919498, 10.3841388732, 10.2861027822, 10.4073542208, 10.0312778341, 9.69509534432, 9.1407478182, 9.49530746341, 8.83980570957, 9.31853343267, 9.31384358381, 9.01096541272, 8.6639794824, 9.34782589791, 9.04365513744, 8.74640247309, 9.10704906498, 8.72294686162, 8.67890215585, 7.97328022847, 7.94422970483, 7.65588163964, 7.5671190899, 7.70398469576, 7.51809034809, 7.70806271591, 7.56842171966, 7.71702139899, 7.18084549517, 7.41675502444, 7.22147764426, 7.26868706804, 6.6018596287, 6.97977588409, 6.47748323423, 6.82885999989, 6.64873729626, 6.6247909052, 6.30979064726, 6.26759564864, 5.97914557869, 5.93292660898, 5.96618289082, 5.64162264733, 5.82327373709, 5.40120965203, 5.28550118821, 5.25272431748, 5.16542311246, 5.07395208711, 4.94914459435, 4.48479926928, 4.56180999537, 4.36842683402, 4.46380547877, 4.00614583747, 3.87231387293, 3.38786616721, 3.10672393424, 2.849253749, 2.50296152192, 2.36846872726, 2.29767510141, 2.03522631017, 1.86770457589, 1.82059945897, 1.65572583852, -0.640170643495],
          "fill": "tozerox",
          "fillcolor": "rgba(178,102,255,0.2)",
          "line": {"color": "transparent"},
          "name": "Policy gradient",
          "showlegend": False,
          "type": "scatter",
          "uid": "20a8e5",
          "xsrc": "nicolepilsworth:19:4de1df",
          "ysrc": "nicolepilsworth:19:b47cc3"
        }
        pgSd = {
          "x": [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120, 2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480, 2520, 2560, 2600, 2640, 2680, 2720, 2760, 2800, 2840, 2880, 2920, 2960, 3000, 3040, 3080, 3120, 3160, 3200, 3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3600, 3640, 3680, 3720, 3760, 3800, 3840, 3880, 3920, 3960, 4000, 4040, 4080, 4120, 4160, 4200, 4240, 4280, 4320, 4360, 4400, 4440, 4480, 4520, 4560, 4600, 4640, 4680, 4720, 4760, 4800, 4840, 4880, 4920, 4960, 5000],
          "y": [2.29, 2.0735, 2.34275, 2.555, 2.74625, 3.065, 3.37775, 3.6205, 4.09125, 4.4435, 4.76325, 5.137, 5.31875, 5.8455, 5.89325, 6.0645, 6.11675, 6.39925, 6.61975, 6.73675, 6.809, 6.8635, 7.0765, 7.278, 7.3385, 7.49125, 7.86525, 7.75775, 7.89525, 8.0665, 8.30625, 8.2895, 8.6825, 8.49425, 8.903, 8.75025, 9.326, 9.39375, 9.463, 9.265, 9.7615, 9.812, 9.81575, 10.144, 10.481, 9.9875, 10.2645, 10.61575, 10.596, 11.12225, 11.12825, 11.69925, 11.70525, 11.59575, 12.12825, 12.31025, 11.9745, 12.437, 12.97325, 13.0205, 13.17425, 13.30875, 13.324, 13.79575, 13.6855, 13.81475, 14.20725, 13.75575, 14.17875, 14.7325, 15.10225, 15.62225, 15.5055, 16.46475, 16.41175, 16.534, 16.859, 17.51225, 17.91225, 16.59825, 16.9705, 16.768, 17.27575, 18.27275, 19.04325, 18.641, 19.34975, 19.58575, 20.12825, 19.99525, 21.09625, 21.1205, 20.32875, 21.395, 21.284, 21.11475, 21.7125, 22.11675, 23.159, 23.2395, 23.664, 23.90325, 25.03875, 24.3455, 25.41375, 25.719, 25.60675, 25.57475, 26.0055, 26.38675, 26.11025, 26.09375, 27.43375, 28.298, 29.74525, 29.0075, 29.791, 30.687, 29.476, 30.88925, 30.5015, 30.743, 31.4135, 31.5665, 31.51325, 31.837],
          "line": {"color": "rgb(178,102,255)"},
          "mode": "lines",
          "name": "Policy gradient",
          "type": "scatter",
          "uid": "a14dd0",
          "xsrc": "nicolepilsworth:19:d7ff89",
          "ysrc": "nicolepilsworth:19:e8d449"
        }

        qTableMean = {
          "x": [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120, 2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480, 2520, 2560, 2600, 2640, 2680, 2720, 2760, 2800, 2840, 2880, 2920, 2960, 3000, 3040, 3080, 3120, 3160, 3200, 3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3600, 3640, 3680, 3720, 3760, 3800, 3840, 3880, 3920, 3960, 4000, 4040, 4080, 4120, 4160, 4200, 4240, 4280, 4320, 4360, 4400, 4440, 4480, 4520, 4560, 4600, 4640, 4680, 4720, 4760, 4800, 4840, 4880, 4920, 4960, 5000, 5000, 4960, 4920, 4880, 4840, 4800, 4760, 4720, 4680, 4640, 4600, 4560, 4520, 4480, 4440, 4400, 4360, 4320, 4280, 4240, 4200, 4160, 4120, 4080, 4040, 4000, 3960, 3920, 3880, 3840, 3800, 3760, 3720, 3680, 3640, 3600, 3560, 3520, 3480, 3440, 3400, 3360, 3320, 3280, 3240, 3200, 3160, 3120, 3080, 3040, 3000, 2960, 2920, 2880, 2840, 2800, 2760, 2720, 2680, 2640, 2600, 2560, 2520, 2480, 2440, 2400, 2360, 2320, 2280, 2240, 2200, 2160, 2120, 2080, 2040, 2000, 1960, 1920, 1880, 1840, 1800, 1760, 1720, 1680, 1640, 1600, 1560, 1520, 1480, 1440, 1400, 1360, 1320, 1280, 1240, 1200, 1160, 1120, 1080, 1040, 1000, 960, 920, 880, 840, 800, 760, 720, 680, 640, 600, 560, 520, 480, 440, 400, 360, 320, 280, 240, 200, 160, 120, 80, 40, 0],
          "y": [2.93256929152, 3.64591486808, 4.01228816428, 4.3911518693, 4.80954673347, 4.98757844822, 5.49089341385, 5.35792904208, 5.93642527217, 6.03529875123, 6.69782263162, 6.58675639753, 7.05733117496, 6.91416004759, 7.46946305642, 7.39155870432, 7.53108101472, 8.44138068887, 8.47438780376, 8.55002631559, 9.0133348201, 9.3182503563, 10.1331076095, 10.6272755711, 10.2086997867, 10.8342655085, 10.8139924051, 12.3877758754, 12.1312022703, 12.6401186888, 12.8880658221, 13.6947025794, 14.6848013125, 14.7338124018, 15.2777659829, 16.7094307812, 16.7393386031, 15.4490463601, 17.6758839213, 18.4356167022, 18.4655727748, 20.7909952619, 22.5588911928, 21.362258195, 22.7770019357, 23.118873092, 25.8109477592, 24.2677575759, 24.3586711485, 26.6250826235, 27.9570407003, 28.9269834763, 32.0203554683, 31.7319566315, 33.4504367471, 33.8492971706, 32.916725187, 36.3099867607, 38.4433472241, 36.2609021616, 41.8123546637, 44.8072542112, 46.1561710534, 43.315601208, 45.7839347707, 46.6973431535, 49.8310012056, 50.7580576544, 53.677670677, 55.212138519, 55.2704255071, 58.157664638, 57.658624727, 58.3567277711, 59.1948582263, 60.0711045204, 62.5680230382, 65.0783059269, 67.3690697049, 71.247374606, 71.8580917601, 68.3927162283, 73.6861808003, 74.312780697, 71.3249147571, 74.3291617683, 81.9217761451, 80.0705596786, 81.802788932, 81.4691268583, 87.2286097623, 89.8110900002, 89.224228006, 85.9314781419, 86.9636561063, 88.612832169, 93.693795773, 93.2468875182, 94.6568210677, 98.7309155357, 105.926573494, 103.225496678, 103.545915458, 101.985063815, 104.175151059, 105.19135996, 108.289603464, 107.288478331, 106.415749313, 110.586376338, 110.642875461, 112.796691645, 111.752999454, 114.866957229, 118.661630044, 118.159732975, 118.714821393, 123.242838632, 124.890457912, 117.782811808, 121.647525502, 121.686663902, 127.331876715, 125.205607502, 128.629435749, 132.965027119, 78.5749728811, 74.7745642513, 73.0783924984, 70.0501232851, 67.7593360984, 68.8764744983, 69.1251881918, 66.4115420878, 65.3071613682, 61.0631786067, 61.3562670247, 60.8443699564, 59.9470427713, 58.5930005455, 54.5233083553, 55.3271245393, 53.8816236622, 51.0482506875, 51.3435216686, 55.3363965358, 47.02064004, 48.388848941, 50.6649361848, 49.0640845418, 51.6685033216, 51.5854265059, 48.4730844643, 43.1011789323, 41.6031124818, 41.448204227, 35.911167831, 38.7223438937, 36.8765218581, 36.979771994, 34.4889099998, 36.9493902377, 34.5268731417, 34.401211068, 29.8054403214, 30.4422238549, 31.9288382317, 26.5470852429, 27.779219303, 27.3918191997, 26.0032837717, 27.0019082399, 25.836625394, 25.2629302951, 21.1916940731, 23.2419769618, 22.9968954796, 21.6351417737, 20.3852722289, 18.687375273, 20.640335362, 18.8475744929, 19.115861481, 16.468329323, 16.1379423456, 17.4809987944, 16.5326568465, 16.1040652293, 15.632398792, 14.3998289466, 13.1207457888, 13.8136453363, 14.3370978384, 14.0466527759, 13.0460132393, 15.141274813, 13.7427028294, 13.2735632529, 12.7320433685, 12.7496445317, 11.6790165237, 10.8809592997, 11.8229173765, 11.2193288515, 10.6142424241, 9.25305224077, 11.449126908, 10.8949980643, 10.997741805, 9.20510880724, 9.60500473807, 9.68842722517, 9.44038329779, 8.4581160787, 8.22295363993, 8.09466139688, 7.80656921878, 7.3982340171, 6.7381875982, 7.55319868753, 7.06129742063, 6.5679341779, 6.89388131119, 6.59879772974, 5.82022412458, 6.07400759495, 5.93773449152, 6.03530021326, 5.50672442889, 5.71889239048, 5.2837496437, 5.2766651799, 5.05397368441, 4.85361219624, 4.81861931113, 4.58091898528, 4.27844129568, 4.21453694358, 4.08983995241, 3.76266882504, 3.93924360247, 3.51817736838, 3.40070124877, 3.10957472783, 3.24407095792, 2.81710658615, 2.73442155178, 2.69645326653, 2.3168481307, 2.17971183572, 1.69608513192, 1.17543070848],
          "fill": "tozerox",
          "fillcolor": "rgba(0,153,76,0.2)",
          "line": {"color": "transparent"},
          "name": "Tabular Q",
          "showlegend": False,
          "type": "scatter",
          "uid": "20dd43",
          "xsrc": "nicolepilsworth:65:3be675",
          "ysrc": "nicolepilsworth:65:8dd15d"
        }
        qTableSd = {
          "x": [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120, 2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480, 2520, 2560, 2600, 2640, 2680, 2720, 2760, 2800, 2840, 2880, 2920, 2960, 3000, 3040, 3080, 3120, 3160, 3200, 3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3600, 3640, 3680, 3720, 3760, 3800, 3840, 3880, 3920, 3960, 4000, 4040, 4080, 4120, 4160, 4200, 4240, 4280, 4320, 4360, 4400, 4440, 4480, 4520, 4560, 4600, 4640, 4680, 4720, 4760, 4800, 4840, 4880, 4920, 4960, 5000],
          "y": [2.054, 2.671, 3.096, 3.354, 3.753, 3.861, 4.154, 4.301, 4.523, 4.718, 5.108, 5.263, 5.41, 5.502, 5.842, 5.835, 6.056, 6.63, 6.664, 6.802, 7.145, 7.301, 7.926, 8.067, 8.122, 8.386, 8.444, 9.104, 9.365, 9.767, 9.728, 10.378, 11.119, 10.736, 11.338, 12.258, 12.417, 11.836, 13.067, 13.938, 14.077, 15.198, 15.882, 16.18, 16.836, 17.284, 17.532, 17.441, 17.789, 19.224, 19.419, 20.303, 22.385, 22.232, 23.362, 23.796, 24.029, 24.678, 26.245, 25.299, 27.813, 28.964, 30.278, 29.474, 30.944, 31.615, 33.656, 33.448, 35.073, 37.164, 37.059, 39.399, 38.173, 39.371, 40.415, 41.534, 42.905, 43.135, 46.316, 48.542, 49.43, 47.198, 50.539, 51.046, 48.936, 53.129, 56.182, 54.938, 58.102, 57.998, 62.089, 62.15, 63.102, 61.404, 62.843, 62.262, 67.571, 67.425, 68.879, 73.602, 78.756, 77.447, 76.305, 76.325, 76.282, 76.106, 81.813, 79.316, 78.732, 82.234, 82.985, 83.66, 85.173, 87.407, 89.753, 89.758, 89.889, 94.275, 95.651, 93.454, 95.262, 94.723, 98.691, 99.142, 101.702, 105.77],
          "line": {"color": "rgb(0,153,76)"},
          "mode": "lines",
          "name": "Tabular Q",
          "type": "scatter",
          "uid": "fe02cb",
          "xsrc": "nicolepilsworth:65:261eb5",
          "ysrc": "nicolepilsworth:65:ce51f0"
        }

        a3cMean = {
          "x": [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120, 2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480, 2520, 2560, 2600, 2640, 2680, 2720, 2760, 2800, 2840, 2880, 2920, 2960, 3000, 3040, 3080, 3120, 3160, 3200, 3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3600, 3640, 3680, 3720, 3760, 3800, 3840, 3880, 3920, 3960, 4000, 4040, 4080, 4120, 4160, 4200, 4240, 4280, 4320, 4360, 4400, 4440, 4480, 4520, 4560, 4600, 4640, 4680, 4720, 4760, 4800, 4840, 4880, 4920, 4920, 4880, 4840, 4800, 4760, 4720, 4680, 4640, 4600, 4560, 4520, 4480, 4440, 4400, 4360, 4320, 4280, 4240, 4200, 4160, 4120, 4080, 4040, 4000, 3960, 3920, 3880, 3840, 3800, 3760, 3720, 3680, 3640, 3600, 3560, 3520, 3480, 3440, 3400, 3360, 3320, 3280, 3240, 3200, 3160, 3120, 3080, 3040, 3000, 2960, 2920, 2880, 2840, 2800, 2760, 2720, 2680, 2640, 2600, 2560, 2520, 2480, 2440, 2400, 2360, 2320, 2280, 2240, 2200, 2160, 2120, 2080, 2040, 2000, 1960, 1920, 1880, 1840, 1800, 1760, 1720, 1680, 1640, 1600, 1560, 1520, 1480, 1440, 1400, 1360, 1320, 1280, 1240, 1200, 1160, 1120, 1080, 1040, 1000, 960, 920, 880, 840, 800, 760, 720, 680, 640, 600, 560, 520, 480, 440, 400, 360, 320, 280, 240, 200, 160, 120, 80, 40, 0],
          "y": [4.23156860163, 2.95676945665, 3.64807243201, 4.73898537973, 5.66044314263, 5.88777800352, 6.48828698895, 6.40044253164, 6.58236828401, 6.69486245138, 7.04313332632, 7.14549933813, 7.31626591404, 7.32464271152, 7.53247406519, 7.98237382052, 8.23965325078, 8.78498455824, 8.837866903, 9.54170101468, 9.41146421078, 9.58724771685, 10.3020032231, 10.4802178887, 11.0763429428, 11.0469114572, 11.708184792, 11.6758869924, 12.0635693487, 12.2259279573, 12.968983161, 12.9770354706, 14.2123232109, 13.5223945134, 14.5051138897, 14.651409582, 14.5407749991, 15.2511571785, 16.1194081071, 16.472664028, 16.8172783714, 16.791541604, 17.9220437741, 17.8589786598, 18.0184523092, 18.4997962299, 19.0390954599, 19.7230126626, 20.2278641194, 19.6875636857, 20.9241873876, 22.6625952844, 22.3223652015, 22.4221942678, 23.2329070476, 24.3393954893, 24.913210478, 25.7875284523, 25.7121338212, 26.4336367576, 26.9107875687, 28.7598775258, 30.1647512929, 31.341771808, 30.7928744302, 31.9286528585, 32.8748383467, 31.5338145707, 34.2272720694, 34.54573199, 34.9195032911, 33.9904105618, 35.1794199611, 36.1759002412, 36.3261633862, 36.9263129551, 37.0897562908, 38.5006673755, 37.2659165901, 37.6950287305, 36.9973845934, 37.7193618879, 36.9738208243, 38.2861561971, 38.7812611939, 37.1712586394, 37.5234242331, 35.7526583909, 35.3569619468, 34.3985909219, 35.3708722168, 34.8360019731, 37.1602523803, 37.1361196036, 37.4849529926, 37.442816563, 37.5283954285, 37.6485537, 38.0205121944, 35.7936690521, 35.8590158237, 35.3789648433, 35.241517211, 36.1467749551, 37.0031181178, 36.8461373884, 36.5940858916, 36.8090699911, 36.4388558694, 35.8464635721, 37.4798744644, 38.3497194793, 38.5807474655, 36.7490354401, 36.0270229639, 36.4892167236, 36.6993126592, 37.1610490959, 37.0318653417, 36.9805646498, 38.8809906711, 36.787198728, 38.5777174979, 38.2828115124, 22.6930813447, 23.8017467878, 22.9507477006, 23.2015986146, 22.9314889216, 23.8181346583, 22.7510044755, 23.1944373408, 22.9929261335, 23.3359234647, 22.9335538456, 22.8357703917, 22.7824233779, 23.4795005356, 22.2830007136, 22.5102512735, 22.0989657232, 21.8005569655, 22.4864518973, 22.6830425964, 22.2849214735, 22.7500006462, 22.5031780138, 23.3097341763, 22.1674916622, 22.0562735199, 21.9210891572, 23.2872295715, 21.9232548656, 22.306118436, 22.4326303964, 22.6218904768, 21.8537301697, 20.9737706404, 21.3072126496, 21.5733951961, 21.8080558948, 21.0252364812, 22.4872235034, 20.5941852347, 20.9468795172, 22.7168041757, 21.8105488264, 21.511543978, 21.1322034124, 21.1305119813, 21.514511196, 21.1651544234, 21.2848477592, 20.5399080424, 19.4419569016, 20.5304014675, 19.4395001524, 18.8804967089, 19.06676801, 18.2789779306, 18.4353818579, 18.232750939, 17.7637578558, 18.0883755698, 17.4479603349, 17.0776594214, 16.5526224742, 15.7726945742, 16.0351132424, 15.6999197502, 15.1544358334, 15.289914522, 14.9503366536, 14.3505750953, 14.8389664465, 14.0722776556, 14.2217797156, 13.2807233267, 13.3035077429, 12.8507073092, 12.2854694803, 12.606886683, 11.8604716273, 11.6891369765, 11.9084320545, 11.4377776544, 11.0383691103, 10.5693287715, 10.049210972, 10.0337168929, 9.80286067867, 9.67306428657, 9.12001898942, 8.98283253883, 8.83742691516, 8.68410536049, 8.45108952938, 8.08503469617, 7.7062148998, 7.70741279412, 7.19018443614, 6.83645806515, 6.81023139991, 6.55312134292, 6.45013925413, 6.24130034838, 6.171234426, 5.89433936065, 5.41588827104, 5.52865095415, 5.11233687033, 5.05543603493, 4.97967975091, 4.81261522053, 4.66955371705, 4.66453765739, 4.49378637616, 4.38767024511, 4.42388754862, 4.44307814457, 4.0223253255, 3.99876658248, 3.60061485362, 3.07303900023, 2.76146104884, 2.1470168537, 1.74635554335, 0.41645814768],
          "fill": "tozerox",
          "fillcolor": "rgba(0,76,153,0.2)",
          "line": {"color": "transparent"},
          "name": "A3C",
          "showlegend": False,
          "type": "scatter",
          "xsrc": "nicolepilsworth:71:c4de94",
          "ysrc": "nicolepilsworth:71:91c558"
        }
        a3cSd = {
          "x": [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320, 1360, 1400, 1440, 1480, 1520, 1560, 1600, 1640, 1680, 1720, 1760, 1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 2120, 2160, 2200, 2240, 2280, 2320, 2360, 2400, 2440, 2480, 2520, 2560, 2600, 2640, 2680, 2720, 2760, 2800, 2840, 2880, 2920, 2960, 3000, 3040, 3080, 3120, 3160, 3200, 3240, 3280, 3320, 3360, 3400, 3440, 3480, 3520, 3560, 3600, 3640, 3680, 3720, 3760, 3800, 3840, 3880, 3920, 3960, 4000, 4040, 4080, 4120, 4160, 4200, 4240, 4280, 4320, 4360, 4400, 4440, 4480, 4520, 4560, 4600, 4640, 4680, 4720, 4760, 4800, 4840, 4880, 4920],
          "y": [2.32401337466, 2.3515625, 2.89754464286, 3.75022321429, 4.36674107143, 4.74419642857, 5.24352678571, 5.21138392857, 5.51272321429, 5.559375, 5.71540178571, 5.81964285714, 5.99040178571, 5.99709821429, 6.17254464286, 6.48102678571, 6.64754464286, 6.94866071429, 7.18325892857, 7.47879464286, 7.65290178571, 7.87924107143, 8.27165178571, 8.46517857143, 8.81473214286, 8.92857142857, 9.27232142857, 9.43303571429, 9.88549107143, 9.96607142857, 10.5270089286, 10.7140625, 11.4482142857, 11.1799107143, 11.7439732143, 11.8857142857, 12.1069196429, 12.5270089286, 13.0765625, 13.2609375, 13.6933035714, 13.9149553571, 14.6799107143, 14.8837053571, 14.8537946429, 15.1801339286, 15.8229910714, 16.0042410714, 16.5392857143, 16.4955357143, 17.1024553571, 18.4421875, 18.1973214286, 18.6305803571, 18.7917410714, 19.6448660714, 20.1015625, 20.4709821429, 20.7060267857, 21.234375, 21.3417410714, 22.65625, 23.6212053571, 24.3948660714, 24.440625, 24.8462053571, 25.5537946429, 24.9845982143, 26.253125, 26.80625, 26.9, 26.7149553571, 27.8549107143, 27.8089285714, 28.4330357143, 29.1055803571, 29.1274553571, 30.0075892857, 29.1982142857, 29.4136160714, 29.2544642857, 29.7649553571, 29.8453125, 29.6165178571, 29.6877232143, 29.8292410714, 29.2743303571, 28.7803571429, 28.4651785714, 27.8529017857, 28.1723214286, 28.3448660714, 29.8910714286, 29.784375, 29.8955357143, 29.6830357143, 30.4078125, 29.7848214286, 30.0383928571, 28.9805803571, 29.584375, 28.9410714286, 28.9957589286, 29.2158482143, 29.8430803571, 29.6662946429, 29.1973214286, 29.4540178571, 29.4745535714, 29.0647321429, 30.4796875, 30.5660714286, 30.7082589286, 29.8412946429, 29.6814732143, 29.7410714286, 29.946875, 29.9560267857, 30.425, 29.9560267857, 31.0412946429, 29.8689732143, 31.1897321429, 30.4879464286],
          "line": {"color": "rgb(0,76,153)"},
          "mode": "lines",
          "name": "A3C",
          "type": "scatter",
          "xsrc": "nicolepilsworth:71:d35146",
          "ysrc": "nicolepilsworth:71:8e7bda"
        }

        self.data = [qTableMean, qTableSd, pgMean, pgSd, a3cMean, a3cSd]
