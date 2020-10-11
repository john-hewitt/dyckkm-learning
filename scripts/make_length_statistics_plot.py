"""
Writes a simple graph plotting the percent of training and testing data of each observed length
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")

m3_training = {3: 388094, 5: 193673, 7: 146006, 9: 121721, 11: 102985, 13: 87980, 15: 74652, 17: 64264, 19: 54996, 21: 46715, 23: 40131, 25: 34246, 27: 28997, 29: 24927, 31: 20981, 33: 18115, 35: 15227, 37: 13298, 39: 11079, 41: 9499, 43: 8086, 45: 6758, 47: 5950, 49: 5082, 51: 4237, 53: 3753, 55: 3186, 57: 2731, 59: 2350, 61: 1966, 63: 1747, 65: 1421, 67: 1156, 69: 1107, 71: 925, 73: 833, 75: 650, 77: 573, 79: 438, 81: 404, 83: 331}
#m3_training_sum = sum(m3_training.values())
#m3_training = {k:v/m3_training_sum for k,v in m3_training.items()}

m3_testing = {87: 497, 89: 375, 91: 331, 93: 280, 95: 227, 97: 182, 99: 145, 101: 173, 103: 120, 105: 104, 107: 72, 109: 72, 111: 68, 113: 57, 115: 48, 117: 47, 119: 48, 121: 37, 123: 20, 125: 19, 127: 23, 129: 21, 131: 13, 133: 4, 135: 11, 137: 9, 139: 10, 141: 7, 143: 3, 145: 7, 147: 5, 149: 3, 151: 3, 157: 2, 159: 1, 161: 3, 165: 2}
#m3_testing_sum = sum(m3_testing.values())
#m3_testing = {k:v/m3_testing_sum for k,v in m3_testing.items()}

m5_training = {3: 240399, 5: 120008, 7: 75523, 9: 52540, 11: 40282, 13: 33595, 15: 28804, 17: 25831, 19: 23700, 21: 22051, 23: 20151, 25: 18610, 27: 17572, 29: 16305, 31: 15416, 33: 14434, 35: 13361, 37: 12306, 39: 11414, 41: 10845, 43: 10047, 45: 9175, 47: 8642, 49: 8322, 51: 7729, 53: 7178, 55: 6579, 57: 6189, 59: 5798, 61: 5300, 63: 5006, 65: 4754, 67: 4465, 69: 3964, 71: 3821, 73: 3567, 75: 3319, 77: 3148, 79: 2831, 81: 2766, 83: 2466, 85: 2394, 87: 2233, 89: 1968, 91: 1967, 93: 1773, 95: 1648, 97: 1552, 99: 1383, 101: 1430, 103: 1271, 105: 1156, 107: 1082, 109: 1033, 111: 963, 113: 908, 115: 815, 117: 783, 119: 719, 121: 668, 123: 590, 125: 541, 127: 510, 129: 504, 131: 485, 133: 436, 135: 421, 137: 364, 139: 375, 141: 361, 143: 313, 145: 296, 147: 271, 149: 249, 151: 232, 153: 237, 155: 211, 157: 219, 159: 172, 161: 152, 163: 173, 165: 121, 167: 145, 169: 121, 171: 119, 173: 121, 175: 101, 177: 88, 179: 75}
#m5_training_sum = sum(m5_training.values())
#m5_training = {k:v/m5_training_sum for k,v in m5_training.items()}

m5_testing = {183: 152, 185: 115, 187: 77, 189: 79, 191: 71, 193: 65, 195: 58, 197: 52, 199: 55, 201: 62, 203: 39, 205: 45, 207: 41, 209: 33, 211: 33, 213: 41, 215: 25, 217: 33, 219: 26, 221: 16, 223: 36, 225: 11, 227: 15, 229: 19, 231: 20, 233: 16, 235: 10, 237: 18, 239: 15, 241: 13, 243: 6, 245: 12, 247: 10, 249: 12, 251: 9, 253: 12, 255: 7, 257: 3, 259: 5, 261: 5, 263: 5, 265: 4, 267: 7, 269: 4, 271: 2, 273: 4, 275: 6, 277: 1, 279: 3, 281: 1, 283: 1, 287: 4, 289: 3, 291: 2, 293: 3, 295: 1, 297: 1, 299: 3, 303: 1, 307: 1, 309: 5, 311: 1, 313: 1, 317: 1, 321: 1, 323: 2, 327: 1, 337: 1, 343: 1}
#m5_testing_sum = sum(m5_testing.values())
#m5_testing = {k:v/m5_testing_sum for k,v in m5_testing.items()}

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
 
xs = sorted(m3_training.keys())
ys = [m3_training[x] for x in xs]
ax1.set_yscale('log')
ax1.bar(xs, ys,edgecolor='none',label='Train')
xs = sorted(m3_testing.keys())
ys = [m3_testing[x] for x in xs]
ax1.set_ylabel('m=3')
ax1.bar(xs, ys,edgecolor='none',label='Test')
ax2.legend()


# 
xs = sorted(m5_training.keys())
ys = [m5_training[x] for x in xs]
ax2.set_yscale('log')
ax2.bar(xs, ys,edgecolor='none',label='Train')
ax2.set_ylabel('m=5')
xs = sorted(m5_testing.keys())
ys = [m5_testing[x] for x in xs]
ax2.bar(xs, ys,edgecolor='none',label='Test')
ax2.legend()

# Write figure
sns.despine(fig=fig)
text1 = fig.text(-0.03, 0.5, 'Sample Count (log scale)', va='center', rotation='vertical')
text2 = fig.text(0.5, 0.00, 'Sample Length', ha='center')
text3 = fig.suptitle('Dyck-$(k,m)$ Length Distributions')
fig.savefig('plt.pdf', bbox_extra_artists=(text1, text2, text3), bbox_inches='tight' )
#fig.savefig('percents.png')

