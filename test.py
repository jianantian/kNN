# _*_ coding:utf-8 _*_
import numpy as np
import kNN
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
#导入中文字体

group, labels = kNN.createDataSet()
a = kNN.classify0([1,1], group, labels, 3)


dataSet, labels = kNN.file2matrix('datingTestSet.txt')
normData, ranges, minVals = kNN.autonorm(dataSet)
patterns = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
patterns2 = {3:u'极具魅力', 2:u'魅力一般', 1:u'不喜欢'}
labelsPatterns = {1:u'游戏时间百分比', 0:u'每年获得飞行里程', 2:u'每周消耗冰淇淋公升数'}
labels2 = [patterns[x] if x in patterns else x for x in labels]
labe = np.array(labels2)
fig = plt.figure()
ax = fig.add_subplot(111)
idx = []
for i in range(1,4):
    idx.append(np.where(labe==i))

p1 = ax.scatter(normData[idx[0], 0], normData[idx[0], 1], color = 'r', label=patterns2[1])
p2 = ax.scatter(normData[idx[1], 0], normData[idx[1], 1], color = 'g', label=patterns2[2])
p3 = ax.scatter(normData[idx[2], 0], normData[idx[2], 1], color = 'b', label=patterns2[3])
plt.xlabel(labelsPatterns[0], fontproperties=font)
plt.ylabel(labelsPatterns[1], fontproperties=font)

plt.legend(loc='upper right', numpoints=1)
#plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize='small', fontproperties=font)
#设置图例的字体

#plt.show()


#kNN.classifyPerson()

#n = kNN.img2Vec('testDigits\\0_13.txt')
#print n[0,0:31]
kNN.handwritingClassTest()