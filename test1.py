from matplotlib import pyplot as plt
import numpy as np

# Creating dataset
# labelTime = ['Morning(5:00am-11:00am)', 'afternoon(11:00am-5:00pm)', 'Evening(5:00pm-10:00pm)','Night(10:00pm-5:00am)']
# coloursTime = ['#2980B9','#76D7C4','#85C1E9','#16A085']
# data = [0.02060125,0.00192527 ,0.9733767,  0.00409679]
# explode = [0.03,0,0,0]
#
# Z = [x for _,x in sorted(zip(data,labelTime))]
# Z.reverse()
# data.sort(reverse=True)
# # Creating plot
# plt.figure(figsize=(8, 5))
# plt.pie(data,labels=Z,colors=coloursTime,explode=explode,shadow=False,autopct='%1.1f%%') # wedgeprops={'edgecolor':'black'}
# plt.legend()
# plt.title("Crime prediction by time")
# # plt.tight_layout()
# plt.savefig('./static/images/img.png')
# plt.show()


x = np.char.array( ['Morning(5:00am-11:00am)', 'afternoon(11:00am-5:00pm)', 'Evening(5:00pm-10:00pm)','Night(10:00pm-5:00am)'])
y = np.array([0.02060125,0.00192527 ,0.9733767,  0.00409679])
colors =['#2980B9','#76D7C4','#16A085','#85C1E9']
porcent = 100.*y/y.sum()
plt.figure(figsize=(10, 5))
patches, texts = plt.pie(y, colors=colors, radius=1.2)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))
plt.title("Crime prediction by time")
plt.legend(patches, labels, loc='center right', bbox_to_anchor=(1,0.5), fontsize=10, bbox_transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.0, bottom=0.1, right=0.96)
plt.savefig('./static/images/piechart.png',bbox_inches="tight") #,bbox_inches="tight"
plt.show()