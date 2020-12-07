from matplotlib import pyplot as plt
import numpy as np

label = ['abuse','accident','assault','burglary','drugs','felony','kidnapping','other','sex related','vandalism','weapon violation']
values = [1.0279007e-06, 1.6315116e-06 ,4.7771334e-07 ,2.8131652e-01, 6.3140127e-05,
  1.2621217e-04, 1.8927572e-05, 5.3140600e-03, 1.4172327e-03 ,7.0932156e-01,
  2.4192049e-03]
Z = [x for _,x in sorted(zip(values,label))]
values.sort()
colour= ['#9ad3bc','#9ad3bc','#9ad3bc','#9ad3bc','#16a596','#16a596','#16a596','#16a596','#aa3a3a','#aa3a3a','#aa3a3a',]

ypos = np.arange(len(Z))
# plt.figure(figsize=(13, 5))
plt.title("PREDICT BY TYPE OF CRIME")
plt.ylabel("types of crimes")
plt.xlabel("probabilty")
plt.yticks(ypos,Z)
plt.barh(ypos,values,color=colour)
plt.tight_layout()
plt.savefig('./static/images/barchart.png')
# plt.show()