import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

IDIR = os.path.abspath('../data')

oXp = pd.read_csv(os.path.join(IDIR,'order_products__prior.csv'),
    dtype={'order_id': np.int32,'product_id': np.uint16,},
    usecols=['order_id', 'product_id'])

print('oXp loaded')

products = pd.read_csv(os.path.join(IDIR,'products.csv'),
    dtype={'product_id': np.uint16},
    usecols=['product_id'])

print('products loaded')

#oXp = oXp[:1024]
#products = products[:10240]

oXp2 = oXp.groupby('order_id').apply(lambda x: [x[i].tolist() for i in x ][1] )

print('Grouping oXp Colplete')

oXp3 = pd.DataFrame(index=oXp2.index)

#  for i in oXp2.index:
#      for j in products.product_id:
#          try:
#              oXp3.loc[i][j] = True
#          except:
#              oXp3[j] = False
#              oXp3.loc[i][j] = True
print(oXp3)

print('Fill False Complete')
print(oXp2.index)
for i in oXp2.index:
    for j in products.product_id:
        if j in oXp2[i]:
            try:
                oXp3[i][j] = True
            except:
                oXp3[j] = False
                oXp3.loc[i][j] = True
                print(i, j)
#        else: oXp3.loc[i][j] = False
    print(i, "th row checked")


#oXp3.to_csv(os.path.join(IDIR,'hello_himyp.csv'))

oXp3_corr = oXp3.corr()
# .to_csv(os.path.join(IDIR,'corr_himyp.csv'))
print(oXp3_corr)
# for i in products.product_id:
#     if np.isnan(oXp3_corr[i]):
#         oXp3_corr = oXp3_corr.drop(i)
# 
# for j in products.product_id:
#     if oXp3_corr[j] == np.nan()
#         oXp3_corr.drop(i)
# 
plt.matshow(oXp3_corr)
plt.show()
