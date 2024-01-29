"""unfold xml data into 3 dataframes: publication, expenses and termination"""

import os
import pandas as pd
from xml_to_pkl import seao_todf
import xml.etree.ElementTree as etree
from utils.utils import cleanList

base = "data/xml"

#getting all files
f = []
for roots, dirs, files in os.walk(base):
	f.append(files)

#flattening list
f = cleanList(f)

# f = f[0:10]

#passing to seao_todf
term, pub, pub_sup, exp, exp_sup = [],[],[],[],[]

for files in f:
	path = os.path.join(base, files)
	print(path)
	root = seao_todf(path)
	root.probe()
	if (root.empty) or ('Revisions' in files):
		print('empty or revision')
		continue
	else:
		if "Contrats" in files:
			term.append(root.extract_data()[0])
		elif "Avis" in files:
			a,b = root.extract_data()
			pub.append(a)
			pub_sup.append(b)
		else:
			a,b = root.extract_data()
			exp.append(a)
			exp_sup.append(b)


#merge axis=0
termination = pd.concat(term, axis=0)
publication = pd.concat(pub, axis=0)
publication_suppliers = pd.concat(pub_sup, axis=0)
expenses = pd.concat(exp, axis=0)
expenses_suppliers = pd.concat(exp_sup, axis=0)

#merge axis=1
publication = publication.merge(publication_suppliers, left_on="numeroseao", 
	right_on="numeroseao")

expenses = expenses.merge(expenses_suppliers, left_on="numeroseao", 
	right_on="numeroseao") 

#save to_pickle
termination.to_pickle("data/termination.pkl")
publication.to_pickle("data/publication.pkl")
expenses.to_pickle("data/expenses.pkl")

#aggregating
dfs = [
publication.set_index('numeroseao'), 
termination.set_index('numeroseao'), 
expenses.set_index('numeroseao')
] 
df = dfs[0].join([dfs[1], dfs[2]])
print(df.info())
#saving raw aggregated data
df.to_pickle('data/aggregated_raw2.pkl')
print("files saved")
