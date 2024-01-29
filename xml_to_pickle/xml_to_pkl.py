'''
This software does:
	- convert SEAO XML file into Pandas dataframes... 
	- ... or conventional list structure
	- add ID columns to link datasheets
'''
import os 
import pandas as pd 
import xml.etree.ElementTree as et

class seao_todf:
	'''
	- Typical use:
		path = 'declare the path'
		x = seao_todf(path)
		data = x.extract_data()
	'''
	def __init__(self, path):
		self.path = path
		self.tree = et.parse(self.path)
		self.root = self.tree.getroot()
		self.subroot = False
		self.empty = True
		
	def probe(self):
		'''
		Detect if subtrees are empty and if there is grandchild nodes 
		'''
		try:
			self.root[0]
		except IndexError:
			pass
		else:
			self.empty = False
			try:
				self.root[0][-1][0]
			except IndexError:
				pass
			else:
				self.subroot = True
	
	def get_tag(self):
		'''
		This subroutine fetches column labels
		'''
		var = []
		for item in self.root[0]:
			var.append(item.tag)
		if self.subroot:
			subvar = []
			for elem in self.root[0][-1][0]:
				subvar.append(elem.tag)
			return var, subvar
		else:
			return var
		
	def get_tag_probe(self):
		'''
		to be used only if the column labels are of interest:
			path = 'declare the path'
			x = seao_todf(path)
			labels = x.get_tag_probe()
		'''
		self.probe()
		if not self.empty:
			var = []
			for item in self.root[0]:
				var.append(item.tag)
			if self.subroot:
				subvar = []
				for elem in self.root[0][-1][0]:
					subvar.append(elem.tag)
				return var, subvar
			else:
				return var
		
	def extract_data(self, df=True):
		'''
		main function that outputs one or two datasheets in pandas dataframe format (by default)
		'''

		count = 0
		data = []

		for item in self.root:
			count += 1
			row = []
			for subitem in item:
				row.append(subitem.text)
			data.append(row)
		
		if self.subroot:
			id_ = []
			subdata = []
			
			for i in range(count):
				subcount = 0
				
				for item in self.root[i][-1]:
					subcount += 1
				
				for j in range(subcount):
					id_.append(self.root[i][0].text)
					subrow = []
					
					for subitem in self.root[i][-1][j]:
						subrow.append(subitem.text)
					subdata.append(subrow)
				data[i][-1] = subcount
			
			if df:
				var, subvar = self.get_tag()
				data, subdata = pd.DataFrame(data, columns=var), pd.DataFrame(subdata, columns=subvar)
				subdata['numeroseao'] = id_
				return data, subdata
			
			else:
				return data, subdata
		else:
			
			if df:
				var = self.get_tag()
				b = 'No subtrees in:' + self.path
				return pd.DataFrame(data, columns=var), 'No subtrees in: ' + self.path
			
			else:
				return data, 'No subtrees in: ' + self.path
	

























		
		
