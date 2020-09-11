# import the necessary packages
import uuid
import os
import datetime
class TempImage:
	def __init__(self, basePath="./", ext=".jpg"):
		timestamp = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
		# construct the file path
		# creating a folder named data 
		if not os.path.exists(timestamp):
			os.makedirs(timestamp) 
		self.path = "{base_path}/{ts}/{ts}{ext}".format(base_path=basePath, ts=timestamp, ext=ext)

	def cleanup(self):
		# remove the file
		os.remove(self.path)
