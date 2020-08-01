import os
import platform
def gci(filepath):

	files = os.listdir(filepath)
	for fi in files:
		fi_d = os.path.join(filepath,fi)
		if os.path.isdir(fi_d):
			#print(os.path.join(filepath, fi_d))
			gci(fi_d)
		else:
			sysstr = platform.system()
			if (sysstr == "Windows"):
			# Win:
				os.system('traces2text.exe v4_2 '+str(os.path.join(filepath,fi_d))+' '+'./trace/'+str(fi_d).split('\\')[-1].replace('trc.bz2','txt'))
			else:
				# Unix:
				os.system('traces2text.exe v4_2 '+str(os.path.join(filepath,fi_d))+' '+'./trace/'+str(fi_d).split('\\')[-1].replace('trc.bz2','txt'))
			print((fi_d).split('\\')[-1])

path='.\DPA_contestv4_2'
gci(path)



