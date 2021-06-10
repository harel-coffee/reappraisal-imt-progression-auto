#!/user/bin/python


import os
from os.path import join as opj
import shutil

# read csv file with the subject IDs and the response variables
with open("Thomas_Javi_ML_Share_PIP_Reappraisal_IMT_Longitudinal_02_NOV_2020.csv", "r") as f:
	lines = f.readlines()

# Extract subject IDS for this project
subjects_id = [l[:4] for l in lines[1:]]

data_dir = "/home/jrd117/ProjectDrive/PIP/SPM12/First_Level"

# contrast to use in this study

# con_0001.nii =  LookNeg>LookNeut, con_0002.nii = RegNeg>LookNeg
con_names = ["con_0001.nii", "con_0002.nii"]

#copy files
for sub_id in subjects_id:
	output_dir = "./sub-%s" % sub_id
	if os.path.exists(output_dir) is False:
		os.mkdir(output_dir)
	
	src_con_0001 = opj(data_dir, sub_id, "ER", con_names[0])
	dst_con_0001 = opj(output_dir, con_names[0])
	try:
		shutil.copy(src_con_0001, dst_con_0001)
	except:
		print("lookNeg>LookNeut for subject %s does not exist" % sub_id)

        src_con_0002 = opj(data_dir, sub_id, "ER", con_names[1])
        dst_con_0002 = opj(output_dir, con_names[1])
	try:
	        shutil.copy(src_con_0002, dst_con_0002)
	except:
		print("RegNeg>LookNeg for subject %s does not exist" % sub_id)
	
	

	



