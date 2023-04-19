## Version 2.0
## update : 2023.04.19

import numpy as np
import json
import os
from collections import OrderedDict

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

#np.set_printoptions(threshold=np.inf,linewidth=np.inf)
file_path="./llff.json"
np_path="./barf_ouput_pose.npy"
img_path="./llff/images"
path=['./llff/images/']
pose_path="./poses_bounds.npy"
img_list=sorted(os.listdir(img_path))


list_homo=[0.0 , 0.0 , 0.0 , 1.0]

np_data=np.load(np_path)
print(np_data.shape)
np_data=np_data.tolist()  # numpy to list

for i in range(0,len(np_data)):	 # make homogeneous matrix
   np_data[i].append(list_homo)

np_data=np.array(np_data) # list to numpy


with open(file_path,'r') as file: # open json file
   data=json.load(file)

for i in range(0,len(img_list)): # make image_path like json file
   img_list[i]=path[0]+img_list[i]


up = np.zeros(3)
np_data=np.linalg.inv(np_data)


for change_num in range(0,len(img_list)):
   for i in data["frames"]:   
      if i["file_path"] == img_list[change_num]:
         np_data[change_num][0:3,2] *= -1 # flip the y and z axis
         np_data[change_num][0:3,1] *= -1
         np_data[change_num]=np_data[change_num][[1,0,2,3],:]
         np_data[change_num][2,:] *= -1  # flip whole world upside down
         up += np_data[change_num][0:3,1]
         print("change num", change_num)
         print("up ",up)

for change_num in range(0,len(img_list)):
   for i in data["frames"]:   
      if i["file_path"] == img_list[change_num]:
         i["transform_matrix"][0]=np_data[change_num][0]
         i["transform_matrix"][1]=np_data[change_num][1]
         i["transform_matrix"][2]=np_data[change_num][2]
         i["transform_matrix"][3]=np_data[change_num][3]

print("change transformation")
up = up / np.linalg.norm(up)
print("up vector was", up)
R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
R = np.pad(R,[0,1])
R[-1, -1] = 1

for f in data["frames"]:
   f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

# find a central point they are all looking at
print("computing center of attention...")
totw = 0.0
totp = np.array([0.0, 0.0, 0.0])
for f in data["frames"]:
   mf = f["transform_matrix"][0:3,:]
   for g in data["frames"]:
      mg = g["transform_matrix"][0:3,:]
      p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
      if w > 0.00001:
         totp += p*w
         totw += w
if totw > 0.0:
   totp /= totw
print(totp) # the cameras are looking at totp

for f in data["frames"]:
   f["transform_matrix"][0:3,3] -= totp

avglen = 0.
nframes = len(data["frames"])
for f in data["frames"]:
   avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
avglen /= nframes
print("avg camera distance from origin", avglen)
for f in data["frames"]:
   f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"


for f in data["frames"]:
   f["transform_matrix"] = f["transform_matrix"].tolist()
print(nframes,"frames")


with open(file_path,'w') as make_file: # overwrite json
   json.dump(data,make_file,indent=2)

print("complete!")
