def readfile(filename): #,natom,rme=',rname)
 data=[]
 box_=[]
 resid=[]
 rname=[]
 aname=[]
 anum=[]
 xcoord=[]
 ycoord=[]
 zcoord=[]

 f=open(filename,'r')
 natom=int(f.readlines()[1])
 f=open(filename,'r')
 data=f.readlines()
 del data[-1]
 del data[1]
 del data[0]
# print('len(total=',len(total),'natom=',natom,'nframes=',nframe)
# for i in range(total):
#  data.append(total[((natom+3)*i+2):((natom+3)*(i+1)-1)])
#  box_.append(total[(natom+3)*(i+1)-1])
# for i in range(len(data)):
 resid_=[]
 rname_=[]
 aname_=[]
 anum_=[]
 xcoord_=[]
 ycoord_=[]
 zcoord_=[]
 box_xyz=[]
 box=[]
 for row in data:
  resid_.append(row[0:5])
  rname_.append(str(row[5:9]).split())
  aname_.append(str(row[9:15]).split())
  anum_.append(row[15:20])

 for i in range(len(resid_)):
  resid.append(int(resid_[i]))
  rname.append(rname_[i][0])
  aname.append(aname_[i][0])
  anum.append(str(anum_[i]))
 return natom,resid,rname,aname,anum,xcoord,ycoord,zcoord,box

filename='box.gro'
a=readfile(filename)
natom=a[0]
resid=a[1]
rname=a[2]
aname=a[3]
anum=a[4]

#for i in range(len(a[1])):
# print resid[i], rname[i], aname[i], anum[i]

#bond_1=[]
#bond_2=[]
g=open('trans_bond_10res.xvg','w')
print>>g, '; add distance restraints to maintein trans bond ' 

for i in range(len(resid)):
 if (resid[i] == 1 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==2):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1] 

for i in range(len(resid)):
 if (resid[i] == 2 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==3):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]

for i in range(len(resid)):
 if (resid[i] == 3 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==4):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]

for i in range(len(resid)):
 if (resid[i] == 4 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==5):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]

for i in range(len(resid)):
 if (resid[i] == 5 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==6):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]

for i in range(len(resid)):
 if (resid[i] == 6 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==7):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]

for i in range(len(resid)):
 if (resid[i] == 7 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==8):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]

for i in range(len(resid)):
 if (resid[i] == 8 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==9):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]

for i in range(len(resid)):
 if (resid[i] == 9 and aname[i]=='O'):
  bond_1=anum[i]
 if (resid[i]==10):
  if (rname[i][0]=='M' and aname[i]=='CM'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i][0]=='N' and aname[i]=='CB'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (rname[i]=='PRO' and aname[i]=='CD'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.37   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]
  elif (aname[i]=='H'):
   bond_2=anum[i]
   print>>g, bond_1,bond_2,'   1   0.31   999999.0  ; ', aname[int(bond_1)-1], aname[int(bond_2)-1]


