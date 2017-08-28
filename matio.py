import scipy.io as sio

imgs = sio.loadmat('data/trainlist_fcn.mat')['trainlist'][0]
for i in imgs:
  print(i)
  img_name = i
  imgs = []
  labels = []
  stp = str(img_name)
  if img_name < 10:
      stp = '0000' + stp
  elif img_name < 100:
      stp = '000' + stp
  elif img_name < 1000:
      stp = '00' + stp
  else:
      stp = '0' + stp
  img_path = 'data/portraitFCN_data/' + stp + '.mat'
  print('img_path : ', img_path)
  imat = sio.loadmat(img_path)['img']
  print('successfully opened :', i)