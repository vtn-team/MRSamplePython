import numpy as np
import cv2
from cv2 import aruco

#設定変数
CAM_INDEX = 3 #番号(3)は各自の環境で変わるので適宜変える事、何番なのかは力業で調べるしかないっぽい


#ArUcoボードの生成と表示(これは画像を保存してUnity側でその画像を表示する)
cellSize = 35 #ここは適当
boardDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
board = aruco.CharucoBoard((4, 4), 16*cellSize, 9*cellSize, boardDict)
imboard = board.generateImage((1920, 1080))

#デバッグ表示
cv2.imshow('ArUcoBoard' , imboard)


#検出器をつくる
mdetector = aruco.ArucoDetector(boardDict)  #マーカー検知器
detector = aruco.CharucoDetector(board)     #ボード検知器

cap = cv2.VideoCapture(CAM_INDEX) 
cap.set(cv2.CAP_PROP_FPS, 30) #60出せるなら60がいい…
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #検知が弱かったら上げる事。解像度が対応しているかは調べる事。
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #検知が弱かったら上げる事。解像度が対応しているかは調べる事。

#ArUcoボードの検知とキャリブレーション
while True:
	#読み込み
	ret, frame = cap.read()
	imsize = frame.shape
	
	if ret == False:
		break
	
	key =cv2.waitKey(10)
	if key == 27:
		break
	
	#画像補正
	mat = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
	
	#検知が弱いかカメラを逆にするときはflipをかける(逆さだと検知されない)
	#mat = cv2.flip(frame, -1)
	
	
	#こっちでもできる
	#corners, ids, rejectedImgPoints = mdetector.detectMarkers(mat)
	#mat = aruco.drawDetectedMarkers(mat, corners, ids)
	
#	こっちの方がアクセスが良い
	corners, ids, markerCorners, markerIds = detector.detectBoard(mat);
	mat = aruco.drawDetectedCornersCharuco(mat, corners, ids)
	cv2.imshow('camera' , mat)
	
	#Idが0,1,4,5のマーカーを探して、画面の位置を特定する
	#ct, lt, rt, lb, rb
	mkpos = [[],[],[],[],[]]
	check = []
	
	#要コンバート
	try:
		if np.all(ids != None):
			for i in range(ids.shape[0]):
				id = ids[i,0]
				
				if id == 4:
					check = np.append(check, 0)
					mkpos[0] = (corners[i,0,0],corners[i,0,1])
				if id == 0:
					check = np.append(check, 1)
					mkpos[1] = (corners[i,0,0],corners[i,0,1])
				if id == 2:
					check = np.append(check, 2)
					mkpos[2] = (corners[i,0,0],corners[i,0,1])
				if id == 6:
					check = np.append(check, 3)
					mkpos[3] = (corners[i,0,0],corners[i,0,1])
				if id == 8:
					check = np.append(check, 4)
					mkpos[4] = (corners[i,0,0],corners[i,0,1])
				
	except ValueError as e:
		print(e)
	
	if len(check) != 5:
		continue

	#台形補正
	#四角系で回転を考慮したキャリブレーション
	edgepos = [[0,0],[0,0],[0,0],[0,0]]
	for i in range(1,5):
		edgepos[i-1] = (mkpos[i][0] + mkpos[i][0] - mkpos[0][0], mkpos[i][1] + mkpos[i][1] - mkpos[0][1])
	
	srcPts = np.array(edgepos, dtype=np.float32)
	dstPts = np.array([[0, 0], [imsize[1], 0], [0, imsize[0]], [imsize[1], imsize[0]]], dtype=np.float32)
	
	rMat = cv2.getPerspectiveTransform(srcPts, dstPts);
	mat2 = cv2.warpPerspective(mat, rMat, (imsize[1], imsize[0]));
	
	cv2.imshow('cropped' , mat2)
	
cap.release()
cv2.destroyAllWindows()
