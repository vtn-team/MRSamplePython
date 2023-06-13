import numpy as np
import cv2
from cv2 import aruco


#ArUcoボードの生成と表示(これは画像を保存してUnity側でその画像を表示する)
cellSize = 35
boardDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
board = aruco.CharucoBoard((9, 5), 16*cellSize, 9*cellSize, boardDict)
imboard = board.generateImage((1920, 1080))

#デバッグ表示
cv2.imshow('ArUcoBoard' , imboard)


#検出器をつくる
mdetector = aruco.ArucoDetector(boardDict)  #マーカー検知器
detector = aruco.CharucoDetector(board)     #ボード検知器

cap = cv2.VideoCapture(3) #番号は各自の環境で変わる
cap.set(cv2.CAP_PROP_FPS, 30) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
	#mat = cv2.flip(frame, -1)
	
	#こっちでもできる
	#corners, ids, rejectedImgPoints = mdetector.detectMarkers(mat)
	#mat = aruco.drawDetectedMarkers(mat, corners, ids)
	
#	こっちの方がアクセスが良い
	corners, ids, markerCorners, markerIds = detector.detectBoard(mat);
	mat = aruco.drawDetectedCornersCharuco(mat, corners, ids)
	cv2.imshow('camera' , mat)
	
	#Idが0,1,4,5のマーカーを探して、画面の位置を特定する
	p0 = p1 = p8 = p9 = None
	
	#要コンバート
	try:
		if np.all(ids != None):
			for i in range(ids.shape[0]):
				id = ids[i,0]
				
				if id == 0:
					p0 = (corners[i,0,0],corners[i,0,1])
				if id == 1:
					p1 = (corners[i,0,0],corners[i,0,1])
				if id == 8:
					p8 = (corners[i,0,0],corners[i,0,1])
				if id == 9:
					p9 = (corners[i,0,0],corners[i,0,1])

				#公式のキャリブレーション
#				cameraMatrixInit = np.array([
#					[ 1000.,    0., imsize[1]/2.],
#					[    0., 1000., imsize[0]/2.],
#					[    0.,    0.,           1.]])
#				distCoeffsInit = np.zeros((5,1))
#				retval, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
#					listCharucoCorners,
#					listCharucoIds,
#					board,
#					(imsize[1],imsize[0]),
#					cameraMatrixInit,
#					distCoeffsInit
#				)
#				print(retval)
	except ValueError as e:
		print(e)
	
	if p0 == None or p1 == None or p8 == None or p9 == None:
		continue

	#台形補正
	sub = ((p1[0] - p0[0]), (p8[1] - p0[1]));
	pos1 = ((p0[0] - sub[0]), (p0[1] - sub[1]))
	pos2 = ((p1[0] + sub[0] * 7), (p8[1] + sub[1]*3))

	print(p0)
	print(p1)
	print(p8)
	print(p9)
	print(sub)
	print(pos1)
	print(pos2)

	srcPts = np.array([pos1, [pos2[0], pos1[1]], [pos1[0], pos2[1]], pos2], dtype=np.float32)
	dstPts = np.array([[0, 0], [imsize[1], 0], [0, imsize[0]], [imsize[1], imsize[0]]], dtype=np.float32)
	
	rMat = cv2.getPerspectiveTransform(srcPts, dstPts);
	mat2 = cv2.warpPerspective(mat, rMat, (imsize[1], imsize[0]));
	
	cv2.imshow('cropped' , mat2)
	
cap.release()
cv2.destroyAllWindows()
