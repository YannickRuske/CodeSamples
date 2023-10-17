import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk,Image
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.io as sio
import sys


sys.path.append('/home/fflab28m/Python/ffwd_python')
from doocs import doocsCamera

dpi = 94

class GUI:

    def __init__(self):

	# Camera controller
        self.especCamPath = 'FLASH.DIAG/FLASHFWDCAM4.CAM/ElectronSpectrometer3'
        self.especCam = doocsCamera('FLASH.DIAG/FLASHFWDCAM4.CAM/ElectronSpectrometer3')

        self.imagesize = (0,0)
        # das Hauptfenster
        self.GUI = tk.Tk()

        self.objpoints = []
        self.imgpoints = []
        self.llCorners = np.array([0,0])
        self.ulCorners = np.array([0,0])
        self.lrCorners = np.array([0,0])
        self.urCorners = np.array([0,0])
        self.numberOfImages = 0
        self.cornersShown=False
        self.picLoaded = False
        self.showSpectrum = False

        self.mainFrame = tk.Frame(self.GUI)

        #image frame
        self.pictureFrame = tk.Frame(self.mainFrame,borderwidth = 1, relief = 'ridge', padx=10, pady=10)
        self.pictureFrame.grid_rowconfigure(0, weight=1)
        self.pictureFrame.grid_rowconfigure(1, weight=1)
        self.pictureFrame.grid_columnconfigure(0, weight=1)

        #main image
        self.emptyImage = Image.new('RGB',[int(1024),int(786)])
        self.imageShown = self.emptyImage
        self.imageShownCrop= self.emptyImage
        self.noCorners = self.emptyImage
        self.noCornersCrop = self.emptyImage
        emptyImage_tk = ImageTk.PhotoImage(self.emptyImage)
        self.pictureLabel = tk.Label(self.pictureFrame,image=emptyImage_tk)
        self.pictureLabel.grid(column=0, row=0, sticky='NWS')

        #spectrum plot
        self.spectrumFigure = plt.figure(figsize = (float((self.imageShown.width)/float(dpi)),3), facecolor = "white", frameon=True)
        self.ax=self.spectrumFigure.add_subplot(111)
        self.ax.axis('off')
        self.spectrumCanvas = FigureCanvasTkAgg(self.spectrumFigure, master = self.pictureFrame)
        self.spectrumCanvas.draw()
        self.spectrumWidget=self.spectrumCanvas.get_tk_widget()
        self.spectrumWidget.grid(column=0, row=1, sticky='NWS')

        #grid image frame
        self.pictureFrame.grid(column=0,row=0,rowspan=2,sticky='NE')

        #frame for user input/action
        self.inputFrame = tk.Frame(self.mainFrame)

        #frame for all ROI stuff
        self.ROIFrame = tk.Frame(self.inputFrame)

        #frame for image cropping/rotaion
        self.cropFrame = tk.Frame(self.ROIFrame)

        #rotation angle
        self.label_angle = tk.Label(self.cropFrame, text="angle:")
        self.angle = tk.Entry(self.cropFrame)
        self.angle.insert(0,'0')
        self.label_angle.grid(column=3, row=0, sticky = 'W')
        self.angle.grid(column=4, row=0, sticky = 'E')

        #crop pixel right
        self.label_x1 = tk.Label(self.cropFrame, text='cut left:')
        self.x1entry = tk.Entry(self.cropFrame)
        self.x1entry.insert(0,'0')
        self.label_x1.grid(column=3, row=1, sticky = 'W')
        self.x1entry.grid(column=4 ,row=1, sticky = 'E')

        #crop pixel left
        self.label_x2 = tk.Label(self.cropFrame, text='cut right:')
        self.x2entry = tk.Entry(self.cropFrame)
        self.x2entry.insert(0,'0')
        self.label_x2.grid(column=3, row=2, sticky = 'W')
        self.x2entry.grid(column=4 ,row=2, sticky = 'E')

        #crop pixel top
        self.label_y1 = tk.Label(self.cropFrame, text='cut top:')
        self.y1entry = tk.Entry(self.cropFrame)
        self.y1entry.insert(0,'0')
        self.label_y1.grid(column=3, row=3, sticky = 'W')
        self.y1entry.grid(column=4 ,row=3, sticky = 'E')

        #crop pixel bottom
        self.label_y2 = tk.Label(self.cropFrame, text='cut bottom:')
        self.y2entry = tk.Entry(self.cropFrame)
        self.y2entry.insert(0,'0')
        self.label_y2.grid(column=3, row=4, sticky = 'W')
        self.y2entry.grid(column=4 ,row=4, sticky = 'E')

        #rotate/crop
        self.rotateButton = tk.Button(self.cropFrame, text="rotate/crop", command = self.rotateCropImage, width=30)
        self.rotateButton.grid(column=3, row=5,columnspan=2, sticky = 'E',pady=5)

        self.cropFrame.grid(column=0,row=0)

        #Frame for saving/loading
        self.ImageControlFrame = tk.Frame(self.inputFrame)

        #get new image from camera
        self.ImageButton = tk.Button(self.ImageControlFrame, text="take new camera image", command = self.getImage, width=30)
        self.ImageButton.grid(column=3, row=11,columnspan=2, sticky = 'E',pady=5)
        
        #espec camera path
        self.CameraPathEntry = tk.Entry(self.ImageControlFrame, width=36)
        self.CameraPathEntry.insert(0,self.especCamPath)
        self.CameraPathEntry.grid(column=3, row=12,columnspan=2, sticky = 'E',pady=5)
        
        #load image from hard drive
        self.ImageFolderButton = tk.Button(self.ImageControlFrame, text="load image from hard drive", command = self.loadImage, width=30)
        self.ImageFolderButton.grid(column=3, row=10,columnspan=2, sticky = 'E',pady=5)

        #frame for corners (detect+show/hide)
        self.CornerFrame = tk.Frame(self.inputFrame)

        #pattern dimensions
        self.label_dimx = tk.Label(self.CornerFrame, text="pattern width:")
        self.label_dimy = tk.Label(self.CornerFrame, text="pattern height:")
        self.dimx_entry = tk.Entry(self.CornerFrame)
        self.dimy_entry = tk.Entry(self.CornerFrame)
        self.dimx_entry.insert(0,'121')
        self.dimy_entry.insert(0,'9')
        self.label_dimx.grid(column=3, row=12, sticky = 'W')
        self.dimx_entry.grid(column=4 ,row=12, sticky = 'E')
        self.label_dimy.grid(column=3, row=13, sticky = 'W')
        self.dimy_entry.grid(column=4 ,row=13, sticky = 'E')

        #show/hide corners
        self.CornerButton = tk.Button(self.CornerFrame, text="toggle corners on/off", command = self.showCorners, width=30)
        self.CornerButton.grid(column=3, row=14,columnspan=2, sticky = 'E',pady=5)

        #Frame for lens calibration
        self.CalibFrame = tk.Frame(self.inputFrame)

        #add image to calibration
        self.addToCalibButton = tk.Button(self.CalibFrame, text='add current image to calibration', command = self.addToCalib, width=30)
        self.addToCalibButton.grid(column=0, row=0,columnspan=2,pady=5)

        self.addFolderToCalibButton = tk.Button(self.CalibFrame, text='add multiple images to calibration', command = self.addFolderToCalib, width=30)
        self.addFolderToCalibButton.grid(column=0, row=1,columnspan=2,pady=5)

        #path for calibration file
        self.CalibParamPath = tk.Button(self.CalibFrame, text='choose path for calib file', command = self.setCalibPath, width=30)
        self.CalibParamPath.grid(column=0, row=2,columnspan=2,pady=5)
        self.CalibPathEntry = tk.Entry(self.CalibFrame, width=36)
        self.CalibPathEntry.insert(0,os.getcwd())
        self.CalibPathEntry.grid(column=0, row=3,columnspan=2,pady=5)

        #name for calibration file
        self.CalibNameLabel = tk.Label(self.CalibFrame, text='file name:')
        self.CalibNameLabel.grid(column=0, row=4, pady=5, sticky='W')
        self.CalibNameEntry = tk.Entry(self.CalibFrame, width=27)
        self.CalibNameEntry.insert(0,'CalibParams')
        self.CalibNameEntry.grid(column=1, row=4, pady=5, sticky='E')

        #calibrate and save file
        self.saveCalibButton = tk.Button(self.CalibFrame, text = 'save calibration', command = self.saveCalibParam, width=30)
        self.saveCalibButton.grid(column=0, row=5,columnspan=2, pady=5)

        #undistort image
        self.undistortButton = tk.Button(self.CalibFrame, text = 'undistort shown image', command = self.undistort, width=30)
        self.undistortButton.grid(column=0, row=6, columnspan=2, pady=5)

        #spectrum control
        self.spectrumFrame = tk.Frame(self.inputFrame)

        self.spectrumButton = tk.Button(self.spectrumFrame, text='show spectrum', command = self.spectrum, width=30)
        self.spectrumButton.grid(column=0, row=0, pady=5, sticky='E')

        #pixel screen calibration frame
        self.pixelFrame = tk.Frame(self.inputFrame)

        self.leftSpaceLabel = tk.Label(self.pixelFrame, text='space left side (mm):')
        self.leftSpaceEntry = tk.Entry(self.pixelFrame, width=15)
        self.leftSpaceEntry.insert(0,'0')
        self.topSpaceLabel = tk.Label(self.pixelFrame, text='space upper side (mm):')
        self.topSpaceEntry = tk.Entry(self.pixelFrame, width=15)
        self.topSpaceEntry.insert(0,'0')
        self.squareSizeLabel = tk.Label(self.pixelFrame, text='pattern square size (mm):')
        self.squareSizeEntry = tk.Entry(self.pixelFrame, width=15)
        self.squareSizeEntry.insert(0,'0')

        self.leftSpaceLabel.grid(column=0,row=0,sticky='W')
        self.leftSpaceEntry.grid(column=1, row=0)
        self.topSpaceLabel.grid(column=0,row=1,sticky='W')
        self.topSpaceEntry.grid(column=1, row=1)
        self.squareSizeLabel.grid(column=0,row=2,sticky='W')
        self.squareSizeEntry.grid(column=1, row=2)

        #grid all frames in input frame
        self.ROIFrame.grid(column=0, row=0)
        self.ImageControlFrame.grid(column=0,row=1,sticky='N',pady=5)
        self.CornerFrame.grid(column=0,row=2,sticky='N',pady=5)
        self.pixelFrame.grid(column=0,row=3,sticky='N',pady=5)
        self.CalibFrame.grid(column=0,row=4,sticky='N',pady=5)
        self.spectrumFrame.grid(column=0,row=5,sticky='N',pady=5)

        #grid input frame
        self.inputFrame.grid(column=1,row=0,rowspan=1,sticky='N',pady=5, padx=5)

        self.mainFrame.grid(column=0,row=0)


        #'build' GUI
        self.GUI.mainloop()


    def rotateCropImage(self):
        alpha = float(self.angle.get())
        x1 = float(self.x1entry.get())
        x2 = float(self.x2entry.get())
        y1 = float(self.y1entry.get())
        y2 = float(self.y2entry.get())

        try:
            self.imageShownCrop=self.imageShown.rotate(alpha).crop((x1,y1,self.imagesize[0]-x2,self.imagesize[1]-y2)).resize((1024,int(1024/(self.imagesize[0]-x1-x2)*(self.imagesize[1]-y1-y2))))
            self.noCornersCrop=self.noCorners.rotate(alpha).crop((x1,y1,self.imagesize[0]-x2,self.imagesize[1]-y2)).resize((1024,int(1024/(self.imagesize[0]-x1-x2)*(self.imagesize[1]-y1-y2))))
        except ZeroDivisionError:
            tk.messagebox.showerror('No Image','No image found.')
        newImage = ImageTk.PhotoImage(self.imageShownCrop, Image.ANTIALIAS)
        self.pictureLabel.configure(image=newImage)

        try:
            self.ax.clear()
            self.ax.axis('off')
        except AttributeError:
            pass

        if self.showSpectrum:
            self.plotSpectrum()
        self.GUI.mainloop()

    def plotSpectrum(self):
        self.brightness()
        self.ax.axis('on')
        self.ax.plot(self.pixelArray, self.spectrumArray)
        self.ax.tick_params(axis='both',which='both',direction='in', left=True, right=True, top=True, bottom=True, labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        self.ax.margins(0)
        matplotlib.pyplot.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        self.spectrumCanvas.draw()


    def getImage(self):
        try:
            self.especCam.getSingle()
            image = self.especCam.getData()
            self.imageShown = Image.fromarray(image)
            self.noCorners = self.imageShown
            self.cornersShown=False
            self.picLoaded=True
            self.rotateCropImage()
        except:
            tk.messagebox.showerror('Error','Image could not be taken. Check if camera is on and adress is correct.')

    def loadImage(self):
        try:
            loadedImage = Image.open(tk.filedialog.askopenfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("png files","*.png"),("all files","*.*"))))
            width, height = loadedImage.size
            self.imagesize = (width,height)
            self.noCorners = loadedImage
            self.imageShown = loadedImage
            self.cornersShown=False
            self.picLoaded=True
            self.rotateCropImage()
        except AttributeError:
            pass
#        self.GUI.mainloop()


    def showCorners(self):
        if self.cornersShown:
            self.cornersShown=False
            self.imageShown = self.noCorners
            self.rotateCropImage()
            self.GUI.mainloop()
        else:
            dimx=int(self.dimx_entry.get())
            dimy=int(self.dimy_entry.get())
            imageShownNp,cornersFound = self.getCorners(self.noCorners,dimx,dimy)
            self.imageShown = Image.fromarray(imageShownNp)
            self.rotateCropImage()
            self.GUI.mainloop()

    def getCorners(self,pilimg,dimx,dimy):
        a = dimx
        b = dimy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        img = np.asarray(pilimg).copy()
        ret, corners = cv2.findChessboardCorners(img, (b,a), None)
        if ret == True:
            self.corners2=cv2.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            cv2.drawChessboardCorners(img, (b,a), self.corners2, ret)
            self.cornersShown=True
        else:
            tk.messagebox.showerror("Error", "No corners found, check lighting and pattern size.")
        return img,ret

    def addToCalib(self):
        a=int(self.dimx_entry.get())
        b=int(self.dimy_entry.get())
        c=a*b
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((c,3), np.float32)
        objp[:,:2] = np.mgrid[0:b,0:a].T.reshape(-1,2)
        img = np.asarray(self.noCorners).copy()
        ret, corners = cv2.findChessboardCorners(img, (b,a), None)
        self.imgShape = img.shape[::-1]
        if ret == True:
            self.objpoints.append(objp)
            self.corners2=cv2.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
            data=self.corners2[:, 0, :]
            self.numberOfImages+=1
            self.llCorners = np.add(data[0],self.llCorners)
            self.ulCorners = np.add(self.ulCorners,data[b-1])
            self.lrCorners = np.add(self.lrCorners,data[b*(a-1)])
            self.urCorners = np.add(self.urCorners,data[c-1])
            self.imgpoints.append(corners)
            print('done')
        else:
            tk.messagebox.showerror("Error", "Pattern not found, check lighting and pattern size.")

    def addFolderToCalib(self):
        files = list(tk.filedialog.askopenfilenames(initialdir = os.getcwd(),title = "Select file",filetypes = (("png files","*.png"),("all files","*.*"))))
        worked = 0
        tried = 0
        for file in files:
            a=int(self.dimx_entry.get())
            b=int(self.dimy_entry.get())
            c=a*b
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            objp = np.zeros((c,3), np.float32)
            objp[:,:2] = np.mgrid[0:b,0:a].T.reshape(-1,2)
            img = np.asarray(Image.open(file)).copy()
            tried +=1
            ret, corners = cv2.findChessboardCorners(img, (b,a), None)
            self.imgShape = img.shape[::-1]
            if ret == True:
                self.objpoints.append(objp)
                self.corners2=cv2.cornerSubPix(img,corners, (11,11), (-1,-1), criteria)
                data=self.corners2[:, 0, :]
                self.numberOfImages+=1
                self.llCorners = np.add(data[0],self.llCorners)
                self.ulCorners = np.add(self.ulCorners,data[b-1])
                self.lrCorners = np.add(self.lrCorners,data[b*(a-1)])
                self.urCorners = np.add(self.urCorners,data[c-1])
                self.imgpoints.append(corners)
                worked +=1
        tk.messagebox.showinfo('calibration finished',str(worked)+' of '+str(tried)+' images succesfully added to calibration. '+str(tried-worked)+' images could not be used.')




    def calibParams(self):
        return cv2.calibrateCamera(self.objpoints, self.imgpoints, self.imgShape, None, None)

    def setCalibPath(self):
        dir_name = tk.filedialog.askdirectory()
        self.CalibPathEntry.delete(0, 'end')
        self.CalibPathEntry.insert(0,dir_name)

    def saveCalibParam(self):
        try:
            imagesize = self.imagesize
            rotation = float(self.angle.get())
            x1 = float(self.x1entry.get())
            x2 = float(self.x2entry.get())
            y1 = float(self.y1entry.get())
            y2 = float(self.y2entry.get())
            roi = (x1,y1,imagesize[0]-x1-x2,imagesize[1]-y1-y2)
            alongscreen, perpscreen = self.getPixelPosition()
            alongscreen = alongscreen[int(x1):int(imagesize[0]-x2)]
            perpscreen = perpscreen[int(y1):int(imagesize[1]-y2)]
            self.RMS_err, self.mtx, self.dist, self.rvecs, self.tvecs = self.calibParams()
            self.newcameramtx, self.roi=cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, self.imagesize, 1, self.imagesize)
            paramDict = {}

            paramDict['imagesize'] = imagesize
            paramDict['rotation'] = rotation
            paramDict['roi'] = roi
            paramDict['alongscreen'] = alongscreen
            paramDict['perpscreen'] = perpscreen
            paramDict['RMS_error'] = self.RMS_err
            paramDict['mtx'] = self.mtx
            paramDict['dist'] = self.dist
            paramDict['newcameramtx'] = self.newcameramtx

            path = self.CalibPathEntry.get()
            name = self.CalibNameEntry.get()+'.mat'


            tk.messagebox.showinfo("Calibration saved","Calibration saved, mean error="+str(self.RMS_err))

            if  os.path.isfile(os.path.join(path,name)):
                MsgBox = tk.messagebox.askokcancel ('File allready exists','File allready exists. Overwrite file?',icon = 'warning')
                if MsgBox == 'ok':
                    sio.savemat(os.path.join(path,name), paramDict)
                if MsgBox == 'cancel':
                    pass
            else:
                sio.savemat(os.path.join(path,name), paramDict)


        except AttributeError:
            tk.messagebox.showerror('No images','Nothing to calibrate from. Add image(s) to calibration first.')




    def undistort(self):
        try:
            img = np.asarray(self.imageShown)
            h,  w = img.shape[:2]
            # undistort
            dst = cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
            # crop the image
            x, y, w, h = self.roi
    #        dst = dst[y:y+h, x:x+w]
            self.imageShown = Image.fromarray(dst)
            self.rotateCropImage()
        except AttributeError:
            tk.messagebox.showerror('Calibration not found','Undistortion failed, no calibration parameters found. Calibrate before undistortion.')
    def getPixelPosition(self):

        dim_x=int(self.dimx_entry.get())
        dim_y=int(self.dimy_entry.get())
        spaceLeft = float(self.leftSpaceEntry.get())
        spaceTop = float(self.topSpaceEntry.get())
        squareSize = float(self.squareSizeEntry.get())

        ulCornerMean = [self.ulCorners[0]/self.numberOfImages,self.ulCorners[1]/self.numberOfImages]
        urCornerMean = [self.urCorners[0]/self.numberOfImages,self.urCorners[1]/self.numberOfImages]
        llCornerMean = [self.llCorners[0]/self.numberOfImages,self.llCorners[1]/self.numberOfImages]
        lrCornerMean = [self.lrCorners[0]/self.numberOfImages,self.lrCorners[1]/self.numberOfImages]

        xPixelCoords = [0.5*(ulCornerMean[0]+llCornerMean[0]),0.5*(urCornerMean[0]+lrCornerMean[0])]
        yPixelCoords = [0.5*(lrCornerMean[1]+llCornerMean[1]),0.5*(urCornerMean[1]+ulCornerMean[1])]
        xMmPerPixel = abs(((dim_x-1)*squareSize)/(xPixelCoords[1]-xPixelCoords[0]))
        yMmPerPixel = abs(((dim_y-1)*squareSize)/(yPixelCoords[1]-yPixelCoords[0]))
        alongscreen = []
        perpscreen = []
        for i in range(self.imagesize[0]):#horizontal image size
            alongscreen.append((i-xPixelCoords[0])*xMmPerPixel+spaceLeft+squareSize)
        for j in range(self.imagesize[1]):#vertical image size
            perpscreen.append((j-yPixelCoords[1])*yMmPerPixel+spaceTop+squareSize)
        return alongscreen,perpscreen


    def brightness(self):
        imgData = self.noCornersCrop
        imgData = np.asarray(imgData)                                          #[x,y]
        imgDataTransposed = np.transpose(imgData)                              #[y,x]
        self.spectrumArray=[]
        for column in imgDataTransposed:
            self.spectrumArray.append(np.sum(column))
        self.pixelArray = np.arange(0,len(self.spectrumArray))

    def spectrum(self):
        self.showSpectrum = True
        self.rotateCropImage()



if __name__ == '__main__':
  program = GUI()