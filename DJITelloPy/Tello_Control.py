from djitellopy import Tello
import cv2, math, time
import threading
import datetime
import os
from os import path
from cv2 import aruco
import matplotlib as mpl
import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageTk
import tkinter as tki
from tkinter import Toplevel, Scale
from threading import Thread, Event

class TelloC:
    def __init__(self):  

        ### Osnoven program za drona ###
        # Ne spreminjat -> #
        self.root = tki.Tk()
        self.root.title("TELLO Controller")

        # Create a label to display the video stream
        self.image_label = tki.Label(self.root)
        self.image_label.pack()

        self.tello = Tello()
        
        self.frame = None               # frame read from h264decoder and used for pose recognition 
        self.frameCopy = None
        self.frameProc = None
        self.thread = None              # thread of the Tkinter mainloop
        self.stopEvent = None 
        self.lastPrintTime = 0 
        
        # control variables
        self.distance = 0.2             # default distance for 'move' cmd
        self.degree = 30                # default degree for 'cw' or 'ccw' cmd

        self.waitSec = 0.1
        self.oldTime = 0
        self.TR = None
        self.Tvec = None
        self.Rvec = None
        #self.state = 1 # ???????????????

        #--- Naše spremenlivke ---#
        self.flightState = 0            # glavna spremenljivka stanja leta
        self.DroneRead = 0              # kdaj dovolimo komunikacijo z dronom
        self.batteryCnt = 0             # vsakih nekaj ciklov ko se kliče baterijo

        # State spremenljivke - stanje, ki ga dron izvaja (da ni številk ampak besede)
        self.state_default = 0          # osnovno stanje 
        self.state_search = 1           # iskanje aruco značke
        self.state_aligne_move = 2      # poravnava z značko
        self.state_go = 3               # pojdi skozi obroč
        self.state_flip = 4             # naredi flip
        self.state_landign = 5          # pristani
        self.state_off = 6              # ugasni se

        self.arucoId = 2                # spreminjanje iskane aruco značke   
        self.arucoList = [0,1,2,3,4,5]  # vse možne aruco značke
        self.arucoFound = 0             # trenutna aruco značka najdena
        self.arucoDone = 0              # 0 - ni še preletel, 1 - je preletel (za namen flipa da ve kdaj naj ga nardi)
        self.arucoNext = 0              # 1 - naslednja značka je višje od trenutne, -1 - naslednja značka je nižje od trenutne, 0 - naslednja značka je vzporedno s trenutno
        self.T1Next = [0, 0, 0, 0]      # T1 matrika naslednje značke
        self.searchProtect = 0          # zaščita da ne najde prav vsake packe kot aruco
        self.brake = 0                  # žlajf ko pride v krog
        self.flipDone = 0               # proži po flipu za rotacijo in pristajanje 
        self.droneRotate = 0            # obrat po prvih treh krogih

        self.visina = 0                 # trenutna visina drona
        self.visinaOld = 0              # za detekcijo kroga
        self.deltaVisina = 20           # sprememba visine pri iskanju značke (v cm)
        self.searchyaw = 0              # za spremljanje premikanja za yaw pri iskanju
        self.pricakovanaVisina = 0      # ciljna višina po ukazu premika
        self.ukaz = 0                   # vrsta ukaza ob iskanju značke
        self.ukazOld = 0                # vrsta prejšnjega ukaza ob iskanju značke
        self.landZaZih = 0              # pristani ne glede na karkoli

        # PID - ločena funkcija za izračun #
        self.sample = 0.032                                         # sample time povprečna vrednost glede na default sample rate ControlAll
        self.dt = self.sample           
        # PID - X (naprej/nazaj)
        self.Kpx = 0.008                                            # Člen: P 0.010
        self.Kix = 0.050                                            # Člen: I 0.050
        self.Kdx = 0.000                                            # Člen: D 0.000
        self.A0x = self.Kpx + self.Kix*self.dt + self.Kdx/self.dt   # poenostavitev
        self.A1x = -self.Kpx - 2*self.Kdx/self.dt                   # poenostavitev
        self.A2x = self.Kdx/self.dt                                 # poenostavitev
        # PID - Y (levo/desno)
        self.Kpy = 0.30                                             # Člen: P 0.30
        self.Kiy = 0.04                                             # Člen: I 0.04
        self.Kdy = 0.02                                             # Člen: D 0.02
        self.A0y = self.Kpy + self.Kiy*self.dt + self.Kdy/self.dt   # poenostavitev
        self.A1y = -self.Kpy - 2*self.Kdy/self.dt                   # poenostavitev
        self.A2y = self.Kdy/self.dt                                 # poenostavitev
        # PID - Z (gor/dol)
        self.Kpz = 0.12                                             # Člen: P 0.08
        self.Kiz = 0.03                                             # Člen: I 0.03
        self.Kdz = 0.01                                             # Člen: D 0.01
        self.A0z = self.Kpz + self.Kiz*self.dt + self.Kdz/self.dt   # poenostavitev
        self.A1z = -self.Kpz - 2*self.Kdz/self.dt                   # poenostavitev
        self.A2z = self.Kdz/self.dt                                 # poenostavitev
        # PID - YAW (rotacija)
        self.Kpr = 0.50                                             # Člen: P 0.40
        self.Kir = 0.05                                             # Člen: I 0.05
        self.Kdr = 0.00                                             # Člen: D 0.00
        self.A0r = self.Kpr + self.Kir*self.dt + self.Kdr/self.dt   # poenostavitev
        self.A1r = -self.Kpr - 2*self.Kdr/self.dt                   # poenostavitev
        self.A2r = self.Kdr/self.dt                                 # poenostavitev
        
        n = 3 # napake
        m = 4 # osi 
        self.napaka = [[0 for k in range(n)] for j in range(m)]     # e(t), e(t-1), e(t-2), list 4x3
        self.izhod = [0, 0, 0, 0]                                   # Običajno trenutna vrednost aktuatorja
        self.hitrost = [0, 0, 0, 0]                                 # Hitrosti v vse tri smeri in yaw
        self.razdalja = [100, 100, 100, 100]                        # Razdalja v vse tri smeri in yaw
        self.radij = 0                                              # Razdalja do centra kroga
        self.radijZ = 0                                             # Za računanje radija
        self.radijY = 0                                             # Za računanje radija
        self.ravnoCnt = 0                                           # Counter za resnično ravno pozicijo
        self.korakiSkozi = 0                                        # Število ponovitev komande za let skozi krog
        self.bubble = 0                                             # D je v bubblu

        # Testne spremenlivke
        self.freq = 0
        self.freqNow = 0

        fname = './DJITelloPy/calib.txt'
        self.cameraMatrix = None
        self.distCoeffs = None
        self.numIter = 1
        self.specific_point_aruco_1 = np.array([[0], [0.2], [1]]) 
        self.specific_point_aruco_2 = np.array([[0], [0.2], [-0.5]]) 
        self.Step_1 = True
        self.prev_T1_filtered = None
        self.prev_T2_filtered = None
        self.last_call_T1_filtered = None

        self.cur_fps = 0 
        self.frame_count = 0
        self.last_fps_calculation = time.time()

        if path.exists(fname):
            self.cameraMatrix = np.loadtxt(fname, max_rows=3, dtype='float', unpack=False) 
            self.distCoeffs = np.loadtxt(fname, max_rows=1,skiprows=3, dtype='float', unpack=False)

        self.tello.connect()
        self.lock = threading.RLock()
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

        # Persistent thread control
        self.controlEvent = Event()
        self.controlArgs = None

        ### Začetek poleta ###
        self.controlEnabled = True 
        self.takeoffEnabled = True # !!!!!!!!!!!!!!!!!!!
        self.landEnabled = True

        # Thread za video
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.daemon = True
        self.thread.start()

        # "Timer" thread
        self.droneData = Thread(target=self.getDroneData, args=())
        self.droneData.daemon = True
        self.droneData.start()

        # Start the persistent control thread
        self.controlThread = Thread(target=self.persistentControlLoop) # V funkciji persistentControlLoop se kliče funkcija ControlAll, ki kliče funkcijo TakeOff
        self.controlThread.daemon = True
        self.controlThread.start()

        self.root.bind('<KeyPress>', self.on_key_press)

        # ˇˇˇ Ne briši ˇˇˇ #
        self.root.mainloop()        

    # Pulling timer
    def getDroneData(self):
        while(1):
            if self.flightState > 0:
                time.sleep(1)
                self.DroneRead = 1
            else:
                time.sleep(self.sample)
                self.DroneRead = 1

    
    def persistentControlLoop(self):
        print(self.stopEvent.is_set())
        while not self.stopEvent.is_set():
            # Wait for the signal to control
            self.controlEvent.wait()
            # Reset the event to wait for the next signal
            self.controlEvent.clear()
            
            # Safely retrieve arguments for controlAll
            with self.lock:
                if self.controlArgs:
                    T1, T2, yaw = self.controlArgs
                    if self.controlEnabled:
                        self.controlAll(T1, T2, yaw)
                        # Dodan čas za upočasnitev regulatorja (v osnovi laufa loop z 0.03-0.035s)
                        #time.sleep(self.sample)
                    else:
                        time.sleep(0.05)
                    self.controlArgs = None  # Reset arguments
                else:
                    time.sleep(0.05)

    def startControlAllThread(self, T1, T2, yaw):
        # Set the arguments for controlAll
        with self.lock:
            self.controlArgs = (T1, T2, yaw)
        # Signal the persistent thread to run controlAll
        self.controlEvent.set()

    def on_key_press(self,event):
        key = event.keysym 
        print("Key: ",key)
        if key == 'w':
            self.tello.move_forward(30)
        elif key == 's':
            self.tello.move_back(30)        
        elif key == 'a':
            self.tello.move_left(30)
        elif key == 'd':
            self.tello.move_right(30)
        elif key == 'e':
            self.tello.rotate_clockwise(30)
        elif key == 'q':
            self.tello.rotate_counter_clockwise(30)
        elif key == 'r':
            self.tello.move_up(30)
        elif key == 'f':
            self.tello.move_down(30)
        elif key == 'o':
            self.tello.takeoff()
        elif key == 'p':
            self.tello.land()
            self.landZaZih = 1
            self.ukazOld = 2
            self.ukaz = 2
            print("Pristani!")
        elif key == "i": # slikaj
            slika = self.tello.get_frame_read()
            ime = './DJITelloPy/slike/slika' + str(time.time()) + '.jpg'
            cv2.imwrite(ime, slika.frame)
        elif key == "x": # emergency stop
            self.tello.emergency()
            self.tello.end
        

    def videoLoop(self):
        try:
            self.tello.streamoff()
            self.tello.streamon()

            self.frame_read = self.tello.get_frame_read()
            time.sleep(0.5)  # Give some time for the stream to start
            
            # Variables to control the FPS
            fps_limit = 30
            time_per_frame = 1.0 / fps_limit
            last_time = time.time()
            start_time = last_time

            while not self.stopEvent.is_set():   
                current_time = time.time()
                elapsed_time = current_time - last_time             
                if elapsed_time > time_per_frame:
                    self.frame = self.frame_read.frame
                    if self.frame is not None:

                        self.frameCopy = self.frame.copy()
                        self.frameProc = self.frame.copy()
                        
                        self.Rvec, self.Tvec = self.detectAruco(self.arucoId)
                        #print(self.Rvec, self.Tvec)

                        if self.Rvec is not None and self.Tvec is not None:
                            T1, T2, yaw = self.transformArucoToDroneCoordinates(self.Rvec, self.Tvec)
                            if T1 is not None and T2 is not None and yaw is not None:
                                self.startControlAllThread(T1, T2, yaw)
                            else:
                                self.startControlAllThread(None, None, None)
                        else:
                            self.startControlAllThread(None, None, None)
                        
                        #if T1 is not None and T2 is not None:
                            #print("T1", T1[0], T1[1], T1[2])
                            #print("T2", T2[0], T2[1], T2[2])

                        pil_image = Image.fromarray(self.frameCopy) 
                        tk_image = ImageTk.PhotoImage(image=pil_image) 
                        self.image_label.configure(image=tk_image, width=960, height=720)
                        self.image_label.image = tk_image  

                        if self.numIter < 100:
                            self.numIter = self.numIter + 1
                        else:
                            end_time = time.time()
                            total_time = end_time - start_time
                            timePerIt = total_time / 100.0
                            #print("Time per iteration: ",timePerIt)
                            self.numIter = 1
                            start_time = time.time()
                    last_time = current_time
                else:
                    time_to_sleep = time_per_frame - elapsed_time
                    time.sleep(time_to_sleep)                                                    
        except RuntimeError as e:
            print("[INFO] caught a RuntimeError",e)


    def transformArucoToDroneCoordinates(self, rvec, tvec):
        if rvec is not None and tvec is not None:
            # Convert rotation vector to rotation matrix
            R_j, _ = cv2.Rodrigues(rvec)
            # Define the camera to drone rotation matrix
            R_cam_drone = np.array([[0.0000, -0.2079, 0.9781],
                                    [-1.0000, 0.0000, 0.0000],
                                    [0, -0.9781, -0.2079]])
            
            # Transform the point in Aruco coordinate system to drone coordinate system
            T_transformed = np.dot(R_cam_drone, tvec.reshape(3, 1)).flatten()
            R_aruco_to_drone = np.dot(R_cam_drone, R_j)

            # Calculate roll
            roll = np.arctan2(-R_aruco_to_drone[2, 0], np.sqrt(R_aruco_to_drone[2, 1]**2 + R_aruco_to_drone[2, 2]**2))
            roll_degrees = np.degrees(roll)
            #print("Roll:", roll_degrees)

            # Calculate pitch
            pitch = np.arctan2(R_aruco_to_drone[2, 1], R_aruco_to_drone[2, 2])
            pitch_degrees = np.degrees(pitch)
            #print("Pitch:", pitch_degrees)
            
            # Calculate yaw from rotation matrix, assuming R_j is the rotation matrix from Aruco to camera            
            yaw = np.arctan2(R_aruco_to_drone[1, 0], R_aruco_to_drone[0, 0]) + np.pi/2
            yaw_degrees = np.degrees(yaw)

            # Transform the specific points in Aruco coordinate system to drone coordinate system
            # Apply rotation
            specific_point_transformed_1 = np.dot(R_aruco_to_drone, self.specific_point_aruco_1)
            specific_point_transformed_2 = np.dot(R_aruco_to_drone, self.specific_point_aruco_2)
            # Apply translation
            specific_point_transformed_1 += T_transformed.reshape(3, 1)
            specific_point_transformed_2 += T_transformed.reshape(3, 1)

            return specific_point_transformed_1.flatten(), specific_point_transformed_2.flatten(), yaw_degrees

        return None, None


    def detectAruco(self, arucoId):
        self.gray = cv2.cvtColor(self.frameProc, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        parameters =  aruco.DetectorParameters()
        parameters.adaptiveThreshConstant = 7
        parameters.minMarkerPerimeterRate = 0.03
        parameters.maxMarkerPerimeterRate = 4.0
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejectedImgPoints = detector.detectMarkers(self.gray) 
        frame_markers = aruco.drawDetectedMarkers(self.frameCopy, corners, ids)

        if ids is not None:
            for i in range(len(ids)):
                if ids[i] == arucoId:
                    c = corners[i]
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(c, 0.10, self.cameraMatrix, self.distCoeffs)
                    
                    if rvec is not None and tvec is not None:
                        cv2.drawFrameAxes(self.frameCopy, self.cameraMatrix, self.distCoeffs, rvec, tvec, 0.20)  
                        """
                        # Convert to Euler
                        R_mat = np.matrix(cv2.Rodrigues(rvec)[0])
                        roll, pitch, yaw = self.rotationMatrixToEulerAngles(R_mat)
                        rollD = math.degrees(roll)
                        pitchD = math.degrees(pitch) 
                        yawD = math.degrees(yaw)
                           
                        rot = np.array([rollD, pitchD, yawD])
                        print("RPY: ",rot)
                        #return np.array([tvec[0][0], rot])"""
                        return rvec, tvec[0][0]
        return None, None

    def rotationMatrixToEulerAngles(self,R):
        assert (self.isRotationMatrix(R))
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        return np.array([x, y, z])

    def isRotationMatrix(self,R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6
    

    ### Funkcija vodenja ###
    def controlAll(self, T1, T2, yaw):

        # TESTI - BRIŠI #
        #self.freqNow = time.time() - self.freq
        #print("Frekvenca",self.freqNow)
        #self.freq = time.time()
        #self.tello.is_flying = True
        #self.DroneRead = 1 # pohitri regulator na max

        T1_filtered = [0,0,0]
        T2_filtered = [0,0,0]

        #--- RAČUNANJE FPS-ja in izpis (ob vzletu kamera zmrzne in s tem preprečiš vodenje v tem času) ---#
        self.frame_count += 1 # Increment the frame count
        # Calculate and print FPS every 2 seconds
        if time.time() - self.last_fps_calculation >= 2:
            self.cur_fps = self.frame_count / (time.time() - self.last_fps_calculation)
            #print(f"FPS: {self.cur_fps:.2f}")

            # Reset the frame count and last FPS calculation time
            self.frame_count = 0
            self.last_fps_calculation = time.time()    

        #--- Čakanje na značko in update za low pass filter ---#  
        if T1 is not None and T2 is not None and yaw is not None and self.cur_fps > 10 and self.tello.is_flying:

            #--- LOW PASS FILTER ---#
            if self.prev_T1_filtered is None:
                self.prev_T1_filtered = T1
            if self.prev_T2_filtered is None:
                self.prev_T2_filtered = T2
            
            # Apply low pass filter
            T1_filtered = 0.1 * T1 + 0.9 * self.prev_T1_filtered
            T2_filtered = 0.1 * T2 + 0.9 * self.prev_T2_filtered

            # Update the previous filtered values for the next call
            self.prev_T1_filtered = T1_filtered
            self.prev_T2_filtered = T2_filtered
                                
            # Update the last call filtered value of T1 for the next comparison
            self.last_call_T1_filtered = T1_filtered

        
        #--- State machine - Python switch stavek ---#
        if self.landZaZih == 0:

            # Koordinate drona:
            # x+: premik naprej, [cm]
            # y+: premik v levo, [cm]
            # z+: premik navzgor [cm]
            # r+: rotacija v smeri ure (?)


            # Pridobivanje informacij iz drona
            if self.batteryCnt >= 30:       
                baterija = self.tello.get_battery()
                print(f"Bat:",baterija,"%")
                self.batteryCnt = 0
            else:
                self.batteryCnt += 1


            match self.flightState:

                #--- VZLET ---#
                case self.state_default: # 0

                    self.controlEnabled = False
                    currTime = time.time()
                    print(self.takeoffEnabled)
                    if self.takeoffEnabled:
                        self.takeoffEnabled = False 
                        self.tello.takeoff()
                    
                    # Začni z iskanjem značke
                    time.sleep(1)
                    self.flightState = self.state_search

                #--- ISKANJE ARUCO ---#
                case self.state_search: # 1
         
                    visina = self.tello.get_height()  
                    print("Iskanje, visina:", visina)  

                    if self.cur_fps > 10:                    
                        if T1 is not None and T2 is not None and yaw is not None and self.tello.is_flying:
                            
                            if self.searchProtect > 10:
                                self.searchProtect = 0
                                print("NAJDU!!!")
                                self.bubble = 0
                                self.arucoFound = 1 
                                self.visina = self.tello.get_distance_tof()  
                                self.visinaOld = self.visina

                                # TODO: iskanje značke naprej     
                                
                                self.flightState = self.state_aligne_move
                            else:
                                self.searchProtect += 1

                        else:

                            if self.DroneRead == 1: # upočasni izvajanje
                                # Rutina iskanja aruco značke
                                if visina >= 280: # 250cm je lahko obroč še
                                    # yaw zasuk v eno in drugo smer
                                    if self.searchyaw == 0:
                                        self.tello.send_rc_control(0,0,-20,-8)
                                        self.searchyaw = -1
                                    if self.searchyaw == 1:
                                        self.tello.send_rc_control(0,0,-20,8)
                                        self.searchyaw = 2
                                    
                                    self.ukazOld = 2

                                if visina <= 50:
                                    self.ukazOld = 1
                                    if self.searchyaw == -1:
                                        self.tello.send_rc_control(0,0,20,8)
                                        self.searchyaw = 1
                                    if self.searchyaw == 2:
                                        self.tello.send_rc_control(0,0,20,-8)
                                        self.searchyaw = 0

                                # če je prenizko, se dvigni
                                if self.ukaz == 0 and (self.ukazOld == 1 or self.ukazOld == 0):
                                    self.pricakovanaVisina = visina + self.deltaVisina
                                    self.tello.send_rc_control(0,0,30,0)
                                    self.ukaz = 1

                                # če je previsoko, se spusti
                                if self.ukaz == 0 and (self.ukazOld == 2 or self.ukazOld == 0):
                                    self.pricakovanaVisina = visina - self.deltaVisina
                                    self.tello.send_rc_control(0,0,-20,0)
                                    self.ukaz = 2

                                # Blokada premikanja
                                if self.ukaz == 1:
                                    if visina >= self.pricakovanaVisina - 10:
                                        self.ukazOld = 1
                                        self.ukaz = 0

                                if self.ukaz == 2:
                                    if visina <= self.pricakovanaVisina + 10:
                                        self.ukazOld = 2
                                        self.ukaz = 0
                    
                
                #--- PORAVNAVA/PREMIK DRONA ---#
                case self.state_aligne_move: # 2

                    print("Bubble: ",self.bubble)

                    # INFO: Pazi katere razdalje uporabljaš. Ene so iz T1 druge pa iz self.razdalja (ta je prirejena glede na state)

                    # V primeru da se značko izgubi iz vidnega polja
                    if T1 is None and T2 is None and yaw is None and self.tello.is_flying and self.bubble == 0:
                        print("Zgubu!") 
                        self.bubble = 0

                        # Dodatna možnost da najde značko če overshoota
                        if self.arucoFound == 1 :
                            self.tello.send_rc_control(0,-20,0,0)
                        
                        # Počakaj da se odmakne čene pojdi v iskanje
                        if self.arucoFound == 0 and self.searchProtect > 4:
                            self.flightState = self.state_search
                            self.searchProtect = 0
                        if self.DroneRead == 1: # "timer" da se odmakne
                            self.searchProtect += 1

                        self.arucoFound = 0
                        
                    else:

                        self.arucoFound = 1 # Če jo najde z odmikom
                        self.visina = self.tello.get_distance_tof() # V primeru da pride do napake lahko vseeno zazna krog

                        # Vodenje s pid regulacijo
                        for i in range(4):

                            self.hitrost[i] = 0
                            self.razdalja[i] = 0

                            if i != 3: # Translacija
                                self.hitrost[i], self.razdalja[i], self.radij = self.CalculatePID(i,T1_filtered[i])
                            if i == 3 and yaw is not None: # Rotacija
                                #break
                                self.hitrost[i], self.razdalja[i], self.radij = self.CalculatePID(i,yaw)

                        print("Razdalja:", np.round(self.razdalja,2))
                        print("Radij:", np.round(self.radij,2))
                        print("Hitrost:", self.hitrost)

                        if self.bubble == 0:
                            self.tello.send_rc_control(self.hitrost[1], self.hitrost[0], self.hitrost[2], self.hitrost[3]) # L-R, F-B, U-D, Y

                        if self.bubble == 1 and self.razdalja[0] < 150 and self.radij > 20:
                            self.tello.send_rc_control(self.hitrost[1], 0, self.hitrost[2], self.hitrost[3]) # L-R, F-B, U-D, Y

                        # Preverjam napake - krogi
                        # Vstop v zunanju bubble -> Slow!
                        #if self.razdalja[1] <= 40 and self.razdalja[1] >= -40 and self.razdalja[2] <= 40 and self.razdalja[2] >= -40 and self.razdalja[3] <= 8 and self.razdalja[3] >= -8 and (self.razdalja[0] <= 130 or self.bubble == 1) and self.arucoId != 0: # yaw 3->8
                        if self.radij < 55 and self.razdalja[3] <= 8 and self.razdalja[3] >= -8 and (self.razdalja[0] <= 150 or self.bubble == 1) and self.arucoId != 0: # yaw 3->8
                            self.bubble = 1

                            #self.tello.send_rc_control(0,-2,0,0) # Brake
                            if T1_filtered[0] == 0 or T1_filtered[1] == 0 or T1_filtered[2] == 0:
                                self.bubble = 0

                            # Vstop v notranji bubble -> GO!
                            if self.radij <= 20 and self.razdalja[0] <= 100:
                                self.flightState = self.state_go
                                self.visina = self.tello.get_height() 
                                self.visinaOld = self.visina
                                self.ravnoCnt = 0
                            else:
                                self.ravnoCnt += 1

                            # Ko zazna veliko spremembo v višini ve da je skozi obroč
                            if self.visinaOld - self.visina > 40:
                                print("Obroč!!!")
                                self.tello.send_rc_control(0,20,0,0)
                                self.arucoDone = 1
                                self.bubble = 0
                                # v vsakem primeru gre v flip al ga rabi ali ne
                                self.flightState = self.state_flip
                            else:
                                self.visinaOld = self.visina # upade višine
                                print("Visina:", self.visina)

                        # Preverjam napake - pristanek
                        if self.razdalja[1] <= 15 and self.razdalja[1] >= -15 and self.razdalja[2] <= 20 and self.razdalja[2] >= -20 and self.razdalja[0] <= 30 and self.razdalja[3] <= 3 and self.razdalja[3] >= -3 and self.arucoId == 0: 
                            self.bubble = 1

                            # zaščita da vidiš da je res ravno poravnan
                            if self.ravnoCnt >= 3:
                                self.flightState = self.state_landign
                                self.ravnoCnt = 0
                            else:
                                self.ravnoCnt += 1
                        
                        if self.radij > 50 and self.razdalja[0] > 110 and self.bubble == 1:
                            self.bubble = 0
                            
                
                #--- LETI SKOZI OBROČ ---#
                case self.state_go: # 3

                    self.visina = self.tello.get_distance_tof()
                    print("GO!, Visina:", self.visina, "VisinaOld:", self.visinaOld)

                    # Ko zazna veliko spremembo v višini ve da je skozi obroč
                    if self.visinaOld - self.visina > 40: #and self.korakiSkozi < 4
                        print("Obroč!!!")
                        self.tello.send_rc_control(0,20,0,0)
                        self.arucoDone = 1
                        self.bubble = 0
                        # v vsakem primeru gre v flip al ga rabi ali ne
                        self.flightState = self.state_flip
                    else:
                        if self.korakiSkozi < 4 and self.DroneRead == 1:
                            self.tello.send_rc_control(0, 15, 0, 0) # L-R, F-B, U-D, Y
                            self.korakiSkozi += 1
                        self.visinaOld = self.visina # upade višine
                        
                
                #--- FLIP ---#
                case self.state_flip: # 4
                    self.bubble = 0

                    # FLIP - po preletenih prvih treh obročih
                    if self.arucoId == 3 and self.arucoDone == 1: 
                        print("Flip")
                        #self.tello.flip_left()
                        self.arucoDone = 0
                        self.flipDone = 1
                        self.arucoId += 1
                        print("Iščem aruco: ", self.arucoId)
                        print("Rotiram")
                    
                    # ROTACIJA - "timer" preden se obrne za 180 deg po flipu
                    if self.droneRotate >= 3:
                        self.flipDone = 0
                        self.tello.send_rc_control(0, 0, 0, 60) # za nastavit še 
                        self.flightState = self.state_search
                    
                    if self.DroneRead == 1 and self.arucoId == 4 and self.flipDone == 1:
                        self.droneRotate += 1

                    # FLIP - po preletenih vseh obročih
                    if self.arucoId == 5 and self.arucoDone == 1: 
                        print("Flip")
                        self.tello.flip_left()
                        self.flipDone = 1
                        self.arucoDone = 0
                        self.arucoId = 0
                        print("Iščem aruco: ", self.arucoId)
                        self.flightState = self.state_search

                    # Skoči nazaj v iskanje
                    if self.arucoId != 3 and self.arucoId != 5 and self.arucoDone == 1:
                        self.arucoDone = 0
                        self.arucoId += 1
                        print("Iščem aruco: ", self.arucoId)
                        self.flightState = self.state_search

                #--- PRISTANI ---#
                case self.state_landign: # 5
                    print("Land")
                    self.tello.land()
                    self.tello.streamoff()
                    self.flightState = self.state_off
                
                #--- IZKLOP ---#
                case self.state_off: # 6
                    print("OFF")
                    z_accel = self.tello.get_acceleration_z
                    if z_accel < 0.02: # zaznan pristanek
                        self.onClose(self)

                #--- NEDEFINIRANO - default stanje ---#
                case default:
                    self.flightState = self.state_default

            # Reset pulling timer
            self.DroneRead = 0

        # // END - DroneRead (state machine) // #

        self.controlEnabled = True


    def CalculatePID(self, os, trenutnaVrednost): # os: 0/1/2/3 - katero os gledaš, trenutnaVrednost: trenutna x,y,z,yaw vrednost
        
        speed = 0 # izhodna vrednost hitrosti
        radij = 0 # izhodna vrednost radija
        
        # Pretvorba m -> cm (brez yaw)
        if os != 3: trenutnaVrednost = trenutnaVrednost * 100

        # Shranjevanje stare napake
        self.napaka[os][2] = self.napaka[os][1]
        self.napaka[os][1] = self.napaka[os][0]

        # Histereza zaradi neustalitve
        if os == 0: # naprej / nazaj
            if self.flightState == self.state_landign: self.napaka[os][0] = (trenutnaVrednost + 100) # za pristanek
            if self.flightState != self.state_landign: self.napaka[os][0] = (trenutnaVrednost + 100) # za kroge v bubblu # and self.bubble == 0
            #if self.flightState != self.state_landign and self.bubble == 0: self.napaka[os][0] = (trenutnaVrednost + 000) # za kroge preden pride do bubbla
        if os == 1: # levo / desno
            if trenutnaVrednost > 3 or trenutnaVrednost < 3: self.napaka[os][0] = 0 - trenutnaVrednost
        if os == 2: # gor / dol
            if (trenutnaVrednost > 3 or trenutnaVrednost) < 3 and self.flightState == self.state_aligne_move: self.napaka[os][0] = trenutnaVrednost + 30 # za kroge
            if (trenutnaVrednost > 3 or trenutnaVrednost) < 3 and self.flightState == self.state_landign: self.napaka[os][0] = trenutnaVrednost # Za pristanek
        if os == 3: # yaw ! ne dat else: ker ne dela nič več (pojma nimam zakaj ne)
            if trenutnaVrednost > 3 or trenutnaVrednost < 3: self.napaka[os][0] = 0 - trenutnaVrednost

        # PID formula
        if os == 0: self.izhod[os] = self.izhod[os] + self.A0x * self.napaka[os][0] + self.A1x * self.napaka[os][1] + self.A2x * self.napaka[os][2] # X - naprej/nazaj
        if os == 1: self.izhod[os] = self.izhod[os] + self.A0y * self.napaka[os][0] + self.A1y * self.napaka[os][1] + self.A2y * self.napaka[os][2] # Y - levo/desno
        if os == 2: self.izhod[os] = self.izhod[os] + self.A0z * self.napaka[os][0] + self.A1z * self.napaka[os][1] + self.A2z * self.napaka[os][2] # Z - gor/dol
        if os == 3: self.izhod[os] = self.izhod[os] + self.A0r * self.napaka[os][0] + self.A1r * self.napaka[os][1] + self.A2r * self.napaka[os][2] # YAW - rotacija

        # Limit output (rabljen speed da se ne križa s self.hitrost)
        if self.izhod[os] > 30: # 100
            speed = 30 # 100
        else:
            speed = int(self.izhod[os])

        if self.bubble == 1 and os == 0 and self.radij > 40:
            speed = 0

        if self.bubble == 1 and os == 0 and self.radij < 30:
            speed = 3


        # Računaje radija
        if os == 1:
            self.radijY = self.napaka[os][0] 
        if os == 2:
            self.radijZ = self.napaka[os][0] 
        if os == 3:
            radij = np.sqrt(self.radijY**2 + self.radijZ**2) # pitagor
    
        return speed, self.napaka[os][0], radij
    
    
    def onClose(self):
        self.stopEvent.set()
        del self.tello
        self.root.quit()

