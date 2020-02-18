# AI Real Time Road Space Rationing control using Jetson Nano
# CRISTIAN LAZO QUISPE
# clazoq@uni.pe
# Lima,Peru

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import pycuda.autoinit  # This is needed for initializing CUDA driver
from utils import *

from openalpr import Alpr
import argparse
import tensorflow as tf
import RPi.GPIO as GPIO
import time                        
from RPLCD import CharLCD
import numpy as np
import cv2
import time
from PIL import Image, ImageDraw
import operator
from sort import Sort
import threading
import random
from time import gmtime, strftime
import copy
import datetime
import logging
import sys
import collections
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import gc
import tensorflow as tf
gc.collect()




lite =  False

times = {}
averageInferenceDurationPerCar = []
durationPerCar = []
totalDuration = []
try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range


class AppException(Exception):
	def __init__(self, message, source):
		super().__init__("[source={0}][message=]".format(source, message))


class QuitException(Exception):
	def __init__(self):
		super().__init__("Quit requested")


class Logger:
	def __init__(self, tag, outType=sys.stdout):
		self.tag = tag

		formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
		handler = logging.FileHandler('logs/log_' + strftime("%Y%m%d", gmtime()) + '.txt', mode='a')
		handler.setFormatter(formatter)
		screen_handler = logging.StreamHandler(stream=outType)
		screen_handler.setFormatter(formatter)
		logger = logging.getLogger(tag)
		logger.setLevel(logging.DEBUG)
		logger.addHandler(handler)
		logger.addHandler(screen_handler)
		
		self.logger = logger

	def error(self, message):
		self.logger.error("[Owner={0}] {1}".format("{:<12}".format(self.tag), message))

	def info(self, message):
		self.logger.info("[Owner={0}] {1}".format("{:<12}".format(self.tag), message))

	def exception(self, message):
		self.logger.exception("[Owner={0}] {1}".format("{:<12}".format(self.tag), message))


class TimeRecord:
	def __init__(self):
		self.time_ini = time.time()
		self.time_end = None

	def step(self):
		if not self.time_end:
			self.time_end = time.time()
			dif = self.duration()
			durationPerCar.append(dif)

	def duration(self):
		if self.time_end:
			return self.time_end - self.time_ini	


class NameCount:
	def __init__(self, name):
		self.name = name
		self.count = 0


class Recognizer(threading.Thread):
	logger = Logger("Recognizer")

	def __init__(self, cvp):
		threading.Thread.__init__(self)

		Recognizer.logger.info("Loading")

		self.event = threading.Event()
		self.working = False
		self.working_lock = threading.Lock()
		self.update_lock = threading.Lock()
		self.fids = []
		self.frame_rgb = None
		self.bboxes_cars = None
		self.cvp = cvp
		Recognizer.logger.info("Loaded")

	def stop(self):
		self.event.set()

	def can_work(self):
		can = False

		self.update_lock.acquire()
		can = self.bboxes_cars is not None
		self.update_lock.release()

		return can

	def is_working(self):
		self.working_lock.acquire()
		working = self.working
		self.working_lock.release()

		return working

	def set_working(self, is_working):
		self.working_lock.acquire()
		self.working = is_working
		self.working_lock.release()

	#def enqueue(self,frame_rgb, bboxes_cars, fids,names_ids):
	def enqueue(self,frame_rgb, bboxes_cars, fids):
		if not self.is_working():
			self.update_lock.acquire()
			self.frame_rgb = frame_rgb
			self.bboxes_cars = bboxes_cars
			self.fids = fids
			#self.names_ids = names_ids
			self.update_lock.release()

	def run(self):
		Recognizer.logger.info("Starting")
		while not self.event.is_set():
			if self.can_work():
				self.recognize()

			self.event.wait(0.1)
		Recognizer.logger.info("Finished")

	def recognize(self):
		self.set_working(True)

		self.update_lock.acquire()
		frame_rgb = self.frame_rgb.copy()
		fids = self.fids.copy()
		bboxes_cars= self.bboxes_cars.copy()
		#names_ids = self.names_ids.copy()
		#self.names_ids = None
		self.frame_rgb = None
		self.fids = None
		self.bboxes_cars = None
		self.update_lock.release()

		##############################################
		
		bboxes_cars2 = []
		fids2 = []
		#names_ids2  = []
        
		for idaux,[x1,y1,x2,y2] in enumerate(bboxes_cars):
			bandera_blurry = False
			bandera_short =  False
			if ((y2-y1)>CVProcessor.MIN_SIZE_ALLOWED and (x2-x1)>CVProcessor.MIN_SIZE_ALLOWED):
				#plt.imsave("image.png",frame_rgb[y1:y2,x1:x2,:])
				(bandera_blurry,blur_factor) = self.cvp.isBlurryImage(frame_rgb[y1:y2,x1:x2,:])
				#Recognizer.logger.info("[Track:"+str(fids[idaux])+"] Blurry image : "+str(blur_factor))
				if (bandera_blurry):
					Recognizer.logger.info("[Track:"+str(fids[idaux])+"] Blurry image : "+str(blur_factor))

			else:
				Recognizer.logger.info("[Track:"+str(fids[idaux])+"] Short image ")
				bandera_short =  True
			if(bandera_blurry or bandera_short):
				name = CVProcessor.UNKNOWN_NAME
				color = (37, 40, 8)
				name_choosen = self.cvp.set_name(fids[idaux], name, color,True)
			else:
				fids2.append(fids[idaux])
				#names_ids2.append(names_ids[idaux])
				bboxes_cars2.append([x1,y1,x2,y2])

		bboxes_cars = bboxes_cars2.copy()
		#names_ids = names_ids2.copy()
		fids = fids2.copy()
		today = datetime.datetime.today()
		dia = today.weekday() 

		if(len(fids)>0):
			name = "Procesando"
			color =  (246, 213, 105)
			(maxi_y,maxi_x,channels)=frame_rgb.shape
			for idx,aea in enumerate(fids):
				[x1,y1,x2,y2] = bboxes_cars[idx]
				Recognizer.logger.info("[Track:"+str(idx)+"] Started recognizing ")
				img_gamma = adjust_gamma(frame_rgb[y1:y2,x1:x2,:], gamma=1.5)	
				bandera,name_license,confidence = get_name_license(self.cvp.alpr,img_gamma,tresh = 60)
				if(bandera and len(name_license)==6):
					Recognizer.logger.info("[Track:"+str(idx)+"] Finished recognizing -> ["+str(name_license)+']_confidence_['+str(round(confidence,2))+']')
					name_choosen = self.cvp.set_name(fids[idx],name_license, color,False)
					if(name_choosen[-1].isdigit()):
						if(int(name_choosen[-1])%2==0):
							if(dia in [0,2,4,5,6]):
								path = CVProcessor.PATH_RESULTS+'/Reconocido/par/legal/'+str(datetime.datetime.now())[:-4]+'_['+str(name_license)+']_confidence_['+str(round(confidence,2))+']_track_'+str(idx)+'.jpg'
								cv2.imwrite(path,cv2.cvtColor(frame_rgb[y1:y2,x1:x2,:],cv2.COLOR_RGB2BGR))
							else:
								path = CVProcessor.PATH_RESULTS+'/Reconocido/par/ilegal/'+str(datetime.datetime.now())[:-4]+'_['+str(name_license)+']_confidence_['+str(round(confidence,2))+']_track_'+str(idx)+'.jpg'
								cv2.imwrite(path,cv2.cvtColor(frame_rgb[y1:y2,x1:x2,:],cv2.COLOR_RGB2BGR))
								CVProcessor.lcd.clear()
								Recognizer.logger.info("[Track:"+str(idx)+"] PROHIBITED->"+str(name_license))
								CVProcessor.lcd.write_string(u''+'ILEGAL : '+str(name_license))
								GPIO.output(CVProcessor.output_pin, GPIO.HIGH)	if name_choosen != "Desconocido" else GPIO.output(CVProcessor.output_pin, GPIO.LOW)
								time.sleep(.500)						
								GPIO.output(CVProcessor.output_pin, GPIO.LOW)

						else:
							if(dia in [1,3,4,5,6]):
								path = CVProcessor.PATH_RESULTS+'/Reconocido/impar/legal/'+str(datetime.datetime.now())[:-4]+'_['+str(name_license)+']_confidence_['+str(round(confidence,2))+']_track_'+str(idx)+'.jpg'
								cv2.imwrite(path,cv2.cvtColor(frame_rgb[y1:y2,x1:x2,:],cv2.COLOR_RGB2BGR))
							else:
								path = CVProcessor.PATH_RESULTS+'/Reconocido/impar/ilegal/'+str(datetime.datetime.now())[:-4]+'_['+str(name_license)+']_confidence_['+str(round(confidence,2))+']_track_'+str(idx)+'.jpg'
								cv2.imwrite(path,cv2.cvtColor(frame_rgb[y1:y2,x1:x2,:],cv2.COLOR_RGB2BGR))
								CVProcessor.lcd.clear()
								Recognizer.logger.info("[Track:"+str(idx)+"] PROHIBITED->"+str(name_license))
								CVProcessor.lcd.write_string(u''+'ILEGAL : '+str(name_license))
								GPIO.output(CVProcessor.output_pin, GPIO.HIGH)	if name_choosen != "Desconocido" else GPIO.output(CVProcessor.output_pin, GPIO.LOW)
								time.sleep(.500)						
								GPIO.output(CVProcessor.output_pin, GPIO.LOW)


				else:
					Recognizer.logger.info("[Track:"+str(idx)+"] Not recognizing ->"+str(name_license)+'-confidence:'+str(round(confidence,2)))
					cv2.imwrite(CVProcessor.PATH_RESULTS+'/Desconocido/'+str(datetime.datetime.now())[:-4]+'_['+str(name_license)+']_confidence_['+str(round(confidence,2))+']_track_'+str(idx)+'.jpg',cv2.cvtColor(frame_rgb[y1:y2,x1:x2,:],cv2.COLOR_RGB2BGR))

					name_choosen=self.cvp.set_name(fids[idx],name, color,False)
		self.set_working(False)

class OpenCVManager:
	#OUTPUT_SIZE_WIDTH = 900
	#OUTPUT_SIZE_HEIGHT = 840

	MESSAGE_IMAGE_WIDTH = 250

	COLOR_BLACK = [0, 0, 0]
	COLOR_WHITE = [255, 255, 255]

	logger = Logger("OpenCVManager")

	def __init__(self, source):
		OpenCVManager.logger.info("Loading")

		try:

			# USING CAMERA :
			#self.video = cv2.VideoCapture(0)
			#self.video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
			
			# USING PICAMERA :
			#print(gstreamer_pipeline(flip_method=0))
			#self.video = cv2.VideoCapture(gstreamer_pipeline(flip_method=0),cv2.CAP_GSTREAMER)
			
			# USING VIDEO FILE:
			self.video = cv2.VideoCapture('videos/IMG_0168.MOV')
			self.video.set(cv2.CAP_PROP_BUFFERSIZE,3)

			if self.video.isOpened():
				self.window_handler = cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)
			else:
				OpenCVManager.logger.error("VideoCapture not open")
				raise QuitException()

		except Exception as e:
			raise AppException(e, "OpenCVManager")

		OpenCVManager.logger.info("Loaded")

	def read(self):
		ret, image = self.video.read()
		if ret == True:
			return image

		raise Exception('Camera is not open')

	def get_last_processed_frame(self):
		return self.last_processed_frame

	def stop(self):
		OpenCVManager.logger.info("Stopping")

		self.video.release()
		cv2.destroyAllWindows()

		OpenCVManager.logger.info("Stopped")

	def show(self, title, image):
		cv2.imshow(title, np.array(image, dtype = np.uint8))

	def waitKey(self):
		return cv2.waitKey(1) & 0xFF


class Tracker:
	def __init__(self):
		self.tracker = Sort()

	def predict(self, locations):
		det = []
		for [x1, y1, x2, y2] in locations:
			det.append((x1, y1, x2, y2, 1))

		return self.tracker.update(np.array(det))


class CVProcessor:
	F_SCALE = 1
	output_pin = 4 # PITIDO O LED DE SIGNAL
	output_pin_puerta = 27 # PITIDO O LED DE SIGNAL
	THRESHOLD_RECOGNITION = 0.425
	THRESHOLD_DETECTION = 0.3
	INTERPOLATION_TYPE = cv2.INTER_CUBIC 
	WIDTH = 1080
	HEIGHT = 1080
	MIN_SIZE_ALLOWED = 120
	REBOOT_TIME_GAP = 5
	REBOOT_TIME_PUERTA = 5*60
	OFFSET_BBOX_WIDTH = 1#.15#1.25
	OFFSET_BBOX_HEIGHT_UP = 1#.15#1.5
	OFFSET_BBOX_HEIGHT_DOWN = 1#.15#1.3
	SAME_NAME_REQUIRED_COUNT = 1
	SAME_NAME_REQUIRED_COUNT_UNKNOWN = 12
	tiny = True
	iou_threshold = 0.2
	confidence_threshold = 0.3
	VARIANCE_OF_LAPLACIAN_THRESHOLD = 100 # HD IMAGES HAS MORE THAN 100 OF BLUR VALUE
	UNKNOWN_NAME = "Procesando"
	PATH_RESULTS = 'results'
	GPIO.setmode(GPIO.BCM)
	lista_clases = ['car','motorcycle','bus','train','truck']
	lcd = CharLCD(cols=16 , rows=2, pin_rs= 11, pin_e=5,pins_data=[6,13,19,26], numbering_mode=GPIO.BCM)
	logger = Logger("CVProcessor")
	def __init__(self):
		CVProcessor.logger.info("Loading")
		if not os.path.exists(CVProcessor.PATH_RESULTS):
			os.mkdir(CVProcessor.PATH_RESULTS)
		if not os.path.exists(CVProcessor.PATH_RESULTS+'/Desconocido'):
			os.mkdir(CVProcessor.PATH_RESULTS+'/Desconocido')
		if not os.path.exists(CVProcessor.PATH_RESULTS+'/Reconocido'):
			os.mkdir(CVProcessor.PATH_RESULTS+'/Reconocido')
		if not os.path.exists(CVProcessor.PATH_RESULTS+'/Reconocido/par'):
			os.mkdir(CVProcessor.PATH_RESULTS+'/Reconocido/par')
		if not os.path.exists(CVProcessor.PATH_RESULTS+'/Reconocido/impar'):
			os.mkdir(CVProcessor.PATH_RESULTS+'/Reconocido/impar')
		if not os.path.exists(CVProcessor.PATH_RESULTS+'/Reconocido/par/legal'):
			os.mkdir(CVProcessor.PATH_RESULTS+'/Reconocido/par/legal')
		if not os.path.exists(CVProcessor.PATH_RESULTS+'/Reconocido/impar/legal'):
			os.mkdir(CVProcessor.PATH_RESULTS+'/Reconocido/impar/legal')
		if not os.path.exists(CVProcessor.PATH_RESULTS+'/Reconocido/par/ilegal'):
			os.mkdir(CVProcessor.PATH_RESULTS+'/Reconocido/par/ilegal')
		if not os.path.exists(CVProcessor.PATH_RESULTS+'/Reconocido/impar/ilegal'):
			os.mkdir(CVProcessor.PATH_RESULTS+'/Reconocido/impar/ilegal')
		GPIO.setup(self.output_pin, GPIO.OUT, initial=GPIO.LOW)
		GPIO.setup(self.output_pin_puerta, GPIO.OUT, initial=GPIO.LOW)
		CVProcessor.logger.info('Starting GPIO JETSON NANO')
		self.cls_dict = get_cls_dict('coco')
		self.trt_ssd = TrtSSD('ssd_mobilenet_v2_coco',(300, 300))
		self.alpr = Alpr("us", "/etc/openalpr/openalpr.conf", "runtime_data/")
		if not self.alpr.is_loaded():
			CVProcessor.logger.info("Error loading OpenALPR")
			sys.exit(1)

		self.alpr.set_top_n(20)
		self.alpr.set_default_region("dm")#"be"

		self.class_names = []
		for i in range(len(self.cls_dict)):
			self.class_names.append(self.cls_dict[i])
		all_names = ""
		for i in self.class_names:
			all_names+=" "+i
		CVProcessor.logger.info("Classes names")    
		CVProcessor.logger.info(all_names)
		self.last_reboot = 0      
		self.car_names = {}
		self.tracker = Tracker()
		self.count = 0
		self.colors = {}
		self.times = {}
		self.car_detected_count = {}
		self.car_detected_names = {}

		self.working_lock = threading.Lock()

		self.font_scale = 1.5
		self.font = cv2.FONT_HERSHEY_PLAIN
		self.text = "Hermano. = 12"
		(self.text_width, self.text_height) = cv2.getTextSize(self.text, self.font, fontScale=self.font_scale, thickness=1)[0]

		self.recognizer = Recognizer(self)

		CVProcessor.logger.info("Loaded")
    
	def get_locations_car(self,frame,class_names):
		boxes, confs, clss = self.trt_ssd.detect(frame,0.3)
		cordenadas_xy = []
		names_xy = []
		scores= []
		for bb, cf, cl in zip(boxes, confs, clss):
			cl = int(cl)
			cls_name = class_names[cl]
			color = (0, 0, 255)
			if(cls_name in self.lista_clases):
				xy = [bb[0], bb[1], bb[2], bb[3]]
				cordenadas_xy.append(xy)
				names_xy.append(cls_name)
				scores.append(cf)
		cordenadas_xy = np.array(cordenadas_xy)
		cordenadas_xy = non_max_suppression_fast(cordenadas_xy,0.3)

		return cordenadas_xy,names_xy
    
	def process(self, image):
		start = time.time()
		frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		car_locations,names_xdxd = self.get_locations_car(image,self.class_names)
		predictions = self.tracker.predict(car_locations)
		self.process_predictions(image, frame_rgb,predictions)
		if time.time() - self.last_reboot >= CVProcessor.REBOOT_TIME_GAP:
			self.reboot_algorithm()
		#if time.time() - self.last_reboot_download_json >= CVProcessor.REBOOT_TIME_PUERTA:
		#	self.download_datajson()
		self.count+=1
		# Display the resulting image
		fps = time.time() - start
		today = datetime.datetime.today()
		dia = today.weekday() 
		if dia in [0,2]:
			PERMITIDO = 'EVEN' 
		if dia in [1,3]:
			PERMITIDO = 'ODD' 
		if dia in [4,5,6]:
			PERMITIDO = 'ALL' 
		fps = 'FPS :'+ str(round(1/fps,1))+' ALLOW:'+PERMITIDO
		text_offset_x = 50
		text_offset_y = 50
		box_coords = ((text_offset_x, text_offset_y), (text_offset_x + int(self.text_width*1.25) - 2, text_offset_y - int(self.text_height*1.25) - 2))
		cv2.rectangle(image, box_coords[0], box_coords[1], (255,255,255), cv2.FILLED)        
		cv2.putText(image, fps, (50,50), self.font, 1.25, (150, 0, 150), 2)
		#image = cv2.resize(image, (1240, 1000), interpolation = CVProcessor.INTERPOLATION_TYPE)
		return image
	def process_predictions(self, frame, frame_rgb, predictions):
		id_not_found = []
		locations = []
		#names_ids = []
		id_unknowns = []
		for id_xd,pre in enumerate(predictions):
			arreglo = np.array(pre, dtype=int)
			arreglo[arreglo<0]=0
			[x1, y1, x2, y2,id] = arreglo 
			if id not in self.car_names.keys() or (id in self.car_names and self.car_names[id] == CVProcessor.UNKNOWN_NAME):
				id_unknowns.append(id)
				y1aux = int((y2+y1)/2-(y2-y1)*CVProcessor.OFFSET_BBOX_HEIGHT_UP/2)
				y2aux = int((y2+y1)/2+(y2-y1)*CVProcessor.OFFSET_BBOX_HEIGHT_DOWN/2)
				x1aux = int((x2+x1)/2-(x2-x1)*CVProcessor.OFFSET_BBOX_WIDTH/2)
				x2aux = int((x2+x1)/2+(x2-x1)*CVProcessor.OFFSET_BBOX_WIDTH/2)
				arreglo = np.array([x1aux, y1aux,x2aux, y2aux], dtype=int)
				arreglo[arreglo<0]=0
				locations.append(arreglo)
				color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
				self.colors[id]= color

			text_offset_x = int(x1)
			text_offset_y = int(y1+4)
			box_coords = ((text_offset_x, text_offset_y), (text_offset_x + int(self.text_width) - 2, text_offset_y - int(self.text_height) - 2))

			if id in self.car_names.keys():
				try:
					cv2.rectangle(frame, box_coords[0], box_coords[1], self.colors[id], cv2.FILLED)
					espesor = 6 if self.car_names[id] !='Procesando' else 3
					cv2.rectangle(frame,(x1,y1) ,(x2,y2),self.colors[id], espesor)
				except:
					cv2.rectangle(frame, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
					cv2.rectangle(frame,(x1,y1) ,(x2,y2),(0,0,0), 3)

				cv2.putText(frame,'['+self.car_names[id]+' ] : '+str(id), (x1,y1), self.font,1, (0, 0,0),1)
			else:
				try:
					cv2.rectangle(frame, box_coords[0], box_coords[1], self.colors[id], cv2.FILLED)
					cv2.rectangle(frame,(x1,y1) ,(x2,y2),self.colors[id], 3)
				except:
					cv2.rectangle(frame, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
					cv2.rectangle(frame,(x1,y1) ,(x2,y2),(0,0,0), 3)
				cv2.putText(frame,'Detecting '+str(id), (x1,y1), self.font,1, (0, 0,0),1)
		if not self.recognizer.is_working() and locations:
			self.recognizer.enqueue(frame_rgb, locations, id_unknowns)

	def set_name(self, fid, name, color,bandera):
		self.working_lock.acquire()
		self.colors[fid] = color
		if fid in self.car_detected_names and not bandera:
			if name in self.car_detected_names[fid]:
				self.car_detected_names[fid][name] += 1
			else:
				self.car_detected_names[fid][name] = 1
		else:
			
			self.car_detected_names[fid] = {}
			self.car_detected_names[fid][name] = 1
			self.car_names[fid] = CVProcessor.UNKNOWN_NAME
		
		if fid in self.car_names.keys() and self.car_detected_names[fid] and not bandera:
			counter = collections.Counter(self.car_detected_names[fid])
			most_commons = counter.most_common(2)
			if(len(most_commons)>1):
				most_common1 = most_commons[0]
				most_common2 = most_commons[1]
				if (most_common1[0] == CVProcessor.UNKNOWN_NAME ):
					if(most_common1[1] < CVProcessor.SAME_NAME_REQUIRED_COUNT_UNKNOWN):
						most_common = most_common2
					else:
						most_common = most_common1
				else:
					most_common = most_common1
			else:
				most_common = most_commons[0]

			CVProcessor.logger.info("[Track:"+str(fid)+"] common name: '"+most_common[0]+"' "+str(most_common[1])+" times")

			if self.car_names[fid] == CVProcessor.UNKNOWN_NAME and most_common[1] >= CVProcessor.SAME_NAME_REQUIRED_COUNT:
				if (most_common[0]=="Procesando" and most_common[1] >= CVProcessor.SAME_NAME_REQUIRED_COUNT_UNKNOWN):
					self.car_names[fid] = "Desconocido"
				else:
					self.car_names[fid] = most_common[0]
					if(most_common[0] != CVProcessor.UNKNOWN_NAME):
						CVProcessor.logger.info("[Track:"+str(fid)+"] recognizated: '"+most_common[0]+"' "+str(most_common[1])+" times")
		else:
			self.car_names[fid] = CVProcessor.UNKNOWN_NAME
		self.working_lock.release()

		return str(self.car_names[fid])
    
    
	def isBlurryImage(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		fm = cv2.Laplacian(gray, cv2.CV_64F).var()
		return (fm < CVProcessor.VARIANCE_OF_LAPLACIAN_THRESHOLD,fm)

	def start(self):
		CVProcessor.logger.info("Starting")

		self.recognizer.start()

		CVProcessor.logger.info("Started")

	def stop(self):
		CVProcessor.logger.info("Stopping")

		self.recognizer.stop()

		CVProcessor.logger.info("Stopped")

	def reboot_algorithm(self):
		CVProcessor.logger.info("Rebooting...")

		self.working_lock.acquire()

		self.car_names = {}
		self.tracker = Tracker()
		self.count = 0
		self.colors = {}
		self.times = {}
		self.car_detected_count = {}
		self.car_detected_names = {}
		self.last_reboot = time.time()
		self.working_lock.release()
		self.lcd = CharLCD(cols=16 , rows=2, pin_rs= 11, pin_e=5,pins_data=[6,13,19,26], numbering_mode=GPIO.BCM)	
		self.lcd.clear()
		self.lcd.write_string(u'Running..')
		GPIO.output(CVProcessor.output_pin, GPIO.LOW)
		GPIO.output(CVProcessor.output_pin_puerta, GPIO.LOW)
	def imageAsByteArray(self, image):
		pil_img = Image.fromarray(image) # convert opencv frame (with type()==numpy) into PIL Image
		stream = io.BytesIO()
		pil_img.save(stream, format='JPEG') # convert PIL Image to Bytes
		return stream.getvalue()


class App:
	logger = Logger("App")

	def __init__(self):
		self.isRunning = False
		self.ocv = OpenCVManager(0)
		self.cvp = CVProcessor()

	def run(self):
		App.logger.info("Started")

		while self.isRunning:
			image = self.ocv.read()
			image = self.cvp.process(image)
			self.ocv.show("Video", image)

			key = self.ocv.waitKey()

			if key == ord('q'):
				raise QuitException()

	def start(self):
		App.logger.info("Starting")

		self.isRunning = True
		self.cvp.start()
		self.run()

	def stop(self):
		App.logger.info("Requesting stop")
		self.lcd = CharLCD(cols=16 , rows=2, pin_rs= 11, pin_e=5,pins_data=[6,13,19,26], numbering_mode=GPIO.BCM)
		self.isRunning = False

		if self.ocv:
			self.ocv.stop()

		if self.cvp:
			self.cvp.stop()

		App.logger.info("Stop requested")



if __name__ == '__main__':
	try:
		app = App()
		app.start()
	except QuitException as qe:
		app.stop()
	
