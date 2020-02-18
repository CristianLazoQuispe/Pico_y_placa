# Real Time Road Space Rationing control using Jetson Nano

Peru's govermment implemented


## GETTING STARTED

#### Using ISO 
- Download ISO pico_placa.gz:
        https://drive.google.com/drive/folders/1_XjzvR0key_jjU8aexq3Zpe6h2fgbZnz?usp=sharing

- Flash SD CARD with Etcher:

- Open folder Pico_placa_SSD :
    
        cd home/dlinano/Documents/Pico_placa_SSD

- Run code pico_placa.py

        python3 pico_placa.py

- If you want to use special hardware of image capture:

    - webcam : 

            self.video = cv2.VideoCapture(0)
    - video  : 
        
            self.video = cv2.VideoCapture('PATH_OF_VIDEO')
    - picam  : 
        
            self.video = cv2.VideoCapture(gstreamer_pipeline(flip_method=0),cv2.CAP_GSTREAMER)

#### Step by step

- Download this repository:

        git clone https://github.com/CristianLazoQuispe/Pico_y_placa.git
- Follow the instructions of installation on pdf:

        Pico_y_placa.pdf

- SSD MobileNet
        This project use the model od SSD MobileNet on TensorRT
        We use the model of the repository tensorrt_demos:
        
        git clone https://github.com/jkjung-avt/tensorrt_demos.git
        
        Download and implement the TensorRT model from SSD Mobilenet and copy it to the ssd folder

- OpenALPR
        Download the repository and copy the runtime_data folder to Pico_y_placa:

        git clone https://github.com/openalpr/openalpr.git
        

- Run code pico_placa.py

        python3 pico_placa.py

# You can see our results in the following link

        https://www.youtube.com/watch?v=ugQnUsgQzWY
