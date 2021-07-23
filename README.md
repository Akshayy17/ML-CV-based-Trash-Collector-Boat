# ML-CV-based-Trash-Collector-Boat
A prototype of unmanned boat/bot has been designed and tested for full functionality where it collects floating trash from various water bodies via remote control or automation, and uses custom trained dataset and computer vision algorithm YOLO to differentiate garbage (draws a rectangular box around detection).

## Images of bot:

![Picture1](https://user-images.githubusercontent.com/69389392/126799431-9b8a737f-8187-4b98-90a8-47be4ecf170d.jpg)
Side view of the bot

![Picture2](https://user-images.githubusercontent.com/69389392/126798347-1bc8bc24-b477-4a27-9960-5c31895c51f6.jpg)
Front view of the bot

### This bot has two modes of operation:
* Manual mode
* Automation mode

## **Manual mode:**
Under this mode, an opearator will control the bot via his controller module (laptop/TFT screen + simple controller) situated at remote place(far away). A small 5MP camera is mounted on the bot at optimum position, which captures a continuous array of frames(video) which is connected to our microprocessor (Raspberry pi 4B). The live feed from bot's camera is processed in our microprocessor and then sent to operator's screen. Now our opeartor located remotely can drive the boat accordingly by looking at his screen using up-down-left-right arrows. The directional input is sent to our microprocessor and respective instructions are given to the motor drive. He drives to the trash, it gets accumulated inside bin of our bot/boat and this process continues. At one point when boat is filled completely, it can be unloaded manually by the banks/shores. The bot is also equipped with IR sensors to detect any obstructions ahead of it on left and right side.

https://user-images.githubusercontent.com/69389392/126797618-fb104545-5b98-436b-bb28-6f705594ae3e.mp4

The operator is controlling the bot via keyboard keys, marbles being trash and white blocks being obstacles.

https://user-images.githubusercontent.com/69389392/126800419-b773e9bc-6b72-4a14-9a58-fd3f6462d60d.mp4

The upper view of manual mode operation.

https://user-images.githubusercontent.com/69389392/126800594-85169d96-3217-42de-9b24-b0d32b3a16a9.mp4

The camera view that is being sent to operator's screen after processing. The trash is being pointed out by square boxes making it easier for operator to go near them. The red vertical bar that can be seen on screen is the notification of obstruction on respective side of bot.

## **Automation mode:**
This mode is speciality of the project. A low level AI is used to generate appropriate output per frame for dynamic input. The functionaing of boat is similar to manual mode except the necessity of operator is removed. We just have to switch on the functioning of bot and leave it in riverstream. The camera feed will be processed by microprocessor and instead of sending it to any other node(device), it will compute results and take decisions on its own. Nearest trash to the boat is calculated for optimum collection path. Whether or where to turn and by how much is calculated per frame at every instant of it. The respective signals are then sent to motor drive and bot moves. The process continues after collecting the first detection for second detection.



https://user-images.githubusercontent.com/69389392/126807752-d0e638ec-03ba-485b-8b20-bb630e10a629.mp4

The bot locates and moves towards trash on it's own. The laptop screen, as we can see is just for the use of surveillance and can be opted out.

https://user-images.githubusercontent.com/69389392/126808103-bacb9fe7-6c6a-4984-87d1-fa502deb2b78.mp4

Upperview of the operation.

https://user-images.githubusercontent.com/69389392/126808620-850db0c6-d310-4fd5-9150-164f3396fb8f.mp4

Surveillance screen output on the laptop with red flag obstruction alerts.


### Two self-designed algos are used for dynamic response of the bot.
* Nearest object detection.
* Auto direction decision.

## Nearest object detection
This algo calculates the vector distance from the mid bottom of the screen to all the detetected trash and sorts them in increasing order. A threshold can also be set to 
limit the number of detections seen of the screen. As you can see in the below video, the nearest marble is denoted by red and the rest by blue as well as threshold is set to max:3.


https://user-images.githubusercontent.com/69389392/126813731-d90edde5-cd73-4da3-a36d-a5bbe9663914.mp4

## Auto direction decision
This algo finds the bias of the detection for respective partition. Cosθ is calulated, neagtive or positive decides left or right turnings and magnitude decides how much to rotate while θ being angle made by two sides of triangle joining at bottom of the screen. A tolerance is provided for straigth movement to have an unambiguous decision taking.


https://user-images.githubusercontent.com/69389392/126814993-804b7eb7-e6b7-4b61-bd37-03e8a13d0cbf.mp4




#### Similar to this we can train a custom model for actual floating trash componenets such as platic bags, bottles, etc.

![Webp net-resizeimage](https://user-images.githubusercontent.com/69389392/126816541-b16c3562-f58c-4411-b2e8-d9a05ac3451c.png)


