1) Check if this directory exists so to save recorded videos:
	/home/pi/Videos/

2) GPIO pins connection:
  Left_Forward = 15	Left_Backward = 13	Left_Enable = 16
  Right_Forward = 11	Right_Backward = 7	Right_Enable = 18

3) Keyboard Instructions:
	4x Arrows for respective movement
	q ----- Quit controlling mode
	s ----- Shut down system
	r ----- Start recording
	t ----- Stop Recording

4) If bot doesn't go as expected: 2 cases
	i) Right <--> Up and Down <--> Left
	   then interchange (7 and 11)
	ii) Right <--> Down and Up <--> Left
	   then interchange (13 and 15)