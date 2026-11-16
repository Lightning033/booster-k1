navigation_instruction = "Move straight to leave the room, then turn left and stand near the human"

no_of_look_behind = 0

no_of_check_points = 0

SAFETY_STOP = 20

# get the robot standing and put it in walk mode

while True:

    # if no_of_check_points > SAFETY_STOP:
        # put the robot in prep mode then break out of the loop

    # get 3 views
        # capture view (front view)
        # turn head left 
        # capture view (left view)
        # turn head right
        # capture view (right view)
        # turn head center
    
    # Images pre-processing
        # overlay grid map on top of each image.

    # feed 3 views to LLM 


    # if LLM output says "action": "stop"
        # put the robot in prep mode then break out of the loop

    # if LLM output says "action": "look_behind"
        # turn body 180 degrees
        # no_of_look_behind ++
        # if no_of_look_behind > 2:
            # unexpected behaviour of LLM. 
            # put robot in prep mode and break out of the loop.
    
    # if LLM output says "action": "move"
        # no_of_look_behind = 0
        # map LLM output to actual target position with respect to Robot
        # send instruction to robot to walk to target position with respect 

    no_of_check_points += 1


