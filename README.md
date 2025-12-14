# tennis project for computer vision (CS673)

## Project Scope

to implement:

1. tracking court lines DONE
2. detecting and tracking the tennis ball in play DONE
3. tracking the players DONE
4. displaying gameplay stats on the side, including but not limited to:

    - the court lines and posiiton of player at all times MINIMAP
    - position of ball in play DONE
    - where the ball lands NOT DONE
    - a heatmap to display which part of the court the ball lands on the most NOT DONE


to try: pose detection of the player, speed of the ball, detecting type of strokes played (fronthand, backhand, etc.)


## Example:

[enter gif here]


## Requirements:


## Conclusion: 


## to initialise the project:

conda create -n tennis-env python=3.10 -y
conda activate tennis-env

pip install -r requirements.txt    

### to run inference:

python court/final_inference.py --video_path dataset/videos/8.mp4 --tracknet_model weights/tracknet.pt --yolo_model weights/best.pt --output_path output8.mp4 --ball_smoothing 0.5 --trace_length 7