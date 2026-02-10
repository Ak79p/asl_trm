# ASL-TRM application

[ User uploads .mp4 (ASL Sign Video) ]
                |
                v
        [ ASL-TRM Web App ]
                |
                v
        [ TRM-Micro Model ]
   (gesture → word predictions)
                |
                v
        [ Custom LLM Layer ]
 (error correction + sentence formation)
                |
                v
[ Captions Generated for .mp4 Video ]

### What the app does:
✔ Accepts ASL sign language videos (.mp4)
✔ Recognizes hand/body gestures using TRM-Micro
✔ Corrects noisy predictions using a Custom LLM
✔ Generates meaningful sentence-level captions
✔ Outputs captions aligned with the input video
