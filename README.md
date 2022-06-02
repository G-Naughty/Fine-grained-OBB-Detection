# Fine-grained-OBB-Detection
The third place winning solution (3/220) in the track of Fine-grained Object Recognition in High-Resolution Optical Images, 2021 Gaofen Challenge on Automated High-Resolution Earth Observation Image Interpretation. 
## Solution
* **models**  
  Under folder：config/fair1m
  Strong backbone: add swin transformer  as backbone
  storng head:     adapt double head to ROI transformer and ReDet
* **Training trick** 
  Under folder：config/fair1m
  Fine-tuning with SWA and cosine learning rate
