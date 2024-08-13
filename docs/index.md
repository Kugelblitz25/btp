---
Author: Vighnesh Nayak
Date: 06/08/2024
Project: Machine Learning based Video Surveillance
tags:
  - "#ml"
  - cv
---
# Machine Learning based Video Surveillance
---
## Requirements
---
- Extract the features like faces, wearables and, unique features like tattoos or scars from an individual entering the system with timestamp.
	- Separate the background from video using DIP for better performance.
	- Ensure efficient storage and retrieval of individuals making sure there are no duplicates.
	- Track the same person across different cameras at different angles and create a history of his movements.
- Summarize the activities in a given room.
	- Number of people at any time
	- Heatmap of popularity of the room
- Action classification into suspicious or not.
	- Digital vault: Create a bounding box around regions and objects of interest and log all individuals who entered it.
## High Level Algorithms
---
### Tasks
- [ ] Foreground separation and human object separation.
	- [ ] MOG2
	- [ ] KNN
	- [ ] CNT
	- [ ] RPCA
	- [ ] Semantic Segmentation
	- Try all the algorithms and find the most efficient and fast algorithm. We might even have to combine multiple of them to get good result.
## Low Level Algorithms
---

## Hardware
---
