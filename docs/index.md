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
## Goals
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

## POA
---
Our initial plan was to use [[Background Separation|background separation]] algorithms to extract the moving foreground to simplify the further image processing tasks. Then use machine learning based models to identify the different parts and features from the humans in foreground for reidentification. 
There are several problems with this approach:
-  Foreground is irregular in shape, hence dimensions of the image to be processed does not change just the information does.
- To identify the humans in the image we still need a ML model as background separation identifies everything that moves

So on prof's suggestion, we changed our plans to use background separation. We are currently using a difference based [[Motion Detection|motion detection]] to identify the frames in which motion occurs. We then use models like **YOLOv8** to identify the different objects in the frame, then pass the bounding box of each human in that frame reidentification and storage.
This approach requires significant [[Machine Learning System Design|system design]] as running ml models takes much longer time in comparison to video recording. We need to do asynchronous handling of each frame. This requires implementation of message queue's like **RabbitMQ**, **Reddis** or **Apache Kafka**. We can use **Celery** for this purpose. A simpler approach is to use some thing like **ThreadPoolExecutor** or **asyncio** in python. But these have limitations in terms of number of tasks that can be performed simultaneously and getting results and status of process and fault tolerance.

| Feature                    | ThreadPoolExecutor | asyncio            | Celery            | Pure Message Queue             |
| -------------------------- | ------------------ | ------------------ | ----------------- | ------------------------------ |
| Distribution               | Single machine     | Single machine     | Multiple machines | Multiple machines              |
| Scalability                | Limited to CPU     | Good for I/O tasks | Highly scalable   | Highly scalable                |
| Persistence                | No                 | No                 | Yes               | Yes                            |
| Task Scheduling            | No                 | Limited            | Yes               | Requires custom implementaion  |
| Result Backend             | In-memory          | In-memory          | Configurable      | Requires custom implementation |
| Ease of Implementation     | Easy               | Moderate           | Moderate          | Complex                        |
| Suitable for               | CPU-bound tasks    | I/O-bound tasks    | Mixed workloads   | High-volume messaging          |
| Built-in Worker Management | Yes                | No                 | Yes               | No                             |
| Language Agnostic          | No                 | No                 | Partially         | Yes                            |
| Monitoring/Admin Tools     | Limited            | Limited            | Yes               | Varies by system               |
| Fault Tolerance            | Limited            | Limited            | Yes               | Yes                            |
Message Queues provide suitable in the following cases:
- High volume of tasks: If you're processing thousands or millions of images per day, a robust message queue can help manage the load.
- Microservices architecture: If your YOLO processing is part of a larger microservices ecosystem, a message queue can facilitate communication between services.
- Complex routing requirements: If you need to route different types of tasks to different processors, many message queue systems offer sophisticated routing capabilities.
- Strong delivery guarantees: If you need assured delivery of tasks, even in the face of network issues or server crashes, message queues often provide stronger guarantees than simpler systems.
These match pretty good with our goals.

There are also two ways to implement these, function based and API based. Function based implementation is simple but suffers from scalability and resource constraint issues, while API based implementation has high maintenance cost . 

| Aspect                   | API Approach                                         | Function Approach                                |
| ------------------------ | ---------------------------------------------------- | ------------------------------------------------ |
| **Use Case**             | Web services, microservices, distributed systems     | Local applications, scripts, embedded systems    |
| **Accessibility**        | Can be accessed over network (HTTP/HTTPS)            | Only accessible within the local application     |
| **Scalability**          | Easier to scale horizontally                         | Limited to local resources                       |
| **Deployment**           | Requires server setup and management                 | Simple, runs as part of the main application     |
| **Latency**              | Higher due to network communication                  | Lower, direct function calls                     |
| **Flexibility**          | Can easily add new endpoints or modify existing ones | Changes might require modifying client code      |
| **Development Overhead** | Higher initial setup time                            | Lower initial setup time                         |
| **Execution Context**    | Runs in a separate process or machine                | Runs in the same process as the main application |
One other thing that we can look into is the use of **gRPC** in place of **HTTP/HTTPS** if we are using API based approach.

## To-Do
---
- [ ] Discuss with prof on how to proceed with the implementations.
- [ ] Look at other models similar to YOLOv8
- [ ] Research on human reidentification
- [ ] Look at different methods for motion detection