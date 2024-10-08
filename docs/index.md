---
Author: Vighnesh Nayak
Date: 06/08/2024
Project: Machine Learning based Video Surveillance
tags:
  - "#ml"
  - cv
---
# Machine Learning-based Video Surveillance
---
## Introduction
---

This project aims to develop a sophisticated video surveillance system that leverages machine learning techniques to enhance security, improve efficiency, and provide valuable insights. The primary goals are to accurately identify individuals, track their movements, and analyze activities within the monitored environment.

## Goals
---

- Extract features like faces, wearables, and unique features like tattoos or scars from an individual entering the system with a timestamp.
    - Separate the background from the video using DIP for better performance.
    - Ensure efficient storage and retrieval of individuals, ensuring no duplicates exist.
    - Track the same person across different cameras at different angles and create a history of his movements.
- Summarize the activities in a given room.
    - Number of people at any time
    - Heatmap of the popularity of the room
- Action classification into suspicious or not.
    - Digital vault: Create a bounding box around regions and objects of interest and log all individuals who entered it.

## Approach

Our initial plan was to use [background separation](Background%20Separation) algorithms to extract the moving foreground to simplify the further image processing tasks. Then, use machine learning-based models to identify the different parts and features from the humans in the foreground for re-identification. There are several problems with this approach:

- The foreground is irregular in shape. Hence, the image's dimensions to be processed do not change; only the information does.
- To identify the humans in the image, we still need an ML model, as background separation identifies everything that moves

So, on the professor's suggestion, we changed our plans to use motion detection instead pf background separation. We currently use difference-based [motion detection](Motion%20Detection) to identify the frames in which motion occurs. We then use models like **YOLOv8** to identify the different objects in the frame, then pass the bounding box of each human in that frame for re-identification and storage. This approach requires significant [system design](Machine%20Learning%20System%20Design) as running ML models takes much longer than video recording. We need to handle each frame asynchronously. This requires implementing message queues like **RabbitMQ**, **Redis**, or **Apache Kafka**. We can use **Celery** for this purpose. A more straightforward approach is to use something like **ThreadPoolExecutor** or **asyncio** in Python. However, these have limitations in terms of the number of tasks that can be performed simultaneously and getting results, the status of the process, and fault tolerance.

| Feature                        | ThreadPoolExecutor | asyncio            | Celery            | Pure Message Queue             |
| ------------------------------ | ------------------ | ------------------ | ----------------- | ------------------------------ |
| **Distribution**               | Single machine     | Single machine     | Multiple machines | Multiple machines              |
| **Scalability**                | Limited to CPU     | Good for I/O tasks | Highly scalable   | Highly scalable                |
| **Persistence**                | No                 | No                 | Yes               | Yes                            |
| **Task Scheduling**            | No                 | Limited            | Yes               | Requires custom implementaion  |
| **Result Backend**             | In-memory          | In-memory          | Configurable      | Requires custom implementation |
| **Ease of Implementation**     | Easy               | Moderate           | Moderate          | Complex                        |
| **Suitable for**               | CPU-bound tasks    | I/O-bound tasks    | Mixed workloads   | High-volume messaging          |
| **Built-in Worker Management** | Yes                | No                 | Yes               | No                             |
| **Language Agnostic**          | No                 | No                 | Partially         | Yes                            |
| **Monitoring/Admin Tools**     | Limited            | Limited            | Yes               | Varies by system               |
| **Fault Tolerance**            | Limited            | Limited            | Yes               | Yes                            |

Message Queues are suitable in the following cases:

- High volume of tasks: If we want to process thousands or millions of images per day, a robust message queue can help manage the load.
- Microservices architecture: A message queue can facilitate communication between services if YOLO processing is part of a larger microservices ecosystem.
- Complex routing requirements: Many message queue systems offer sophisticated routing capabilities to route tasks to different processors.
- Strong delivery guarantees: If assured delivery of tasks is required, message queues often provide stronger guarantees than simpler systems, even in network issues or server crashes. These match pretty well with our goals.

We plan to use **[Celery](Celery)** for our task with *Redis* as message broker.

There are also two ways to implement these: function-based and API-based. Function-based implementation is simple but suffers from scalability and resource constraint issues, while API-based implementation has a high maintenance cost.

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

In an API-based approach, we have another choice to make. We can use either **HTTP** or **WebSockets** to communicate between the motion detection and tracking modules. Using WebSockets in a very crowded or active environment like public places makes sense since there is a high likelihood of motion in almost all the frames. On the other hand, keeping the WebSocket connection open is wasteful in more controlled environments and at times of low activity. In such cases, we might need to switch to HTTP. Another thing we can look into is the use of **gRPC** instead of **HTTP/HTTPS**.

## To-Do
---
- [ ] Discuss with prof on how to proceed with the implementations.
- [ ] Look at other models similar to YOLOv8
- [ ] Research on human reidentification
- [ ] Look at different methods for motion detection