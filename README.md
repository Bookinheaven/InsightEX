
<a id="readme-top"></a>

[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
<br />
<div align="center">
  <a href="">
    <img src="assets/images/logo.png" alt="INSIGHT Logo" width="200" height="200">
  </a>
  <h1 align="center">INSIGHT</h1>
  <h3 align="center">Enhancing Customer Experience with AI-Driven Insights</h3>
  <p align="center">
    A smart AI system designed to optimize store layouts, enhance security, and elevate customer experience in real time.
  </p>
</div>
<hr>

<!-- TABLE OF CONTENTS -->
<details open>
  <summary><strong>Table of Contents</strong></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#features">Features</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#flowchart">Flowchart</a></li>
    <li><a href="#future-insights">Future Insights</a></li>
    <li><a href="#challenges-addressed">Challenges Addressed</a></li>
    <li><a href="#contributors">Contributors</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
InsightEX is an AI-powered system that helps businesses:
- **Track & analyze customer movement** in real time.
- **Optimize store layouts** based on customer behavior.
- **Enhance security** with real-time alerts and zone monitoring.
- **Improve staff assistance** by monitoring entry zones.
- **Identify high-traffic areas** using dynamic heatmaps.

With InsightEX, businesses can make data-driven decisions to boost efficiency and provide an exceptional customer experience.

### Built With
[![Python][Python]][Python-url]
[![OpenCV][OpenCV]][OpenCV-url]
[![YOLOv8][YOLOv8]][YOLOv8-url]
[![DeepSORT][DeepSORT]][DeepSORT-url]
[![OpenVINO][OpenVINO]][OpenVINO-url]
[![Numpy][Numpy]][Numpy-url]
[![PyTorch][PyTorch]][PyTorch-url]

<!-- FEATURES -->
## Features
- **Real-Time Analysis:** Processes live camera feeds or pre-recorded videos.
- **Heatmap Generation:** Visualize customer movement to identify high-traffic zones.
- **Dwell-Time Alerts:** Get notified if a customer lingers too long in a designated area.
- **Entry Zone Monitoring:** Customize zones for targeted staff assistance.
- **Multi-Source Zones Management:** Supports saving multiple zones in one shared file for different inputs.
- **Extensible & Scalable:** Easily integrate new features and improvements.


<!-- GETTING STARTED -->
## Getting Started

Follow these instructions to get your copy of InsightEX up and running.

### Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.7+**  
- **Git** (optional, for cloning the repository)  
- **Hardware:** A system with a capable CPU/GPU/NPU for optimal performance  

### Installation
1. **Clone the Repository**
    ```sh
    git clone https://github.com/Bookinheaven/InsightEX.git
    cd InsightEX
    ```
2. **Run the Setup Script**  
   For Windows:
    ```sh
    ./setup.bat
    ```
   For Linux:
    ```sh
    chmod +x setup.sh
    ./setup.sh
    ```
3. **Start the System**  
   The setup script automatically runs the InsightEX system. Alternatively, run:
    ```sh
    python InsightEX.py
    ```
    
<p align="right"><a href="#readme-top"><strong>Back to Top</strong></a></p>

<!-- USAGE -->
## Usage

InsightEX can process both video files and live camera feeds. Follow these steps:

1. **Start the System:**  
   Run the script using one of the commands below:
   - **Video File:**
     ```sh
     python InsightEX.py --video input_video.mp4
     ```
   - **Live Camera Feed:**
     ```sh
     python InsightEX.py --camera 0
     ```
     *(Replace `0` with the appropriate camera index if needed.)*

2. **Monitor the Outputs:**  
   - **Bounding Boxes & Keypoints:** View detected persons with annotated keypoints.
   - **Heatmaps:** Observe areas with high customer activity.
   - **Dwell-Time Alerts:** Receive console alerts when a customer remains in an area beyond the threshold.
   - **Entry Zone Monitoring:** Draw or erase entry zones with mouse clicks (left-click to add, right-click to remove).

3. **Stop the System:**  
   Use `Ctrl + C` in the terminal or close the window to stop the program.


<!-- FLOWCHART -->
## Flowchart
![Flowchart][flowchart-img]

<!-- FUTURE INSIGHTS -->
## Future Insights

- **Enhanced AI Models:** Further improvements in detection accuracy and behavioral analysis.
- **Multiple Camera Support:** Seamless monitoring of different areas simultaneously.
- **Customer Insights:** Analysis of demographics (age, gender, emotion) for personalized experiences.
- **Predictive Analytics:** Forecasting peak times and customer trends using historical data.
- **IoT Integration:** Connecting with smart devices for automated store management.
- **Mobile Integration:** Enabling remote access and monitoring via mobile devices.
- **Cloud Storage:** Storing and processing data remotely for enhanced scalability.
- **Advanced Security Alerts:** Detecting and responding to unusual behavior automatically.

<!-- CHALLENGES ADDRESSED -->

### Challenges Addressed
- No Customer Insights – Tracks movement to understand shopping behavior.

- Staff Mismanagement – Ensures employees are in the right place at the right time.

- Security Issues – Detects unusual activity and alerts staff.

- Bad Store Layouts – Helps improve design for better shopping flow.

- Missed Sales – Identifies popular areas to boost promotions.

- No Real-Time Tracking – Provides instant updates and alerts.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTORS -->
## Contributors
- [Tanvik Sri Ram](https://github.com/Bookinheaven)  
- [Choppa Sai Akshitha](https://github.com/akshithachoppa)  
- [Febin Renu](https://github.com/febinrenu)
<hr>
<a href="https://github.com/othneildrew/Best-README-Template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Bookinheaven/InsightEX" alt="contrib.rocks image" />
</a>
<p align="right"><a href="#readme-top"><strong>Back to Top</strong></a></p>

<!-- MARKDOWN LINKS & IMAGES -->
[tanvik]: https://github.com/Bookinheaven
[akshitha]: https://github.com/akshithachoppa 
[febin]: https://github.com/febinrenu
[forks-shield]: https://img.shields.io/github/forks/Bookinheaven/InsightEX.svg?style=for-the-badge
[forks-url]: https://github.com/Bookinheaven/InsightEX/network/members
[stars-shield]: https://img.shields.io/github/stars/Bookinheaven/InsightEX.svg?style=for-the-badge
[stars-url]: https://github.com/Bookinheaven/InsightEX/stargazers

[flowchart-img]: assets/images/flowchart.png
[contributors-img]: contrib.rocks/image?repo=Bookinheaven/InsightEX
[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://www.python.org
[OpenCV]: https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[OpenCV-url]: https://opencv.org
[YOLOv8]: https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black
[YOLOv8-url]: https://yolov8.com
[DeepSORT]: https://img.shields.io/badge/DeepSORT-0088CC?style=for-the-badge&logo=ai&logoColor=white
[DeepSORT-url]: https://pypi.org/project/deep-sort-realtime/
[OpenVINO]: https://img.shields.io/badge/OpenVINO-0071C5?style=for-the-badge&logo=intel&logoColor=white
[OpenVINO-url]: https://pypi.org/project/openvino/
[Numpy]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]: https://numpy.org
[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org