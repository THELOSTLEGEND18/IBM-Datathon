# IBM-Datathon Project
# Title : VisionGuard

## ONL-185 (Team Number)

This repository contains the project files for the IBM Z Datathon 2024, focusing on developing an advanced content moderation system using machine learning algorithms.

Project Overview
Our project aims to create a robust content moderation system capable of detecting inappropriate content in images and videos. 

The system focuses on two main aspects:
Blur Detection: Identifying and flagging blurred content that may be used to obscure inappropriate material.
Nude Content Detection: Utilizing machine learning to detect and filter out nude or explicit content.
By combining these features, we strive to create a comprehensive tool for maintaining safe and appropriate online environments.

Key Features

Blur Detection Algorithm: Implements advanced image processing techniques to identify intentionally blurred areas in images and videos.
Nude Content Detection: Utilizes a state-of-the-art machine learning model to accurately identify and flag nude or explicit content.
Streamlit Integration: Provides an intuitive user interface for easy interaction with the moderation tools.
Video Processing: Extends moderation capabilities to video content, analyzing frame by frame.
Exception Handling: Implements custom exception rules to manage edge cases and improve reliability.

## Repository Structure

- `Blur/`: Updated documentation related to blur detection
- `Logs/`: Feature addition and use of streamlit
- `Models/`: Feature addition and use of streamlit
- `Prosses/`: Commit information
- `_pycache_/`: Documentation update
- `output/`: Commit information
- `.gitattributes`: Added Blur Exception Rule
- `BlurException.rule`: Added Blur Exception Rule
- `__init__.py`: Added Blur Exception Rule
- `app.py`: Implemented ML algorithm
- `main.py`: Feature addition and use of streamlit
- `nsfw_vid.mp4`: Implemented ML algorithm
- `nude_detector.py`: Added content moderation frameworks
- `requirements.txt`: Implemented ML algorithm
- `video.py`: Commit information

## Getting Started

1. Clone the repository
2. Check wheter your system has python version 3.10.0 so that all libraries used in this project will run without errors and are compatible with this python version
3. Open the folder in VScode and open a new terminal and then follow step 4
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   pip install streamlit
   pip install open cv
   ```
5. Run the main application:
   ```
   streamlit run app.py
   ```

UI Interface
![image](https://github.com/user-attachments/assets/f8a94b01-2ab6-4487-b904-39dec3d73a3a)

Technical Details
Machine Learning Model
The project uses an ONNX model for content detection, located at Models/best.onnx. This model is loaded by the NudeDetector class for analyzing video frames.

Video Processing Pipeline
The uploaded video is temporarily saved to the server.
The VideoProcessor class processes the video frame by frame.
Each frame is analyzed by the NudeDetector for sensitive content.
Detected sensitive areas are blurred.
The processed frames are compiled back into a video.
The final video is made available for download and preview.

Error Handling
The application includes error handling for cases such as missing model files or processing failures.
Temporary files are cleaned up after processing to manage server storage efficiently.
   
Usage

Image Moderation:
Upload an image through the Streamlit interface.
The system will analyze the image for blur and nude content.
Results will be displayed, indicating any detected issues.

Video Moderation:
Upload a video file.
The system will process the video frame by frame.
A summary of detected content issues will be provided.

Custom Analysis:
Use app.py for custom integration of the moderation algorithms into other projects.

Ethical Considerations

As developers of a content moderation system, we recognize the ethical implications of our work:
Privacy: Our system processes user-uploaded content, but we ensure that no data is stored longer than necessary for analysis.
Bias: We continuously work to reduce algorithmic bias by diversifying our training data and regularly auditing our model's performance across different demographics.
Transparency: We're committed to being open about the capabilities and limitations of our system.
False Positives: We acknowledge the potential harm of incorrectly flagging content and have implemented an appeal system for users.

Future Roadmap

We're committed to continually improving our content moderation system. Here are some features and improvements we're planning for future releases:
Multi-language Support: Extending text-based moderation to multiple languages.
Audio Content Moderation: Implementing speech recognition and audio content analysis.
Real-time Streaming Support: Adapting our system to moderate live video streams.
API Development: Creating a robust API for easy integration with various platforms.
Customizable Rulesets: Allowing clients to fine-tune moderation rules to their specific needs.

## Contributors

- THELOSTLEGEND18 (Suchir Thaokar)
- nathan31dsouza (Nathan Dsouza)
- jeetshorey123 (Jeet Shorey)
- Emilp10 (Emil Pereira)

## Languages Used

The primary language used in this project is Python and deployed on streamlit.

