# PRS-PM-2023-07-01-GRP5-roomifAI
NUS-ISS PRS Practical Module Group 5 Project

RoomifAI (Room Visualizer) Version 2 is powered by advanced generative AI technology. RoomifAI can effortlessly create stunning design for any room in your home - be it living room, bedroom or dining room. All you have to do is provide a text prompt and RoomifAI will generate a unique and stylish interior design tailored to your preferences.


## PROJECT TITLE
## RoomifAI (Room Visualizer)

`<RoomifAI>`: <https://www.youtube.com/watch?v=fuN3DmhdAOs>

---
## CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Chua Jack Yune | A0269363U | Architect and Main Developer | jckynchua@yahoo.com |
| Borromeo, Angelie Quiapo | A0270177A | Project Coordinator and Data Manager| angelieqborromeo@gmail.com |
| Kwatt Ivy | A0269639H | Satable Diffusion Package Manager and Secondary Developer| kwattivy@gmail.com |
| Yeoh Wee Chye | A0165226H | Secondary Developer| weechye0532@gmail.com |
---

## EXECUTIVE SUMMARY / PAPER ABSTRACT / INTRODUCTION
Owning a property in Singapore is not uncommon, as Singapore's home ownership rate remains one of the highest in the world at nearly 90 per cent, thanks to government grant and financial support from parents (1). However, home prices are still rising and acquiring a home is becoming an important milestone in one’s lifetime, this leads to a growing demand in the interior design industry where one is willing to spend the time and effort to create a personal space that is both functional and aesthetically pleasing, for that property that could accompany for the rest of their lifetime.
But most often than not, the whole process of interior design is highly time consuming, usually the interior designer takes on many projects at once and relies on conventional methods of generating interior design concept drawings which require gathering reference images, producing 2D designs and 3D modeling and rendering. This process will undergo further refinement and feedback from clients, which is time consuming.
Our primary objective of this project is to provide interior design visualizations using a quick and easy method. Through textual prompts and an interactive graphic user interface, users can generate images of beautifully-designed room interiors based on their desired style or specifications. 
For professionals, the tool can be used to quickly and easily generate multiple prototype proposals to elicit client feedback, to be used for marketing or before moving on further along the design pipeline. The motivation behind this tool is to make room staging readily accessible, and easy to use, through the integration of deep learning technology. 
On the other hand, for the general user, the proposed tool allows them to generate interior design concept art without in depth knowledge in the interior design domain, while being able to work out different design ideas through the use of textual prompts. 

---
## SECTION 1 : AIM AND OBJECTIVE
`Refer to project report at Github Folder: ProjectReport`

---

## SECTION 2 : SYSTEM DESIGN
`Refer to project report at Github Folder: ProjectReport`

---

## SECTION 3 : MODEL
`Refer to project report at Github Folder: ProjectReport`

---

## SECTION 4 : APPLICATION INTEGRATION
`Refer to project report at Github Folder: ProjectReport`

---

## SECTION 5 : DISCUSSION
`Refer to project report at Github Folder: ProjectReport`

---

## SECTION 4 : REFERENCES
`Refer to project report at Github Folder: ProjectReport`

---

## SECTION 4 : APPENDIX
`Refer to project report at Github Folder: ProjectReport`

## SECTION 6 : PROJECT REPORT / PAPER
`Refer to project report at Github Folder: ProjectReport`

- Appendix A: Full Stable Diffusion 1.5 U-Net neural network diagram
- Appendix B: Individual Project Report
- Appendix C: User Guide
- Appendix D: Project Proposal

## USER GUIDE

`Refer to UserGuide.pdf in project report at Github Folder: ProjectReport`

This video intends to guide the user in installing and configuring RoomifAI in local machine.
[Installation and User Guide]()

### [ 1 ] To run the system in other/local machine:
### Install additional necessary libraries. This application works in python 3.10 only.

Installation can be done via Anaconda or Miniconda. The steps for installation of Anaconda or Miniconda and of the application libraries are given below.

Anaconda/Miniconda installation. Miniconda can be installed by downloading the installer for your operating system from the official website: [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html). For those who prefer a GUI, Anaconda would be preferable; the installation file is available via the link: [Anaconda](https://www.anaconda.com/download).

### The steps for installation of the packages are as follows:
In the bash command line, go to the project root folder.

> $ cd ~/SourceCode/src/main

Run the Conda command line tool to create a virtual environment and install the packages listed in the `environment.yml` file.
NOTE: If you are using MAC please use environment-mac.yml

> conda env create -f environment.yml

After the installation is completed, the backend API server can be started by running the Flask command.

> flask run

NOTE: For mac users if you encounter any issue relating to OPENMP you can add the following entry at the top of app.py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

If the API is running locally, the app can be opened via a browser at the URL http://localhost:5000/ 

---

**The [Reasoning Systems](https://www.iss.nus.edu.sg/executive-education/course/detail/reasoning-systems "Reasoning Systems") courses is part of the Analytics and Intelligent Systems and Graduate Certificate in [Intelligent Reasoning Systems (IRS)](https://www.iss.nus.edu.sg/stackable-certificate-programmes/intelligent-systems "Intelligent Reasoning Systems") series offered by [NUS-ISS](https://www.iss.nus.edu.sg "Institute of Systems Science, National University of Singapore").**
