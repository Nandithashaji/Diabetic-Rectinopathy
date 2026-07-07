

# Diabetic Retinopathy Detection Using MobileNetV2

---

# Abstract

Diabetic Retinopathy (DR) is an eye disease caused by diabetes and can lead to blindness if it is not detected early. This project develops a web-based system to detect DR from retinal images using deep learning. The model used is **MobileNetV2**, which classifies the severity of DR from uploaded retinal images.
The system is built using **Django** for backend, **MySQL** for database, and **HTML, CSS, JavaScript, Bootstrap, and jQuery** for frontend. Image preprocessing is done using **OpenCV**, and the model is trained using **TensorFlow/Keras**.
The system helps in quick and accurate DR detection and can support doctors in early diagnosis.

---

# Chapter 1: Introduction

## 1.1 Background

Diabetic Retinopathy is one of the major complications of diabetes that affects the retina of the eye. If it is not diagnosed at an early stage, it may cause vision loss or blindness.
Usually, DR is detected manually by eye specialists by examining retinal images. This process takes time and requires expert knowledge. Because of this, there is a need for an automatic system that can quickly detect DR.
Deep learning models, especially CNN-based models like **MobileNetV2**, are useful for image classification and can help in building such an automated system.

## 1.2 Existing System

The existing system mainly depends on **manual diagnosis by ophthalmologists** using retinal fundus images.
This approach has several drawbacks:

* It is **time-consuming**
* It depends on the **experience of the doctor**
* It is **costly**
* It is **not easily available in rural areas**

Some automated systems already exist, but many of them need high computational power or do not have an easy-to-use interface.

## 1.3 Problem Statement

Manual detection of Diabetic Retinopathy is slow, expensive, and not always accessible to all patients. There is a need for a system that can:

* detect DR automatically,
* reduce human effort,
* give faster results,
* and support early treatment.

## 1.4 Need and Significance

This project is important because:

* DR can cause blindness if not detected early.
* An automated system can help in **faster screening**.
* It reduces the burden on doctors.
* It can be used in **rural and remote areas**.
* It provides a **cost-effective solution** for early diagnosis.

## 1.5 Objectives

The main objectives of the project are:

* To build a **web-based DR detection system**
* To use **CNN and MobileNetV2** for retinal image classification
* To preprocess retinal images for better accuracy
* To provide a user-friendly platform for image upload and result display
* To help doctors and patients with early diagnosis

## 1.6 Scope

The scope of this project includes:

* collecting retinal image datasets,
* preprocessing the images,
* training a deep learning model,
* deploying the trained model in a web application,
* and providing real-time prediction of DR.

This system acts as a **support tool** for doctors and does not replace medical experts.

## 1.7 Scheme / Working of the System

The project follows these steps:

1. Collect retinal image data
2. Preprocess the images
3. Train the MobileNetV2 model
4. Evaluate model performance
5. Integrate the model into a Django web application
6. Allow users to upload retinal images and get prediction results

---

# Chapter 2: Literature Review

The literature review studies previous work done in Diabetic Retinopathy detection.

### Main findings from reviewed papers:

* Earlier DR detection methods were mostly **manual** or based on **traditional image processing**.
* Machine learning methods such as **SVM, Decision Trees, and Random Forest** were later used.
* Deep learning methods, especially **CNNs**, improved performance significantly.
* **MobileNetV2** was found to be a lightweight and accurate model suitable for DR classification.
* Some papers used datasets such as **EyePacs, APTOS, Messidor, and IDRiD**.
* Researchers also highlighted challenges such as:

  * imbalanced datasets,
  * poor image quality,
  * lack of interpretability,
  * and difficulty in deployment in real-world healthcare settings.

### Conclusion of literature review

Deep learning-based systems are more effective than traditional methods for DR detection, and MobileNetV2 is a suitable model for building a practical DR identification system.

---

# Chapter 3: System Development

## 3.1 Objectives of System Development

The system development process includes:

* requirement analysis,
* system design,
* data collection,
* preprocessing,
* model training,
* implementation,
* testing,
* and deployment.

## SDLC Phases Used

### 1. Planning Phase

In this phase, the project goals, requirements, tools, risks, and timeline are decided.

### 2. Analysis Phase

In this phase:

* system requirements are studied,
* existing system limitations are analyzed,
* dataset and technology stack are selected,
* and performance measures are identified.

### 3. Design Phase

In this phase:

* system architecture is prepared,
* user interface is designed,
* database structure is planned,
* and data flow is defined.

### 4. Implementation Phase

In this phase:

* frontend and backend are developed,
* database is connected,
* model is trained and integrated,
* image preprocessing is implemented,
* and APIs are developed.

### 5. Testing Phase

In this phase:

* unit testing,
* integration testing,
* functional testing,
* performance testing,
* accuracy testing,
* and security testing are carried out.

### 6. Deployment Phase

In this phase:

* the system is hosted on a server or cloud,
* model and database are deployed,
* frontend and backend are connected,
* and user access is enabled.

## 3.2 Hypothesis

The project assumes that:

* MobileNetV2 and CNN can detect DR accurately from retinal images.
* preprocessing improves image quality and classification accuracy.
* a web-based system can provide fast and accessible DR detection.
* the system can reduce manual effort and support doctors.

## 3.3 Sources of Data

The dataset is collected mainly from:

* **Kaggle**
* publicly available retinal image datasets
* medical research repositories

## 3.4 Primary and Secondary Data

### Primary Data

Primary data consists of retinal images collected mainly from **Kaggle datasets**.

### Secondary Data

Secondary data comes from:

* medical repositories,
* research institutions,
* and published research papers.

### 3.4.3 Block Diagram

The block diagram represents the full workflow:
**Input retinal image → preprocessing → model classification → result generation → display in web application**

---

# Chapter 4: Software Requirement Specification (SRS)

## 4.1 Introduction

The system is a web-based application for DR detection using deep learning. It processes retinal images and classifies them into DR severity levels.

## 4.2 Use Case View

Main users of the system are:

* **Admin** – manages the system
* **Doctor / Healthcare Professional** – uploads retinal images and gets results
* **Patient** – views report

## 4.2.1 Use Case: Upload Retinal Image

* The doctor logs in to the system
* Uploads the retinal image
* The system processes the image
* The model predicts the DR class
* The result is displayed and stored

## 4.3 Data Flow Diagram (DFD)

### Level 0 DFD

* User uploads image
* System preprocesses image
* Model classifies image
* Result is stored
* User retrieves report

### Level 1 DFD

* User authentication
* Image preprocessing
* Deep learning model prediction
* Result storage in database
* Result display in frontend

## 4.4 Activity Diagram

The workflow is:

1. User logs in
2. User uploads retinal image
3. Image is preprocessed
4. Model predicts DR class
5. Result is stored
6. User views the report

## 4.5 Non-Functional Requirements

The system should provide:

* **fast performance**
* **security**
* **scalability**
* **reliability**
* **easy usability**

## 4.6 Sequence Diagram

The sequence is:

* login → authentication
* upload image → preprocessing
* classification → store result
* display result to user

---

# Chapter 5: Detailed Design Document

## 5.1 Introduction

This chapter explains the internal design of the system, including architecture, preprocessing, and model details.

## 5.2 System Architecture

The system follows **MVC architecture**.

### Components:

* **Frontend:** HTML, CSS, Bootstrap, JavaScript, jQuery
* **Backend:** Django
* **Database:** MySQL
* **Model:** TensorFlow/Keras with MobileNetV2

## 5.3 Pre-Processing Details

Before giving the image to the model, preprocessing is done using OpenCV:

* **Resizing** the image to 224 × 224
* **Contrast enhancement**
* **Noise reduction**
* **Normalization** of pixel values

This helps improve model accuracy.

---

# Chapter 6: Software Testing

## 6.1 Types of Testing Used

### Unit Testing

Checks individual components such as preprocessing, prediction, and database queries.

### Integration Testing

Checks whether frontend, backend, database, and model work together properly.

### Functional Testing

Ensures all features such as image upload, prediction, and result display work correctly.

### Performance Testing

Checks response time and system speed.

### Accuracy Testing

Measures model performance using test data.

### Security Testing

Checks user authentication and data protection.

### User Acceptance Testing

Collects feedback from users such as doctors or healthcare professionals.

---

# Chapter 7: Results and Discussion

The project produced good results in detecting Diabetic Retinopathy from retinal images.

### Main outcomes:

* The MobileNetV2 model achieved **high classification performance**
* Preprocessing improved the quality of input images
* Average prediction time was **less than 5 seconds**
* The web interface worked smoothly for image upload and result display
* The system showed good accuracy in classifying DR cases

However, some minor errors occurred in early-stage DR classification, which can be improved with more training data.

## 7.1 Confusion Matrix

The confusion matrix is used to compare **actual class** and **predicted class**.

### Terms in confusion matrix:

* **True Positive (TP):** DR correctly detected
* **True Negative (TN):** No DR correctly identified
* **False Positive (FP):** Healthy eye wrongly predicted as DR
* **False Negative (FN):** DR case missed by the system

## 7.1.3 Performance Metrics

The model is evaluated using:

* **Accuracy** – overall correct predictions
* **Precision** – how many predicted DR cases are actually DR
* **Recall** – how many actual DR cases are detected
* **F1 Score** – balance between precision and recall

## 7.1.4 ROC Curve

ROC curve is used to measure classification performance by comparing:

* **True Positive Rate**
* **False Positive Rate**

A good ROC curve indicates strong model performance.

## 7.2 Snapshots

The report includes snapshots of:

* login page
* registration pages
* user home page
* questionnaire page
* image upload page
* technician home page
* confusion matrix and model evaluation pages

---

# Chapter 8: Conclusion and Future Scope

## 8.1 Conclusion

This project successfully developed a **web-based Diabetic Retinopathy Identification System** using **MobileNetV2**.
The system can:

* preprocess retinal images,
* classify them into DR severity levels,
* and provide results quickly through a web application.

The use of Django, MySQL, TensorFlow/Keras, and OpenCV helped in building a practical and effective diagnostic support system.

## 8.1.1 Key Achievements

* Successful image preprocessing and DR classification
* Web-based interface for easy usage
* Good model accuracy and fast prediction
* Secure and scalable MVC-based system
* Proper testing and performance evaluation

## 8.2 Future Scope

### 8.2.1 Improvement in Model Accuracy

* Use larger and more diverse datasets
* Try advanced models such as EfficientNet or Vision Transformers

### 8.2.2 Integration with Cloud and IoT

* Deploy the system on cloud for remote access
* connect with IoT-based retinal cameras

### 8.2.3 Multi-Disease Detection

* Extend the system to detect other eye diseases like:

  * glaucoma
  * macular degeneration
  * hypertensive retinopathy

### 8.2.4 Enhanced User Experience

* Develop a mobile app
* improve interface and result visualization

### 8.2.5 Integration with Electronic Health Records (EHR)

* connect the system with hospital patient record systems
* support secure data sharing

### 8.2.6 AI Explainability

* add explainable AI methods to show why the model predicted DR

### 8.2.7 Deployment in Low-Resource Settings

* optimize the system for rural clinics and areas with limited medical resources

---

# Conclusion

The DiaRet project is a **deep learning-based web application for Diabetic Retinopathy detection**. It uses **retinal images**, **MobileNetV2**, and **web technologies** to provide fast and accurate DR classification. The system helps in early detection, supports healthcare professionals, and has strong future scope for improvement and real-world medical use.
