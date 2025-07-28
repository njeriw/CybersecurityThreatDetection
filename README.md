# Detecting Cybersecurity Threats: Deep Learning with Bayesian Hyperparameter Tuning

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Introduction">Introduction</a></li>
    <li><a href="#built with">Built with</a></li>
    <li><a href="#data dictionary">Data Dictionary</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

![960x0](https://github.com/user-attachments/assets/686bbce6-51ae-4d87-b1f8-90cb002525fa)

### Introduction 
This project presents a deep learning solution for Cyber Threat Detection.

In today's evolving digital landscape, organizations face an escalating barrage of sophisticated cyber threats, from malware and phishing to denial-of-service attacks. This project addresses this challenge by leveraging the power of deep learning to identify malicious activities proactively.

Utilizing the BETH dataset, which simulates real-world network and kernel-process logs, this project focuses on designing and implementing a deep learning model to classify events as either benign (0) or malicious (1) based on the sus_label target. The preprocessed nature of the BETH dataset allows for a direct focus on model architecture, evaluation, and hyperparameter for effective threat identification.

By developing this model, the aim is to demonstrate proficiency in applying deep learning techniques to critical cybersecurity problems, ultimately contributing to more robust organizational defenses.

### Built with 
This project is built using `Python 3.12` and relies on several key libraries for its functionality, including `torch` for deep learning model development, `pandas` and `numpy` for efficient data manipulation and numerical operations, and `scikit-learn` for various machine learning utilities such as data splitting and performance evaluation. 
All necessary dependencies can be installed via pip using the command: `pip install torch pandas numpy scikit-learn sklearn`.
<p align="left">
  <a href="https://skillicons.dev">
    <img src="https://skillicons.dev/icons?i=pytorch,sklearn,anaconda,pandas,numpy" />
  </a>
</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Data Dictionary

| Column     | Description              |
|------------|--------------------------|
|`processId`|The unique identifier for the process that generated the event - int64 |
|`threadId`|ID for the thread spawning the log - int64|
|`parentProcessId`|Label for the process spawning this log - int64|
|`userId`|ID of user spawning the log|Numerical - int64|
|`mountNamespace`|Mounting restrictions the process log works within - int64|
|`argsNum`|Number of arguments passed to the event - int64|
|`returnValue`|Value returned from the event log (usually 0) - int64|
|`sus_label`|Binary label as suspicous event (1 is suspicious, 0 is not) - int64|

More information on the dataset: [BETH dataset](accreditation.md)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Acknowledgements
This project's structure and initial problem outline were inspired by the DataCamp "Cybersecurity Threat Detection" project. I acknowledge DataCamp for providing a valuable foundation and dataset context that guided this implementation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

