# Large Multimodal Models: Vision-Language Project

A university project exploring **Large Multimodal Models (LMMs)** and **Vision-Language Models (VLMs)** using Meta's Llama 3.2 Vision.

This project investigates how a vision-language model can interpret images, identify objects, describe scenes, answer visual questions, and perform visual reasoning.

**Repository:** [LMM-Vision-Language-Project](https://github.com/abhinash7729/LMM-Vision-Language-Project)

---

## Project Overview

Large Multimodal Models combine information from different modalities, such as text and images, to understand and respond to complex inputs.

This project uses **Meta Llama 3.2 11B Vision Instruct** to explore image understanding and visual reasoning through a series of practical experiments.

The project also includes a Streamlit-based interface designed to make image analysis easier to access.

## Objectives

- Understand the architecture and capabilities of Vision-Language Models.
- Explore image preprocessing and multimodal model inference.
- Generate image descriptions and captions.
- Perform visual question answering (VQA).
- Identify objects, activities, and scene characteristics.
- Explore visual reasoning and contextual interpretation.
- Build an interactive application for image analysis.

## Model and Technologies

| Component | Technology |
|---|---|
| Vision-language model | Meta Llama 3.2 11B Vision Instruct |
| Model repository | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| Programming language | Python |
| Deep learning framework | PyTorch |
| Model library | Hugging Face Transformers |
| Interface | Streamlit |
| API communication | FastAPI and Requests |
| Cloud execution option | Google Colab |
| Tunneling | ngrok |

## Repository Structure

```text
LMM-Vision-Language-Project/
│
├── data/
│   ├── input_image.jpg
│   └── sample_image.jpg
│
├── src/
│   ├── inspect_model.py
│   ├── experiment_02.py
│   ├── experiment_03.py
│   ├── experiment_04.py
│   ├── experiment_05.py
│   └── experiment_06.py
│
├── results/
│   ├── model_config.json
│   ├── experiment_02_results.json
│   ├── experiment_03_caption.txt
│   ├── experiment_04_vqa_results.json
│   ├── experiment_05_object_scene_results.json
│   ├── experiment_06_visual_reasoning_results.json
│   └── [additional application results]
│
├── reports/
│   ├── experiment_01.md
│   ├── experiment_02.md
│   ├── experiment_03.md
│   ├── experiment_04_vqa.md
│   └── experiment_05_object_scene_recognition_report.md
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Experiments

The following experiments form the practical part of this project.

| Experiment | Description | Result status |
|---|---|---|
| 01 | Model architecture and configuration inspection | Results/report available |
| 02 | Image preprocessing and inspection | Results/report available |
| 03 | Image understanding and caption generation | Result available |
| 04 | Visual Question Answering (VQA) | Result available |
| 05 | Object and Scene Recognition | Result available |
| 06 | Visual Reasoning and Scene Understanding | Result available |

### Experiment 03: Image Understanding

The model is provided with an input image and generates a natural-language description of its visual content.

**Output:** `results/experiment_03_caption.txt`

### Experiment 04: Visual Question Answering

The model receives an image and answers questions about its contents, surroundings, and activities.

Three questions were tested using the input image.

| Question | Recorded inference time |
|---|---:|
| Identify the main objects or people | 678.74 seconds |
| Describe the activity or event | 1,597.33 seconds |
| Describe the background and surroundings | 2,112.00 seconds |

The sum of the three recorded inference times is approximately **73 minutes and 8 seconds**.

**Important:** This experiment's saved execution record identifies CPU execution, with GPU unavailable. The third answer is truncated in the saved JSON output and should be recovered before treating it as a complete result.

Result file: `results/experiment_04_vqa_results.json`

### Experiment 05: Object and Scene Recognition

The model analyzes the image and produces structured observations about objects, scene characteristics, activities, and spatial relationships.

**Key observations from the saved result:**

- A small brown dog is identified as the main object.
- The scene is interpreted as an outdoor grassy area or park.
- The dog is described as sitting and looking towards the camera.
- The background is described as blurred.
- The scene is characterized as peaceful and calm.

**Recorded execution time:** 18.37 seconds.

Result file: `results/experiment_05_object_scene_results.json`

### Experiment 06: Visual Reasoning and Scene Understanding

This experiment explores the model's ability to interpret visual context and distinguish observations from inferences.

**Key observations from the saved result:**

- The model identifies a small brown dog sitting in tall grass.
- It interprets the environment as a natural outdoor setting.
- It identifies the dog as likely being a domesticated pet.
- It describes the dog's posture as relaxed.
- It acknowledges uncertainty about the dog's breed, age, and health.

**Recorded execution time:** 38.63 seconds.

Result file: `results/experiment_06_visual_reasoning_results.json`

## Experiment 07: Image Comparison and Cross-Image Reasoning

### Objective
To evaluate the ability of Meta Llama 3.2 11B Vision Instruct to analyze and compare two images, identify similarities and differences, and perform cross-image visual reasoning.

### Methodology
- Load two input images.
- Process both images using the Llama 3.2 Vision model.
- Generate descriptions of each image.
- Identify common objects, visual similarities, and differences.
- Compare scene characteristics and relationships.
- Generate an overall comparative conclusion.

### Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Meta Llama 3.2 11B Vision Instruct
- Pillow
- Google Colab

### Files
- Source code: `src/experiment_07_image_comparison.py`
- Results: `results/experiment_07_image_comparison_results.json`
- Report: `reports/experiment_07_image_comparison_report.md`

### Expected Outcome
A structured comparison of two images, including their descriptions, similarities, differences, and visual reasoning.

---

## Experiment 08: OCR and Text Understanding

### Objective
To evaluate the ability of Meta Llama 3.2 11B Vision Instruct to recognize, extract, and understand text contained in images.

### Methodology
- Retrieve the input image directly from the GitHub repository.
- Input image: `data/text_image.png`.
- Load the image using Pillow.
- Process the image using the Llama 3.2 Vision model.
- Extract visible text while preserving spelling, punctuation, headings, and numbers.
- Organize the extracted text and provide a summary.
- Identify uncertain or unreadable text rather than inventing content.

### Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Meta Llama 3.2 11B Vision Instruct
- Pillow
- Requests
- Google Colab

### Files
- Source code: `src/experiment_08_ocr.py`
- Results: `results/experiment_08_ocr_results.json`
- Report: `reports/experiment_08_ocr_report.md`
- Input image: `data/text_image.png`

### Expected Outcome
A structured OCR output containing extracted text, document organization, and a summary of the image's textual content.


## Application

The project includes a Streamlit application (`app.py`) that provides an interface for submitting images and requesting visual analysis.

The application is designed to communicate with a model-serving backend hosted through Google Colab.

The intended workflow is:

1. Start the model backend in Google Colab.
2. Expose the backend through an ngrok tunnel.
3. Copy the backend URL and API key into the Streamlit application.
4. Upload an image and submit an analysis request.
5. View the model's response.
6. Save the generated result locally in the `results/` directory.

The Colab backend must be running and accessible for this remote-execution workflow to work.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/abhinash7729/LMM-Vision-Language-Project.git
cd LMM-Vision-Language-Project
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure model access

The Llama 3.2 Vision model is hosted on Hugging Face and may require access approval.

Visit the model page:

[Meta Llama 3.2 11B Vision Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

Accept the applicable access terms and authenticate with Hugging Face as required by your execution environment.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

For remote inference, start the corresponding Colab backend and configure the backend URL and API key in the application.

**Note:** Local model inference may require substantial RAM and suitable GPU resources. The application’s remote-execution workflow depends on an active Colab session and a valid ngrok endpoint.

## Results and Reports

Experiment outputs are stored in the `results/` directory, primarily as JSON and text files.

Experiment documentation is stored in the `reports/` directory.

These files provide the recorded outputs and execution details for reviewing and reproducing the experiments.

## Limitations

- The project uses a limited set of test images and is intended as an educational exploration, not a formal benchmark.
- Model-generated descriptions and inferences may contain inaccuracies.
- Experiment 04 was recorded as running on CPU, resulting in long inference times.
- The saved third response in Experiment 04 is incomplete.
- Device information was not recorded in the supplied Experiment 05 and Experiment 06 result files.
- Remote inference depends on the availability of the Colab runtime and ngrok connection.

## Future Improvements

- Complete and document additional image-understanding experiments.
- Recover the incomplete Experiment 04 response.
- Compare outputs across different vision-language models.
- Evaluate performance using a larger and more diverse image dataset.
- Improve the Streamlit interface and error handling.
- Add screenshots and demonstrations of the application.
- Explore OCR, image comparison, and other multimodal tasks.

## References

- [Meta Llama 3.2](https://www.llama.com/)
- [Llama 3.2 Vision on Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/index)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## Author and Project

**Project:** Large Multimodal Models (LMMs) / Vision-Language Models (VLMs)

**GitHub:** [abhinash7729](https://github.com/abhinash7729)

This project is developed for academic and educational purposes to explore the practical capabilities of modern vision-language models.

