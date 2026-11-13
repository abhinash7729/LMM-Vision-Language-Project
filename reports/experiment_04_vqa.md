# Experiment 04 – Visual Question Answering (VQA)

## 1. Objective

The objective of this experiment is to evaluate the ability of **Meta Llama 3.2 11B Vision Instruct** to answer natural-language questions about the contents of an input image. This task is known as Visual Question Answering (VQA), because the model uses both visual information from an image and the text of a question to generate an answer.

## 2. Model and Environment

| Item | Details |
|---|---|
| Model | `meta-llama/Llama-3.2-11B-Vision-Instruct` |
| Task | Visual Question Answering (VQA) |
| Execution device | CPU |
| GPU | Not available for this run |
| Input image | `input_image.jpg` |
| Image dimensions | 1024 × 768 pixels |
| Execution date | 13 November 2026 |

**Input image path recorded in the results:**  
`C:\Users\KIIT\LMM_Vision_Language_Project\data\input_image.jpg`

## 3. Methodology

1. Load the input image using the image-processing library.
2. Load the Llama 3.2 Vision processor and vision-language model.
3. Prepare a natural-language question about the image.
4. Pass the image and question to the model using its chat-template input format.
5. Generate a text response.
6. Record the generated answer and execution time for each question.

## 4. Questions and Actual Model Outputs

### Question 1: Identify the main objects or people

**Question:**

> What are the main objects or people visible in this image?

**Model output:**

> The main object visible in this image is a small brown dog sitting in the grass. The dog is the only object visible in the image.

**Execution time:** 678.74 seconds (approximately 11 minutes 19 seconds)

**Observation:** The model identified a small brown dog as the main visible subject and noted that it was sitting in grass.

### Question 2: Identify the activity or event

**Question:**

> What activity or event is taking place in the image?

**Model output:**

> The image shows a dog sitting in a field of grass. The dog is looking at the camera, and the background is out of focus, but it appears to be a natural setting, possibly a park or a field. There is no activity or event taking place in the image, as it seems to be a still life of a dog in a natural setting.

**Execution time:** 1597.33 seconds (approximately 26 minutes 37 seconds)

**Observation:** The model described the dog as sitting and looking toward the camera. It interpreted the image as a still scene rather than an image showing a distinct activity or event.

### Question 3: Describe the background and surroundings

**Question:**

> Describe the background and surroundings of the image.

**Model output saved in the results file:**

> The image features a small, brown dog sitting in a lush grassy field. The dog, positioned centrally, has a short, scruffy coat with a distinctive white patch on its chest. Its ears are slightly pointed, and its dark brown nose is prominent. A collar adorns its neck, although the tag is not visible. The dog's gaze is directed towards the camera, with its head slightly turned to the left.
>
> The surrounding environment is characterized by a vibrant, green grassy field, which

**Execution time:** 2112.00 seconds (approximately 35 minutes 12 seconds)

**Important note:** The third answer is incomplete in the saved JSON result; it ends mid-sentence. The text above is reproduced only up to the point recorded in that file. No missing text has been inferred or fabricated. Rerun this question or recover the complete model output before presenting the result as a complete answer.

## 5. Execution-Time Summary

| Question | Execution time (seconds) | Approximate time |
|---|---:|---:|
| Question 1 – Main objects | 678.74 | 11 min 19 sec |
| Question 2 – Activity or event | 1597.33 | 26 min 37 sec |
| Question 3 – Background and surroundings | 2112.00 | 35 min 12 sec |
| **Sum of recorded question times** | **4388.07** | **73 min 8 sec** |

The 73 minutes 8 seconds figure is the sum of the three per-question execution times recorded in the results file. It is not a separately measured end-to-end runtime.

## 6. Results and Discussion

The experiment demonstrates that the model can use image content to answer different types of questions:

- **Object identification:** It identified the dog as the main subject.
- **Scene and activity interpretation:** It described the dog sitting in a grassy outdoor setting and concluded that no specific activity was taking place.
- **Background description:** The saved response began describing the grass, the dog's position, and visible appearance, but the recorded answer was truncated.

The first two answers provide usable descriptions of the image. The third result cannot be evaluated as a complete response until the missing output is recovered.

## 7. Limitations

1. **CPU execution:** This run used the CPU because a GPU was unavailable. The recorded generation times were therefore long.
2. **Incomplete third response:** The third answer was truncated in the saved JSON file.
3. **Single-image evaluation:** The experiment used one input image, so it does not establish general accuracy across different image types.
4. **Qualitative evaluation:** The outputs were reviewed descriptively; no formal benchmark or quantitative accuracy score was calculated.

## 8. Conclusion

Experiment 04 tested the Visual Question Answering capability of **Meta Llama 3.2 11B Vision Instruct**. The model successfully generated answers about the main subject and the activity shown in the input image. It also began producing a detailed description of the background, although that response was incomplete in the saved result.

The experiment illustrates how a vision-language model combines image understanding with natural-language questions to generate contextual answers. For a complete final evaluation, the third question should be rerun and its full response saved.

---

**Result status:** Questions 1 and 2 have complete recorded answers. Question 3 has a truncated recorded answer and requires rerunning or recovery before final publication.
